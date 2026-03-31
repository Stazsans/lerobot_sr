"""
Convert a PI0Policy HuggingFace checkpoint (including LoRA) to the .pkl format
required by Pi0RTPolicy (Triton inference engine).

Usage:
    # Standard HF checkpoint:
    python -m lerobot.policies.pi0_rt.convert_from_hf \
        --hf_path /path/to/pretrained_model \
        --output /path/to/pi0_rt.pkl \
        --prompt "Transfer the top disk from the left pillar to the right pillar." \
        --tokenizer_path google/paligemma-3b-pt-224

    # LoRA checkpoint (auto-merges into base model):
    python -m lerobot.policies.pi0_rt.convert_from_hf \
        --hf_path /path/to/lora_checkpoint \
        --output /path/to/pi0_rt.pkl \
        --prompt "Transfer the top disk from the left pillar to the right pillar." \
        --tokenizer_path google/paligemma-3b-pt-224

Notes:
    - The prompt is BAKED INTO the .pkl file via pre-computed language embeddings.
    - You need a separate .pkl for each different task prompt.
    - At inference time, the --task argument must exactly match the prompt used here.
"""

import argparse
import math
import pickle  # nosec
from pathlib import Path

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: str = "cpu",
) -> torch.Tensor:
    dtype = torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def _interleave_head_dim(w: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Convert RoPE head-dim layout: split → interleaved.

    HF Gemma stores Q/K as [..., head_dim] in "split" format:
        [x0, x1, ..., x_{d/2-1}, y0, y1, ..., y_{d/2-1}]
    The Triton kernel expects "interleaved" format:
        [x0, y0, x1, y1, ..., x_{d/2-1}, y_{d/2-1}]

    Args:
        w: Tensor with last dimension = head_dim (may have leading batch dims).
    Returns:
        Tensor with same shape but head_dim reordered to interleaved.
    """
    half = head_dim // 2
    # reshape last dim into (half, 2) where dim-0 = first half, dim-1 = second half
    shape = w.shape[:-1] + (half, 2)
    w_reshaped = w.reshape(shape)
    # w_reshaped[..., 0] = first-half (x), w_reshaped[..., 1] = second-half (y)
    # We want interleaved order, so transpose last two dims then flatten
    w_interleaved = w_reshaped.transpose(-2, -1).reshape(w.shape)
    return w_interleaved


# ---------------------------------------------------------------------------
# Vision encoder (SigLIP, 27 layers)
# ---------------------------------------------------------------------------

def _convert_vision_encoder(pg_model, num_layers: int = 27) -> dict:
    """Extract and convert SigLIP vision encoder weights."""
    vit = pg_model.vision_tower.vision_model
    emb = vit.embeddings

    patch_w = emb.patch_embedding.weight.float()   # (1152, 3, 14, 14)
    patch_b = emb.patch_embedding.bias.float()     # (1152,)
    pos_emb = emb.position_embedding.weight.float()  # (256, 1152)

    # SigLIP patch conv: PyTorch (out, in, kH, kW) → pkl (kH, kW, in, out)
    patch_w_pkl = patch_w.permute(2, 3, 1, 0).contiguous()

    qkv_w_list, qkv_b_list = [], []
    o_w_list, o_b_list = [], []
    ffn_up_w_list, ffn_up_b_list = [], []
    ffn_down_w_list, ffn_down_b_list = [], []
    pre_attn_norm_w_list, pre_attn_norm_b_list = [], []
    pre_ffn_norm_w_list, pre_ffn_norm_b_list = [], []

    for i in range(num_layers):
        layer = vit.encoder.layers[i]
        attn = layer.self_attn
        mlp = layer.mlp

        # No RoPE in SigLIP → no interleaving needed
        # PyTorch weight (out, in) → JAX/pkl (in, out) via .T
        q_w = attn.q_proj.weight.float().T   # (1152, 1152)
        k_w = attn.k_proj.weight.float().T   # (1152, 1152)
        v_w = attn.v_proj.weight.float().T   # (1152, 1152)

        q_b = attn.q_proj.bias.float()   # (1152,)
        k_b = attn.k_proj.bias.float()   # (1152,)
        v_b = attn.v_proj.bias.float()   # (1152,)

        qkv_w_list.append(torch.cat([q_w, k_w, v_w], dim=1))  # (1152, 3*1152)
        qkv_b_list.append(torch.cat([q_b, k_b, v_b], dim=0))  # (3*1152,)

        o_w_list.append(attn.out_proj.weight.float().T)   # (1152, 1152)
        o_b_list.append(attn.out_proj.bias.float())       # (1152,)

        # SigLIP FFN is a standard 2-layer MLP (no gating)
        ffn_up_w_list.append(mlp.fc1.weight.float().T)   # (1152, 4304)
        ffn_up_b_list.append(mlp.fc1.bias.float())        # (4304,)
        ffn_down_w_list.append(mlp.fc2.weight.float().T) # (4304, 1152)
        ffn_down_b_list.append(mlp.fc2.bias.float())      # (1152,)

        pre_attn_norm_w_list.append(layer.layer_norm1.weight.float())
        pre_attn_norm_b_list.append(layer.layer_norm1.bias.float())
        pre_ffn_norm_w_list.append(layer.layer_norm2.weight.float())
        pre_ffn_norm_b_list.append(layer.layer_norm2.bias.float())

    final_norm_w = vit.post_layernorm.weight.float()
    final_norm_b = vit.post_layernorm.bias.float()

    return {
        "vision_patch_embedding_w":   patch_w_pkl.bfloat16(),
        "vision_patch_embedding_b":   patch_b.bfloat16(),
        "vision_position_embedding":  pos_emb.bfloat16(),
        "vision_attn_qkv_w":          torch.stack(qkv_w_list).bfloat16(),
        "vision_attn_qkv_b":          torch.stack(qkv_b_list).bfloat16(),
        "vision_attn_o_w":            torch.stack(o_w_list).bfloat16(),
        "vision_attn_o_b":            torch.stack(o_b_list).bfloat16(),
        "vision_ffn_up_w":            torch.stack(ffn_up_w_list).bfloat16(),
        "vision_ffn_up_b":            torch.stack(ffn_up_b_list).bfloat16(),
        "vision_ffn_down_w":          torch.stack(ffn_down_w_list).bfloat16(),
        "vision_ffn_down_b":          torch.stack(ffn_down_b_list).bfloat16(),
        "vision_pre_attn_norm_w":     torch.stack(pre_attn_norm_w_list).bfloat16(),
        "vision_pre_attn_norm_b":     torch.stack(pre_attn_norm_b_list).bfloat16(),
        "vision_pre_ffn_norm_w":      torch.stack(pre_ffn_norm_w_list).bfloat16(),
        "vision_pre_ffn_norm_b":      torch.stack(pre_ffn_norm_b_list).bfloat16(),
        "vision_final_norm_w":        final_norm_w.bfloat16(),
        "vision_final_norm_b":        final_norm_b.bfloat16(),
    }


# ---------------------------------------------------------------------------
# Multi-modal projector
# ---------------------------------------------------------------------------

def _convert_projector(pg_model) -> dict:
    proj = pg_model.multi_modal_projector.linear
    # PyTorch (2048, 1152) → pkl (1152, 2048)
    w = proj.weight.float().T
    b = proj.bias.float()
    return {
        "encoder_multi_modal_projector_w": w.bfloat16(),
        "encoder_multi_modal_projector_b": b.bfloat16(),
    }


# ---------------------------------------------------------------------------
# Encoder LLM (Gemma 2B, 18 layers)
# ---------------------------------------------------------------------------

def _convert_encoder(pg_model, num_layers: int = 18) -> dict:
    """Convert PaliGemma Gemma-2B LLM weights with RMSNorm fusion."""
    llm = pg_model.language_model.model

    # In HF Gemma, RMSNorm uses (1 + weight) where weight is initialized to 0.
    # The Triton kernel fuses the norm into the QKV/FFN weights directly.

    attn_qkv_w_list = []
    attn_o_w_list = []
    ffn_gate_w_list = []
    ffn_up_w_list = []
    ffn_down_w_list = []

    for i in range(num_layers):
        layer = llm.layers[i]
        attn = layer.self_attn
        mlp = layer.mlp

        pre_attn_norm = layer.input_layernorm.weight.float()          # (2048,)
        pre_ffn_norm = layer.post_attention_layernorm.weight.float()  # (2048,)
        # Effective scale: (1 + weight)
        attn_scale = 1.0 + pre_attn_norm   # (2048,)
        ffn_scale = 1.0 + pre_ffn_norm     # (2048,)

        # Q: PyTorch (2048, 2048) → JAX (2048, 2048) via .T → interleave head_dim
        head_dim = 256
        num_q_heads = 8

        q_w = attn.q_proj.weight.float().T   # (2048, 2048)  [input, output]
        # reshape to (input, num_heads, head_dim) for interleaving
        q_w_heads = q_w.reshape(2048, num_q_heads, head_dim)
        q_w_interleaved = _interleave_head_dim(q_w_heads, head_dim)  # (2048, 8, 256)
        q_w_final = q_w_interleaved.reshape(2048, 2048)              # (2048, 2048)
        # Fuse norm (broadcast over output dim)
        q_w_fused = q_w_final * attn_scale[:, None]                  # (2048, 2048)

        # K: PyTorch (256, 2048) → JAX (2048, 256) via .T → interleave
        k_w = attn.k_proj.weight.float().T   # (2048, 256)
        k_w_interleaved = _interleave_head_dim(k_w.reshape(2048, 1, head_dim), head_dim)
        k_w_final = k_w_interleaved.reshape(2048, head_dim)          # (2048, 256)
        k_w_fused = k_w_final * attn_scale[:, None]                  # (2048, 256)

        # V: PyTorch (256, 2048) → JAX (2048, 256) via .T; no RoPE → no interleave
        v_w = attn.v_proj.weight.float().T                           # (2048, 256)
        v_w_fused = v_w * attn_scale[:, None]                        # (2048, 256)

        # Fused QKV: (2048, 2048+256+256) = (2048, 2560)
        qkv_w = torch.cat([q_w_fused, k_w_fused, v_w_fused], dim=1)
        attn_qkv_w_list.append(qkv_w)

        # O: PyTorch (2048, 2048) → JAX (2048, 2048) via .T; no norm fusion
        o_w = attn.o_proj.weight.float().T                           # (2048, 2048)
        attn_o_w_list.append(o_w)

        # FFN (gated: gate + up → down); fuse pre_ffn_norm into gate and up
        gate_w = mlp.gate_proj.weight.float().T                      # (2048, 16384)
        up_w = mlp.up_proj.weight.float().T                          # (2048, 16384)
        down_w = mlp.down_proj.weight.float().T                      # (16384, 2048)

        gate_w_fused = gate_w * ffn_scale[:, None]                   # (2048, 16384)
        up_w_fused = up_w * ffn_scale[:, None]                       # (2048, 16384)

        ffn_gate_w_list.append(gate_w_fused)
        ffn_up_w_list.append(up_w_fused)
        ffn_down_w_list.append(down_w)

    return {
        "encoder_attn_qkv_w": torch.stack(attn_qkv_w_list).bfloat16(),
        "encoder_attn_o_w":   torch.stack(attn_o_w_list).bfloat16(),
        "encoder_ffn_gate_w": torch.stack(ffn_gate_w_list).bfloat16(),
        "encoder_ffn_up_w":   torch.stack(ffn_up_w_list).bfloat16(),
        "encoder_ffn_down_w": torch.stack(ffn_down_w_list).bfloat16(),
    }


# ---------------------------------------------------------------------------
# Decoder action expert (Gemma 300M, 18 layers)
# ---------------------------------------------------------------------------

def _convert_decoder(expert_model, num_layers: int = 18) -> dict:
    """Convert Gemma-300M action expert weights with RMSNorm fusion."""
    model = expert_model.model

    attn_qkv_w_list = []
    attn_o_w_list = []
    ffn_gate_w_list = []
    ffn_up_w_list = []
    ffn_down_w_list = []

    head_dim = 256
    num_q_heads = 8

    for i in range(num_layers):
        layer = model.layers[i]
        attn = layer.self_attn
        mlp = layer.mlp

        pre_attn_norm = layer.input_layernorm.weight.float()          # (1024,)
        pre_ffn_norm = layer.post_attention_layernorm.weight.float()  # (1024,)
        attn_scale = 1.0 + pre_attn_norm   # (1024,)
        ffn_scale = 1.0 + pre_ffn_norm     # (1024,)

        # Q: PyTorch (2048, 1024) → JAX (1024, 2048) via .T → interleave
        q_w = attn.q_proj.weight.float().T   # (1024, 2048)
        q_w_heads = q_w.reshape(1024, num_q_heads, head_dim)
        q_w_interleaved = _interleave_head_dim(q_w_heads, head_dim)  # (1024, 8, 256)
        q_w_final = q_w_interleaved.reshape(1024, 2048)              # (1024, 2048)
        q_w_fused = q_w_final * attn_scale[:, None]                  # (1024, 2048)

        # K: PyTorch (256, 1024) → JAX (1024, 256) via .T → interleave
        k_w = attn.k_proj.weight.float().T   # (1024, 256)
        k_w_interleaved = _interleave_head_dim(k_w.reshape(1024, 1, head_dim), head_dim)
        k_w_final = k_w_interleaved.reshape(1024, head_dim)          # (1024, 256)
        k_w_fused = k_w_final * attn_scale[:, None]                  # (1024, 256)

        # V: PyTorch (256, 1024) → JAX (1024, 256); no interleave
        v_w = attn.v_proj.weight.float().T                           # (1024, 256)
        v_w_fused = v_w * attn_scale[:, None]                        # (1024, 256)

        # Fused QKV: (1024, 2048+256+256) = (1024, 2560)
        qkv_w = torch.cat([q_w_fused, k_w_fused, v_w_fused], dim=1)
        attn_qkv_w_list.append(qkv_w)

        # O: PyTorch (1024, 2048) → JAX (2048, 1024)
        o_w = attn.o_proj.weight.float().T                           # (2048, 1024)
        attn_o_w_list.append(o_w)

        # FFN
        gate_w = mlp.gate_proj.weight.float().T                      # (1024, 4096)
        up_w = mlp.up_proj.weight.float().T                          # (1024, 4096)
        down_w = mlp.down_proj.weight.float().T                      # (4096, 1024)

        ffn_gate_w_list.append(gate_w * ffn_scale[:, None])
        ffn_up_w_list.append(up_w * ffn_scale[:, None])
        ffn_down_w_list.append(down_w)

    return {
        "decoder_attn_qkv_w": torch.stack(attn_qkv_w_list).bfloat16(),
        "decoder_attn_o_w":   torch.stack(attn_o_w_list).bfloat16(),
        "decoder_ffn_gate_w": torch.stack(ffn_gate_w_list).bfloat16(),
        "decoder_ffn_up_w":   torch.stack(ffn_up_w_list).bfloat16(),
        "decoder_ffn_down_w": torch.stack(ffn_down_w_list).bfloat16(),
    }


# ---------------------------------------------------------------------------
# Action projections (state / action / fused time)
# ---------------------------------------------------------------------------

def _convert_action_projs(pi0_pytorch, final_norm_weight: torch.Tensor) -> dict:
    """Convert action / state projection weights including fused time biases.

    The Triton kernel pre-fuses:
        action_in_proj  ─┐
                          ├─ matmul ─→  decoder_action_fused_in_proj_w
        action_time_mlp_in[:, :action_dim] ─┘

    And bakes the flow-matching time embedding + bias into per-step biases.
    """
    pi0 = pi0_pytorch  # shorthand

    # ── state projection ──────────────────────────────────────────────────
    # Linear(32, 1024): weight (1024, 32) → pkl (32, 1024)
    state_in_w = pi0.state_proj.weight.float().T     # (32, 1024)
    state_in_b = pi0.state_proj.bias.float()          # (1024,)

    # ── action projection (fused) ─────────────────────────────────────────
    # action_in_proj: Linear(32, 1024) → weight (1024, 32)
    action_in_w = pi0.action_in_proj.weight.float().T   # (32, 1024)
    action_in_b = pi0.action_in_proj.bias.float()        # (1024,)

    # action_time_mlp_in: Linear(2048, 1024) → weight (1024, 2048)
    # First 1024 input channels = action, last 1024 = time embedding
    mlp_in_w = pi0.action_time_mlp_in.weight.float()  # (1024, 2048)
    mlp_in_b = pi0.action_time_mlp_in.bias.float()    # (1024,)

    mlp_in_w_action = mlp_in_w[:, :1024]   # (1024, 1024) – action part (PyTorch layout)
    mlp_in_w_time = mlp_in_w[:, 1024:]    # (1024, 1024) – time part

    # Fused action input: (32, 1024) @ (1024, 1024) = (32, 1024)
    # In JAX: fused = action_in_proj_w_jax @ mlp_in_w_action_jax
    #       = action_in_w.T @ mlp_in_w_action.T
    #       which in PyTorch matmul terms is:
    fused_in_proj_w = action_in_w @ mlp_in_w_action.T   # (32, 1024)

    # Per-step time biases (10 steps for 10-step flow matching)
    n_decode_steps = 10
    action_bias_contrib = mlp_in_w_action @ action_in_b   # (1024,)
    time_biases = torch.zeros(n_decode_steps, 1024, dtype=torch.float32)

    for t in range(n_decode_steps):
        time_val = 1.0 - t / n_decode_steps
        time_tensor = torch.tensor([time_val])
        time_emb = _create_sinusoidal_pos_embedding(time_tensor, 1024, 4e-3, 4.0).squeeze(0)
        time_contrib = mlp_in_w_time @ time_emb             # (1024,)
        time_biases[t] = action_bias_contrib + time_contrib + mlp_in_b

    # ── action output projection (fused with final norm and -0.1 scale) ───
    # action_out_proj: Linear(1024, 32) → weight (32, 1024) → pkl (1024, 32)
    out_proj_w = pi0.action_out_proj.weight.float().T   # (1024, 32)
    out_proj_b = pi0.action_out_proj.bias.float()        # (32,)

    # Fuse final norm: effective scale = (1 + weight)
    final_norm = (1.0 + final_norm_weight.float())       # (1024,)
    out_proj_w_fused = out_proj_w * final_norm[:, None]  # (1024, 32)

    # Apply -0.1 scale (matches JAX conversion)
    out_proj_w_fused = out_proj_w_fused * -0.1
    out_proj_b_fused = out_proj_b * -0.1                 # (32,)

    # ── action_time_mlp_out ───────────────────────────────────────────────
    # Linear(1024, 1024): weight (1024, 1024) → pkl (1024, 1024)
    mlp_out_w = pi0.action_time_mlp_out.weight.float().T  # (1024, 1024)
    mlp_out_b = pi0.action_time_mlp_out.bias.float()       # (1024,)

    return {
        "decoder_state_in_proj_w":          state_in_w.bfloat16(),
        "decoder_state_in_proj_b":          state_in_b.bfloat16(),
        "decoder_action_fused_in_proj_w":   fused_in_proj_w.bfloat16(),
        "decoder_action_fused_time_biases": time_biases.bfloat16(),
        "decoder_action_mlp_w":             mlp_out_w.bfloat16(),
        "decoder_action_mlp_b":             mlp_out_b.bfloat16(),
        "decoder_action_fused_out_proj_w":  out_proj_w_fused.bfloat16(),
        "decoder_action_fused_out_proj_b":  out_proj_b_fused.bfloat16(),
    }


# ---------------------------------------------------------------------------
# Language embeddings (prompt baked into pkl)
# ---------------------------------------------------------------------------

def _compute_language_embeds(
    embed_tokens_weight: torch.Tensor,
    prompt: str,
    tokenizer_path: str,
    max_length: int = 48,
) -> torch.Tensor:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompt_clean = prompt.strip().replace("_", " ") + "\n"
    tokens = tokenizer(
        [prompt_clean],
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"].squeeze(0)  # (seq_len,)

    embed_w = embed_tokens_weight.float().cuda()   # (vocab_size, 2048)
    embeds = F.embedding(tokens.cuda(), embed_w)   # (seq_len, 2048)
    # PaliGemma scales embeddings by sqrt(hidden_dim)
    embeds = embeds * (embeds.shape[-1] ** 0.5)
    return embeds.cpu().bfloat16()


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    hf_path: str,
    output: str,
    prompt: str,
    tokenizer_path: str,
) -> None:
    print(f"Loading model from: {hf_path}")

    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    policy = PI0Policy.from_pretrained(hf_path)
    policy.eval()

    # If the checkpoint contains LoRA adapters, merge them into the base model
    try:
        from peft import PeftModel
        if isinstance(policy, PeftModel) or hasattr(policy, "peft_config"):
            print("LoRA adapters detected — merging into base model …")
            policy = policy.merge_and_unload()
            print("Merge complete.")
    except ImportError:
        pass  # peft not installed, assume base model

    pi0 = policy.model  # PI0Pytorch
    pge = pi0.paligemma_with_expert  # PaliGemmaWithExpertModel
    pg = pge.paligemma  # PaliGemmaForConditionalGeneration

    print("Converting vision encoder …")
    weights = _convert_vision_encoder(pg)

    print("Converting multi-modal projector …")
    weights.update(_convert_projector(pg))

    print("Converting encoder (Gemma 2B) …")
    weights.update(_convert_encoder(pg))

    print("Converting decoder / action expert (Gemma 300M) …")
    weights.update(_convert_decoder(pge.gemma_expert))

    print("Converting action projections …")
    final_norm_w = pge.gemma_expert.model.norm.weight  # (1024,)
    weights.update(_convert_action_projs(pi0, final_norm_w))

    print("Computing language embeddings …")
    embed_w = pg.language_model.model.embed_tokens.weight  # (257152, 2048)
    weights["language_embeds"] = _compute_language_embeds(embed_w, prompt, tokenizer_path)
    print(f"  prompt tokens: {weights['language_embeds'].shape[0]}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(weights, f)  # nosec

    # Quick shape check
    expected_shapes = {
        "vision_patch_embedding_w":         (14, 14, 3, 1152),
        "vision_attn_qkv_w":                (27, 1152, 3 * 1152),
        "encoder_attn_qkv_w":               (18, 2048, 2560),
        "decoder_attn_qkv_w":               (18, 1024, 2560),
        "decoder_action_fused_in_proj_w":   (32, 1024),
        "decoder_action_fused_time_biases": (10, 1024),
        "decoder_action_fused_out_proj_w":  (1024, 32),
    }
    all_ok = True
    for key, expected in expected_shapes.items():
        actual = tuple(weights[key].shape)
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_ok = False
        print(f"  {status} {key}: {actual}")

    if all_ok:
        print("\nConversion successful!")
    else:
        print("\nWARNING: Some shapes do not match — check your model architecture.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PI0Policy HF checkpoint to pi0_rt .pkl")
    parser.add_argument("--hf_path", required=True, help="Path to HF checkpoint directory")
    parser.add_argument("--output", required=True, help="Output .pkl file path")
    parser.add_argument("--prompt", required=True, help="Task prompt (baked into pkl)")
    parser.add_argument(
        "--tokenizer_path",
        default="google/paligemma-3b-pt-224",
        help="PaliGemma tokenizer path (local dir or HF Hub id)",
    )
    args = parser.parse_args()
    convert(args.hf_path, args.output, args.prompt, args.tokenizer_path)
