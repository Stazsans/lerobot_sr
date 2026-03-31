import os
import sys
import math
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _posemb_sincos(time_val, dim, device="cpu"):
    dtype = torch.float64
    time = torch.tensor([time_val], dtype=dtype, device=device)
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=dtype, device=device)
    min_period = 4e-3
    max_period = 4.0
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).squeeze(0)

def _to_interleaved(tensor, num_heads, head_dim):
    shape = tensor.shape
    tensor = tensor.view(*shape[:-1], num_heads, 2, head_dim // 2)
    tensor = tensor.transpose(-1, -2).reshape(shape)
    return tensor.contiguous()

def convert_weights(weights, model, device="cuda"):
    config = model.config
    diffusion_steps = config.diffusion_steps
    num_layers = config.action_num_layers
    hidden = config.action_hidden_size
    num_q_heads = config.action_num_heads
    num_kv_heads = config.action_num_kv_heads
    head_dim = config.action_head_dim
    dt = -1.0 / diffusion_steps
    action_in = model.model.action_in_proj
    time_mlp_in = model.model.action_time_mlp_in
    w_action = time_mlp_in.weight[:, :hidden]
    w_time = time_mlp_in.weight[:, hidden:]
    fused_action_weight = w_action @ action_in.weight
    fused_action_bias = time_mlp_in.bias.clone()
    if action_in.bias is not None:
        fused_action_bias = fused_action_bias + F.linear(action_in.bias, w_action, None)

    weights['decoder_action_fused_in_proj_w'].copy_(
        fused_action_weight.data.float().T.contiguous().to(torch.bfloat16).to(device))
    fused_time_w_cpu = w_time.data.float().cpu()
    fused_action_bias_cpu = fused_action_bias.data.float().cpu()
    time_biases = torch.zeros(diffusion_steps, hidden, dtype=torch.float32)
    for step in range(diffusion_steps):
        time_val = 1.0 - step / diffusion_steps
        time_emb = _posemb_sincos(time_val, hidden, device="cpu").float()
        time_proj = F.linear(time_emb, fused_time_w_cpu)
        time_biases[step] = fused_action_bias_cpu + time_proj
    weights['decoder_action_fused_time_biases'].copy_(time_biases.to(torch.bfloat16).to(device))
    weights['decoder_action_mlp_w'].copy_(
        model.model.action_time_mlp_out.weight.data.float().T.contiguous().to(torch.bfloat16).to(device))
    weights['decoder_action_mlp_b'].copy_(
        model.model.action_time_mlp_out.bias.data.float().to(torch.bfloat16).to(device))
    final_norm_w = model.model.action_expert.model.norm.weight.data.float()
    out_proj_w = model.model.action_out_proj.weight.data.float().T.contiguous()
    out_proj_b = model.model.action_out_proj.bias.data.float()
    weights['decoder_action_fused_out_proj_w'].copy_(
        (out_proj_w * final_norm_w[:, None] * dt).to(torch.bfloat16).to(device))
    weights['decoder_action_fused_out_proj_b'].copy_(
        (out_proj_b * dt).to(torch.bfloat16).to(device))
    decoder_attn_qkv_w, decoder_q_norm_w, decoder_k_norm_w = [], [], []
    decoder_attn_o_w = []
    decoder_ffn_gate_w, decoder_ffn_up_w, decoder_ffn_down_w = [], [], []

    for i in range(num_layers):
        layer = model.model.action_expert.model.layers[i]
        input_norm_w = layer.input_layernorm.weight.data.float()
        q_w = layer.self_attn.q_proj.weight.data.float().T.contiguous() * input_norm_w[:, None]
        k_w = layer.self_attn.k_proj.weight.data.float().T.contiguous() * input_norm_w[:, None]
        v_w = layer.self_attn.v_proj.weight.data.float().T.contiguous() * input_norm_w[:, None]
        q_w = _to_interleaved(q_w, num_q_heads, head_dim)
        k_w = _to_interleaved(k_w, num_kv_heads, head_dim)
        decoder_attn_qkv_w.append(torch.cat([q_w, k_w, v_w], dim=1).to(torch.bfloat16).to(device))
        q_norm = layer.self_attn.q_norm.weight.data.float()
        k_norm = layer.self_attn.k_norm.weight.data.float()
        decoder_q_norm_w.append(q_norm.view(2, head_dim // 2).T.reshape(head_dim).to(torch.bfloat16).to(device))
        decoder_k_norm_w.append(k_norm.view(2, head_dim // 2).T.reshape(head_dim).to(torch.bfloat16).to(device))

        decoder_attn_o_w.append(layer.self_attn.o_proj.weight.data.float().T.contiguous().to(torch.bfloat16).to(device))
        post_norm_w = layer.post_attention_layernorm.weight.data.float()
        gate_w = layer.mlp.gate_proj.weight.data.float().T.contiguous() * post_norm_w[:, None]
        up_w = layer.mlp.up_proj.weight.data.float().T.contiguous() * post_norm_w[:, None]
        decoder_ffn_gate_w.append(gate_w.to(torch.bfloat16).to(device))
        decoder_ffn_up_w.append(up_w.to(torch.bfloat16).to(device))
        decoder_ffn_down_w.append(layer.mlp.down_proj.weight.data.float().T.contiguous().to(torch.bfloat16).to(device))

    weights['decoder_attn_qkv_w'].copy_(torch.stack(decoder_attn_qkv_w))
    weights['decoder_q_norm_w'].copy_(torch.stack(decoder_q_norm_w))
    weights['decoder_k_norm_w'].copy_(torch.stack(decoder_k_norm_w))
    weights['decoder_attn_o_w'].copy_(torch.stack(decoder_attn_o_w))
    weights['decoder_ffn_gate_w'].copy_(torch.stack(decoder_ffn_gate_w))
    weights['decoder_ffn_up_w'].copy_(torch.stack(decoder_ffn_up_w))
    weights['decoder_ffn_down_w'].copy_(torch.stack(decoder_ffn_down_w))
    llm_hidden = config.llm_hidden_size
    llm_num_layers = config.llm_num_layers
    llm_num_q_heads = config.llm_num_heads
    llm_num_kv_heads = config.llm_num_kv_heads
    llm_head_dim = config.llm_head_dim

    llm_attn_qkv_w, llm_q_norm_w, llm_k_norm_w = [], [], []
    llm_attn_o_w = []
    llm_ffn_gate_w, llm_ffn_up_w, llm_ffn_down_w = [], [], []

    for i in range(llm_num_layers):
        layer = model.model.llm.layers[i]
        input_norm_w = layer.input_layernorm.weight.data.float()

        q_w = layer.self_attn.q_proj.weight.data.float().T.contiguous() * input_norm_w[:, None]
        k_w = layer.self_attn.k_proj.weight.data.float().T.contiguous() * input_norm_w[:, None]
        v_w = layer.self_attn.v_proj.weight.data.float().T.contiguous() * input_norm_w[:, None]
        q_w = _to_interleaved(q_w, llm_num_q_heads, llm_head_dim)
        k_w = _to_interleaved(k_w, llm_num_kv_heads, llm_head_dim)
        llm_attn_qkv_w.append(torch.cat([q_w, k_w, v_w], dim=1).to(torch.bfloat16).to(device))

        q_norm = layer.self_attn.q_norm.weight.data.float()
        k_norm = layer.self_attn.k_norm.weight.data.float()
        llm_q_norm_w.append(q_norm.view(2, llm_head_dim // 2).T.reshape(llm_head_dim).to(torch.bfloat16).to(device))
        llm_k_norm_w.append(k_norm.view(2, llm_head_dim // 2).T.reshape(llm_head_dim).to(torch.bfloat16).to(device))

        llm_attn_o_w.append(layer.self_attn.o_proj.weight.data.float().T.contiguous().to(torch.bfloat16).to(device))

        post_norm_w = layer.post_attention_layernorm.weight.data.float()
        gate_w = layer.mlp.gate_proj.weight.data.float().T.contiguous() * post_norm_w[:, None]
        up_w = layer.mlp.up_proj.weight.data.float().T.contiguous() * post_norm_w[:, None]
        llm_ffn_gate_w.append(gate_w.to(torch.bfloat16).to(device))
        llm_ffn_up_w.append(up_w.to(torch.bfloat16).to(device))
        llm_ffn_down_w.append(layer.mlp.down_proj.weight.data.float().T.contiguous().to(torch.bfloat16).to(device))

    weights['llm_attn_qkv_w'].copy_(torch.stack(llm_attn_qkv_w))
    weights['llm_q_norm_w'].copy_(torch.stack(llm_q_norm_w))
    weights['llm_k_norm_w'].copy_(torch.stack(llm_k_norm_w))
    weights['llm_attn_o_w'].copy_(torch.stack(llm_attn_o_w))
    weights['llm_ffn_gate_w'].copy_(torch.stack(llm_ffn_gate_w))
    weights['llm_ffn_up_w'].copy_(torch.stack(llm_ffn_up_w))
    weights['llm_ffn_down_w'].copy_(torch.stack(llm_ffn_down_w))
    sd = model.state_dict()
    vp = 'model.mm_vision_tower.vision_tower'
    weights['vision_conv1_w_t'].copy_(
        sd[f'{vp}.conv1.weight'].float().reshape(1024, -1).T.contiguous().to(torch.bfloat16).to(device))
    weights['vision_class_embedding'].copy_(sd[f'{vp}.class_embedding'].to(torch.bfloat16).to(device))
    weights['vision_pos_emb'].copy_(sd[f'{vp}.positional_embedding'].to(torch.bfloat16).to(device))
    weights['vision_ln_pre_w'].copy_(sd[f'{vp}.ln_pre.weight'].to(torch.bfloat16).to(device))
    weights['vision_ln_pre_b'].copy_(sd[f'{vp}.ln_pre.bias'].to(torch.bfloat16).to(device))
    v_fused_qkv_w, v_fused_qkv_b, v_qkv_col_sum = [], [], []
    v_out_proj_w, v_out_proj_b = [], []
    v_fused_fc_w, v_fused_fc_b, v_fc_col_sum = [], [], []
    v_proj_w, v_proj_b = [], []

    for i in range(23):
        bp = f'{vp}.transformer.resblocks.{i}'
        ln1_w = sd[f'{bp}.ln_1.weight'].float()
        ln1_b = sd[f'{bp}.ln_1.bias'].float()
        in_proj_w = sd[f'{bp}.attn.in_proj_weight'].float().T.contiguous()
        in_proj_b = sd[f'{bp}.attn.in_proj_bias'].float()
        qkv_fused = ln1_w[:, None] * in_proj_w
        v_fused_qkv_w.append(qkv_fused.to(torch.bfloat16).to(device))
        v_fused_qkv_b.append((torch.matmul(ln1_b, in_proj_w) + in_proj_b).to(torch.float32).to(device))
        v_qkv_col_sum.append(qkv_fused.sum(dim=0).to(torch.float32).to(device))

        ls1 = sd[f'{bp}.ls_1.gamma'].float()
        ow = sd[f'{bp}.attn.out_proj.weight'].float().T.contiguous() * ls1
        ob = sd[f'{bp}.attn.out_proj.bias'].float() * ls1
        v_out_proj_w.append(ow.to(torch.bfloat16).to(device))
        v_out_proj_b.append(ob.to(torch.bfloat16).to(device))

        ln2_w = sd[f'{bp}.ln_2.weight'].float()
        ln2_b = sd[f'{bp}.ln_2.bias'].float()
        fc_w = sd[f'{bp}.mlp.c_fc.weight'].float().T.contiguous()
        fc_b = sd[f'{bp}.mlp.c_fc.bias'].float()
        fc_fused = ln2_w[:, None] * fc_w
        v_fused_fc_w.append(fc_fused.to(torch.bfloat16).to(device))
        v_fused_fc_b.append((torch.matmul(ln2_b, fc_w) + fc_b).to(torch.float32).to(device))
        v_fc_col_sum.append(fc_fused.sum(dim=0).to(torch.float32).to(device))

        ls2 = sd[f'{bp}.ls_2.gamma'].float()
        pw = sd[f'{bp}.mlp.c_proj.weight'].float().T.contiguous() * ls2
        pb = sd[f'{bp}.mlp.c_proj.bias'].float() * ls2
        v_proj_w.append(pw.to(torch.bfloat16).to(device))
        v_proj_b.append(pb.to(torch.bfloat16).to(device))

    weights['vision_fused_qkv_w'].copy_(torch.stack(v_fused_qkv_w))
    weights['vision_fused_qkv_b'].copy_(torch.stack(v_fused_qkv_b))
    weights['vision_qkv_col_sum'].copy_(torch.stack(v_qkv_col_sum))
    weights['vision_out_proj_w'].copy_(torch.stack(v_out_proj_w))
    weights['vision_out_proj_b'].copy_(torch.stack(v_out_proj_b))
    weights['vision_fused_fc_w'].copy_(torch.stack(v_fused_fc_w))
    weights['vision_fused_fc_b'].copy_(torch.stack(v_fused_fc_b))
    weights['vision_fc_col_sum'].copy_(torch.stack(v_fc_col_sum))
    weights['vision_proj_w'].copy_(torch.stack(v_proj_w))
    weights['vision_proj_b'].copy_(torch.stack(v_proj_b))
    weights['vision_ds1_w'].copy_(
        sd[f'{vp}.vit_downsampler1.weight'].permute(2, 3, 1, 0).contiguous().to(torch.bfloat16).to(device))
    weights['vision_ds1_b'].copy_(
        sd[f'{vp}.vit_downsampler1.bias'].to(torch.bfloat16).to(device))
    weights['vision_ds2_w'].copy_(
        sd[f'{vp}.vit_downsampler2.weight'].permute(2, 3, 1, 0).contiguous().to(torch.bfloat16).to(device))
    weights['vision_ds2_b'].copy_(
        sd[f'{vp}.vit_downsampler2.bias'].to(torch.bfloat16).to(device))
    weights['vision_projector_w_t'].copy_(
        sd['model.mm_projector.weight'].T.contiguous().to(torch.bfloat16).to(device))
    weights['vision_embed_tokens_w'].copy_(
        sd['model.llm.embed_tokens.weight'].to(torch.bfloat16).to(device))

def load_dm0_model(model_path, device="cuda"):
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from modeling_dm0_init import DM0ForCausalLM, DB0Config

    config = DB0Config()
    try:
        pretrained_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        for key in vars(config).keys():
            if hasattr(pretrained_config, key):
                setattr(config, key, getattr(pretrained_config, key))
        print(f"Synced config from pretrained: {model_path}")
    except Exception as e:
        print(f"Warning: Could not load pretrained config: {e}")

    config.bf16 = True
    model = DM0ForCausalLM(config)

    state_dict = {}
    for i in range(1, 10):
        sf_path = os.path.join(model_path, f"model-0000{i}-of-00002.safetensors")
        if os.path.exists(sf_path):
            print(f"Loading {os.path.basename(sf_path)}")
            state_dict.update(load_file(sf_path))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if "lm_head.weight" in missing:
        model.lm_head.weight = model.model.llm.embed_tokens.weight

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model

def create_weights_dict(config, device="cuda"):
    ds1_out_c = 2048
    ds2_out_c = 4096
    return {
        'decoder_attn_qkv_w':        torch.empty(28, 1024, 4096, dtype=torch.bfloat16, device=device),
        'decoder_q_norm_w':          torch.empty(28, 128,              dtype=torch.bfloat16, device=device),
        'decoder_k_norm_w':          torch.empty(28, 128,              dtype=torch.bfloat16, device=device),
        'decoder_attn_o_w':          torch.empty(28, 2048, 1024,       dtype=torch.bfloat16, device=device),
        'decoder_ffn_gate_w':        torch.empty(28, 1024, 1536,       dtype=torch.bfloat16, device=device),
        'decoder_ffn_up_w':          torch.empty(28, 1024, 1536,       dtype=torch.bfloat16, device=device),
        'decoder_ffn_down_w':        torch.empty(28, 1536, 1024,       dtype=torch.bfloat16, device=device),
        'decoder_action_fused_in_proj_w': torch.empty(32, 1024,         dtype=torch.bfloat16, device=device),
        'decoder_action_fused_time_biases': torch.empty(10, 1024,      dtype=torch.bfloat16, device=device),
        'decoder_action_mlp_w':       torch.empty(1024, 1024,           dtype=torch.bfloat16, device=device),
        'decoder_action_mlp_b':       torch.empty(1024,                 dtype=torch.bfloat16, device=device),
        'decoder_action_fused_out_proj_w': torch.empty(1024, 32,        dtype=torch.bfloat16, device=device),
        'decoder_action_fused_out_proj_b': torch.empty(32,               dtype=torch.bfloat16, device=device),
        'llm_attn_qkv_w':            torch.empty(28, 2048, 4096,       dtype=torch.bfloat16, device=device),
        'llm_q_norm_w':              torch.empty(28, 128,               dtype=torch.bfloat16, device=device),
        'llm_k_norm_w':              torch.empty(28, 128,               dtype=torch.bfloat16, device=device),
        'llm_attn_o_w':              torch.empty(28, 2048, 2048,        dtype=torch.bfloat16, device=device),
        'llm_ffn_gate_w':            torch.empty(28, 2048, 6144,        dtype=torch.bfloat16, device=device),
        'llm_ffn_up_w':              torch.empty(28, 2048, 6144,        dtype=torch.bfloat16, device=device),
        'llm_ffn_down_w':            torch.empty(28, 6144, 2048,       dtype=torch.bfloat16, device=device),
        'vision_conv1_w_t':          torch.empty(588, 1024,            dtype=torch.bfloat16, device=device),
        'vision_class_embedding':     torch.empty(1024,                dtype=torch.bfloat16, device=device),
        'vision_pos_emb':             torch.empty(2705, 1024,            dtype=torch.bfloat16, device=device),
        'vision_ln_pre_w':            torch.empty(1024,                dtype=torch.bfloat16, device=device),
        'vision_ln_pre_b':            torch.empty(1024,                dtype=torch.bfloat16, device=device),
        'vision_fused_qkv_w':        torch.empty(23, 1024, 3072,       dtype=torch.bfloat16, device=device),
        'vision_fused_qkv_b':        torch.empty(23, 3072,             dtype=torch.float32, device=device),
        'vision_qkv_col_sum':        torch.empty(23, 3072,             dtype=torch.float32, device=device),
        'vision_out_proj_w':         torch.empty(23, 1024, 1024,         dtype=torch.bfloat16, device=device),
        'vision_out_proj_b':          torch.empty(23, 1024,             dtype=torch.bfloat16, device=device),
        'vision_fused_fc_w':         torch.empty(23, 1024, 4096,       dtype=torch.bfloat16, device=device),
        'vision_fused_fc_b':         torch.empty(23, 4096,              dtype=torch.float32, device=device),
        'vision_fc_col_sum':          torch.empty(23, 4096,              dtype=torch.float32, device=device),
        'vision_proj_w':              torch.empty(23, 4096, 1024,        dtype=torch.bfloat16, device=device),
        'vision_proj_b':              torch.empty(23, 1024,              dtype=torch.bfloat16, device=device),
        'vision_ds1_w':               torch.empty(3, 3, 1024, ds1_out_c, dtype=torch.bfloat16, device=device),
        'vision_ds1_b':               torch.empty(ds1_out_c,            dtype=torch.bfloat16, device=device),
        'vision_ds2_w':               torch.empty(3, 3, ds1_out_c, ds2_out_c, dtype=torch.bfloat16, device=device),
        'vision_ds2_b':               torch.empty(ds2_out_c,            dtype=torch.bfloat16, device=device),
        'vision_projector_w_t':       torch.empty(4096, 2048,           dtype=torch.bfloat16, device=device),
        'vision_embed_tokens_w':      torch.empty(config.llm_vocab_size, 2048, dtype=torch.bfloat16, device=device),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="dm0_triton_weights.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model = load_dm0_model(args.model_path, device=args.device)
    weights = create_weights_dict(model.config, device=args.device)
    convert_weights(weights, model, device=args.device)
    torch.save(weights, args.output)
    print(f"\nSaved to {args.output}")
    total = sum(t.numel() for t in weights.values())
    print(f"Total: {total/1e6:.1f}M params")
