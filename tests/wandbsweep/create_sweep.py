import wandb

sweep_config = {
    "name": "lerobot_act_sweep",
    "program":"/home/sr/PycharmProjects/lerobot_sr/src/lerobot/scripts/lerobot_train.py",
    "method": "bayes",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters":{
        "batch_size":{"values": [8, 16, 32]},
        "optimizer.lr":{"max": 1e-4, "min": 1e-6},
        "policy.chunk_size":{"max": 200, "min": 50}
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 2,     # Agent 会先跑 min_iter 个单位，一次 log 是一个单位（现在是 2000 个 step）
        "eta": 2,          # 每次淘汰一半
        },
    "command": [
        "${interpreter}",
        "${program}",
        "--dataset.root=/home/sr/datasets/hanoi_smallleft2middle",
        "--dataset.repo_id=sr/hanoi",
        "--policy.type=act",
        "--policy.device=cuda",
        "--wandb.enable=true",
        "--policy.push_to_hub=false",
        "--steps=10000",
 #       "--policy.chunk_size=50",
        "--policy.temporal_ensemble_coeff=0.01",
        "--policy.n_action_steps=1",
        "--save_freq=2000",
        "--dataset.video_backend=pyav",
        "--policy.use_amp=true",
        "--output_dir=outputs/train/hanoi_sweep/${wandb_run_id}",
        "${args}", 
    ],
}
sweep_id = wandb.sweep(sweep_config, project="lerobot")