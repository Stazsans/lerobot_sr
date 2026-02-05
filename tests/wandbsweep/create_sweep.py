import wandb
import time
from pathlib import Path

# 全用相对路径，方便服务器和本地跑同一份代码
current_file = Path(__file__).resolve()

PROJECT_ROOT = current_file.parent.parent.parent

HOME_DIR = Path.home()

PROGRAM_PATH = str(PROJECT_ROOT / "src/lerobot/scripts/lerobot_train.py")
DATASET_PATH = str(HOME_DIR / "datasets/hanoi_smallleft2middle")

# 生成一个时间戳，用于 wandb 命名
unique_tag = f"act_{time.strftime('%m%d_%H%M')}"

sweep_config = {
    "name": "lerobot_act_sweep",
    "program":PROGRAM_PATH,
    "method": "bayes",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters":{
        "batch_size":{"values": [8, 16, 32]},
        "optimizer.lr":{"distribution": "log_uniform_values", "max": 1e-4, "min": 1e-6},
        "policy.chunk_size":{"distribution": "q_uniform","max": 200, "min": 50,"q": 10},
        "policy.kl_weight":{"distribution": "log_uniform_values","min": 1,"max": 20},
        "policy.temporal_ensemble_coeff":{"min":0.005,"max":0.05},
#        "wandb.job_name": {"values": [unique_tag]},

    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 1,     # Agent 会先跑 min_iter 个单位，一次 log 是一个单位（现在是 2000 个 step）
        "eta": 2,          # 3 个里面淘汰一个
        },
    "command": [
        "${interpreter}",
        "${program}",
 #       "--batch_size=32",
        f"--dataset.root={DATASET_PATH}",
        "--dataset.repo_id=sr/hanoi",
        "--policy.type=act",
        "--policy.device=cuda",
        "--wandb.enable=true",
        "--policy.push_to_hub=false",
        "--steps=200", # 测试一下，之后改回 10000
 #       "--policy.chunk_size=50",
 #       "--policy.temporal_ensemble_coeff=0.01",
        "--policy.n_action_steps=1",
        "--save_freq=200",  # 测试一下，之后改回 2000
        "--dataset.video_backend=pyav",
        "--dataset.image_transforms.enable=true",
        "--dataset.image_transforms.max_num_transforms=2",
        "--policy.use_amp=true",
        "${args}", 
    ],
}
sweep_id = wandb.sweep(sweep_config, project="sweep")