#!/usr/bin/env python
import os
import subprocess

DATASET_BASE = "/home/sr/datasets/0326act"
FEISHU_HOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/859cf906-1681-471e-9985-ea915d9f7f8c"


def feishu_notify(msg: str):
    subprocess.run(
        [
            "curl", "-X", "POST",
            "-H", "Content-Type: application/json",
            "-d", f'{{"msg_type":"text","content":{{"text":"{msg}"}}}}',
            FEISHU_HOOK,
        ],
        check=False,
    )


def main():
    folders = sorted([
        f for f in os.listdir(DATASET_BASE)
        if os.path.isdir(os.path.join(DATASET_BASE, f))
    ])

    for folder in folders:
        print(f"========== Training on: {folder} ==========")

        ret = subprocess.run([
            "lerobot-train",
            f"--dataset.root={DATASET_BASE}/{folder}",
            "--dataset.repo_id=sr/hanoi",
            "--policy.type=act",
            f"--output_dir=outputs/train/0326act/{folder}",
            f"--job_name={folder}",
            "--policy.device=cuda",
            "--wandb.enable=true",
            "--policy.repo_id=sr/act_policy",
            "--policy.push_to_hub=false",
            "--batch_size=32",
            "--steps=16000",
            "--policy.chunk_size=50",
            "--policy.n_action_steps=30",
            "--save_freq=2000",
            "--dataset.video_backend=pyav",
            "--policy.use_amp=true",
            "--policy.kl_weight=16",
            "--policy.dropout=0.05",
            "--dataset.image_transforms.enable=true",
            "--dataset.image_transforms.max_num_transforms=3",
        ])

        if ret.returncode == 0:
            feishu_notify(f"📢📢📢好消息！ACT {folder} 练完了")
        else:
            feishu_notify(f"❌❌❌ ACT {folder} 训练失败，退出码 {ret.returncode}")
            print(f"Training failed for {folder}, stopping.")
            break

        print(f"========== Done: {folder} ==========")

    else:
        feishu_notify("🎉🎉🎉所有任务全部训练完成！")


if __name__ == "__main__":
    main()
