"""已有数据集的离线畸变矫正脚本。

对数据集中的视频文件逐帧解码、矫正、重新编码。

Usage:
    python scripts/undistort_dataset.py \
        --dataset-path /path/to/dataset \
        --calibrations camera1=cam1_calib.json camera2=cam2_calib.json

    # 指定编码参数
    python scripts/undistort_dataset.py \
        --dataset-path /path/to/dataset \
        --calibrations camera1=cam1_calib.json \
        --vcodec libsvtav1 --crf 30
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from lerobot.cameras.opencv.calibrate import compute_undistort_maps, load_calibration
from lerobot.datasets.video_utils import encode_video_frames


def undistort_video(video_path, map1, map2, output_path, vcodec="libsvtav1", crf=30, g=2):
    """解码视频 → 逐帧矫正 → 重新编码。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        container = av.open(str(video_path))
        video_stream = container.streams.video[0]
        fps = int(video_stream.average_rate)

        frame_idx = 0
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            # cv2.remap expects BGR or any format, works with RGB too
            img_undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            img_path = tmpdir / f"frame-{frame_idx:06d}.png"
            Image.fromarray(img_undistorted).save(str(img_path))
            frame_idx += 1

        container.close()

        if frame_idx == 0:
            print(f"  Warning: no frames decoded from {video_path}")
            return

        encode_video_frames(
            tmpdir, output_path, fps=fps, vcodec=vcodec, crf=crf, g=g, overwrite=True
        )
        print(f"  Processed {frame_idx} frames, fps={fps}")


def main():
    parser = argparse.ArgumentParser(description="Undistort dataset videos offline")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset root directory")
    parser.add_argument(
        "--calibrations", nargs="+", required=True,
        help="Camera calibrations: camera_name=calibration_file.json (camera_name matches video key)",
    )
    parser.add_argument("--no-backup", action="store_true", help="Skip backing up original videos")
    parser.add_argument("--vcodec", default="libsvtav1", help="Video codec (default: libsvtav1)")
    parser.add_argument("--crf", type=int, default=30, help="CRF quality (default: 30)")
    parser.add_argument("--g", type=int, default=2, help="GOP size / keyframe interval (default: 2)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    videos_dir = dataset_path / "videos"

    if not videos_dir.exists():
        print(f"Error: {videos_dir} does not exist")
        return

    # 解析标定文件映射
    calib_map = {}
    for spec in args.calibrations:
        if "=" not in spec:
            print(f"Error: invalid calibration spec '{spec}', expected format: camera_name=file.json")
            return
        name, path = spec.split("=", 1)
        calib = load_calibration(path)
        calib_map[name] = calib
        print(f"Loaded calibration for '{name}': model={calib['model_type']}, "
              f"size={calib['image_size']}, error={calib['reprojection_error']:.4f}")

    # 遍历视频目录
    processed = 0
    for video_key_dir in sorted(videos_dir.iterdir()):
        if not video_key_dir.is_dir():
            continue

        video_key = video_key_dir.name

        # 匹配标定文件
        matched_calib = None
        matched_name = None
        for cam_name, calib in calib_map.items():
            if cam_name in video_key:
                matched_calib = calib
                matched_name = cam_name
                break

        if matched_calib is None:
            print(f"Skipping {video_key}: no matching calibration")
            continue

        # 获取视频分辨率并计算映射表
        mp4_files = sorted(video_key_dir.rglob("*.mp4"))
        if not mp4_files:
            print(f"Skipping {video_key}: no MP4 files found")
            continue

        # 从第一个视频获取实际分辨率
        probe_container = av.open(str(mp4_files[0]))
        probe_stream = probe_container.streams.video[0]
        video_width = probe_stream.width
        video_height = probe_stream.height
        probe_container.close()

        image_size = (video_width, video_height)
        map1, map2 = compute_undistort_maps(matched_calib, image_size)
        print(f"\n{video_key} -> calibration '{matched_name}' (resolution: {video_width}x{video_height})")

        for mp4_file in mp4_files:
            rel_path = mp4_file.relative_to(dataset_path)
            print(f"  Processing: {rel_path}")

            # 备份
            if not args.no_backup:
                backup_path = mp4_file.with_suffix(".mp4.bak")
                if not backup_path.exists():
                    shutil.copy2(mp4_file, backup_path)
                    print(f"  Backed up to: {backup_path.name}")

            # 矫正到临时文件，然后替换
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            undistort_video(
                mp4_file, map1, map2, tmp_path,
                vcodec=args.vcodec, crf=args.crf, g=args.g,
            )
            shutil.move(str(tmp_path), str(mp4_file))
            processed += 1

    print(f"\nDone! Processed {processed} video files.")
    if not args.no_backup:
        print("Original videos backed up with .mp4.bak extension.")
        print("To remove backups: find <dataset_path>/videos -name '*.bak' -delete")


if __name__ == "__main__":
    main()
