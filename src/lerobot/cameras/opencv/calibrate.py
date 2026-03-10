"""摄像头畸变标定工具。支持普通镜头和鱼眼镜头的标定与矫正。

Usage:
    # 1. 生成标定板图像（打印后使用）
    python -m lerobot.cameras.opencv.calibrate generate-board

    # 2. 交互式标定（需连接摄像头）
    python -m lerobot.cameras.opencv.calibrate calibrate --camera-index 3 --output cam1_calib.json

    # 3. 验证标定结果
    python -m lerobot.cameras.opencv.calibrate verify --calibration cam1_calib.json --camera-index 3
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ChArUco 标定板默认参数
DEFAULT_SQUARES_X = 7
DEFAULT_SQUARES_Y = 9
DEFAULT_SQUARE_LENGTH_MM = 27.0
DEFAULT_MARKER_LENGTH_MM = 20.0
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50


def _create_board(squares_x, squares_y, square_length_mm, marker_length_mm):
    """创建 ChArUco 标定板对象。"""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_mm / 1000.0,
        marker_length_mm / 1000.0,
        dictionary,
    )
    return board, dictionary


def _get_board_corners_3d(board):
    """获取标定板所有内角点的 3D 坐标。"""
    try:
        return board.getChessboardCorners()
    except AttributeError:
        return board.chessboardCorners


def _detect_charuco(image, board, dictionary):
    """检测图像中的 ChArUco 角点。兼容多个 OpenCV 版本。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    charuco_corners = None
    charuco_ids = None
    marker_corners = None
    marker_ids = None

    try:
        charuco_detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
    except (AttributeError, cv2.error):
        try:
            params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            params = cv2.aruco.DetectorParameters()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        if marker_ids is not None and len(marker_ids) >= 4:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )

    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        return None, None, marker_corners, marker_ids

    return charuco_corners, charuco_ids, marker_corners, marker_ids


def generate_board(
    output_path="charuco_board.png",
    squares_x=DEFAULT_SQUARES_X,
    squares_y=DEFAULT_SQUARES_Y,
    square_length_mm=DEFAULT_SQUARE_LENGTH_MM,
    marker_length_mm=DEFAULT_MARKER_LENGTH_MM,
    dpi=300,
):
    """生成 ChArUco 标定板图像，适合 A4 纸打印。"""
    board, _ = _create_board(squares_x, squares_y, square_length_mm, marker_length_mm)

    mm_to_px = dpi / 25.4
    board_width_px = int(squares_x * square_length_mm * mm_to_px)
    board_height_px = int(squares_y * square_length_mm * mm_to_px)

    a4_width_px = int(210 * mm_to_px)
    a4_height_px = int(297 * mm_to_px)

    board_img = board.generateImage((board_width_px, board_height_px), marginSize=0)

    # 居中放置在 A4 画布上
    canvas = np.ones((a4_height_px, a4_width_px), dtype=np.uint8) * 255
    offset_x = (a4_width_px - board_width_px) // 2
    offset_y = (a4_height_px - board_height_px) // 2
    canvas[offset_y : offset_y + board_height_px, offset_x : offset_x + board_width_px] = board_img

    # 参数信息
    font_scale = 0.5 * dpi / 300
    thickness = max(1, dpi // 300)
    info = (
        f"ChArUco {squares_x}x{squares_y} | "
        f"Square={square_length_mm}mm Marker={marker_length_mm}mm | "
        f"Print at 100%, {dpi}DPI"
    )
    cv2.putText(
        canvas, info, (offset_x, a4_height_px - offset_y // 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, thickness,
    )

    # 10mm 刻度尺（验证打印尺寸）
    ruler_y = offset_y + board_height_px + int(15 * mm_to_px)
    ruler_start_x = offset_x
    segment_px = int(10 * mm_to_px)
    for i in range(5):
        x1 = ruler_start_x + i * segment_px
        x2 = x1 + segment_px
        color = 0 if i % 2 == 0 else 180
        cv2.rectangle(canvas, (x1, ruler_y), (x2, ruler_y + int(5 * mm_to_px)), color, -1)
        cv2.putText(
            canvas, f"{i * 10}mm", (x1, ruler_y + int(10 * mm_to_px)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35 * dpi / 300, 0, thickness,
        )

    from PIL import Image

    Image.fromarray(canvas).save(output_path, dpi=(dpi, dpi))
    print(f"标定板已保存至: {output_path}")
    print(f"参数: {squares_x}x{squares_y}, 方格={square_length_mm}mm, 标记={marker_length_mm}mm")
    print("请以 100% 比例打印（不要「缩放至适合页面」），用刻度尺验证尺寸。")


def calibrate_camera(
    camera_index,
    output_path,
    num_captures=15,
    squares_x=DEFAULT_SQUARES_X,
    squares_y=DEFAULT_SQUARES_Y,
    square_length_mm=DEFAULT_SQUARE_LENGTH_MM,
    marker_length_mm=DEFAULT_MARKER_LENGTH_MM,
    width=640,
    height=480,
):
    """交互式摄像头标定。同时尝试普通和鱼眼模型，自动选择更优的。"""
    board, dictionary = _create_board(squares_x, squares_y, square_length_mm, marker_length_mm)
    all_obj_pts_3d = _get_board_corners_3d(board)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = (actual_w, actual_h)

    captures_obj = []
    captures_img = []

    print(f"\n{'=' * 50}")
    print(f"摄像头标定 - Camera {camera_index} ({actual_w}x{actual_h})")
    print(f"{'=' * 50}")
    print(f"需要 {num_captures} 张标定图片。")
    print("操作:")
    print("  SPACE - 捕获当前帧")
    print("  C     - 用已有捕获进行标定（至少 6 张）")
    print("  Q     - 退出")
    print("提示: 在不同角度和位置展示标定板，覆盖画面各区域。\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        charuco_corners, charuco_ids, marker_corners, marker_ids = _detect_charuco(
            frame, board, dictionary
        )

        vis = frame.copy()
        if marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

        n = len(captures_obj)
        if charuco_corners is not None and len(charuco_corners) >= 6:
            status = f"[{n}/{num_captures}] {len(charuco_corners)} corners - SPACE to capture"
            color = (0, 255, 0)
        else:
            status = f"[{n}/{num_captures}] Move board into view"
            color = (0, 0, 255)

        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Camera Calibration", vis)

        key = cv2.waitKey(30) & 0xFF

        if key == ord(" ") and charuco_corners is not None and len(charuco_corners) >= 6:
            ids_flat = charuco_ids.flatten()
            obj_pts = all_obj_pts_3d[ids_flat].astype(np.float32)
            img_pts = charuco_corners.reshape(-1, 2).astype(np.float32)
            captures_obj.append(obj_pts)
            captures_img.append(img_pts)
            print(f"  Captured {len(captures_obj)}/{num_captures} ({len(charuco_corners)} corners)")

            if len(captures_obj) >= num_captures:
                print(f"\nReached {num_captures} captures. Calibrating...")
                break

        elif key == ord("c") and len(captures_obj) >= 6:
            print(f"\nCalibrating with {len(captures_obj)} captures...")
            break

        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("Calibration cancelled.")
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(captures_obj) < 6:
        print(f"Error: need at least 6 captures, got {len(captures_obj)}")
        return

    # === 普通模型 ===
    print("\n--- Normal lens model ---")
    obj_normal = [pts.reshape(-1, 1, 3) for pts in captures_obj]
    img_normal = [pts.reshape(-1, 1, 2) for pts in captures_img]

    try:
        ret_normal, K_normal, D_normal, _, _ = cv2.calibrateCamera(
            obj_normal, img_normal, image_size, None, None
        )
        print(f"  Reprojection error: {ret_normal:.4f} px")
    except cv2.error as e:
        print(f"  Failed: {e}")
        ret_normal = float("inf")
        K_normal, D_normal = None, None

    # === 鱼眼模型 ===
    print("--- Fisheye lens model ---")
    obj_fish = [pts.reshape(1, -1, 3).astype(np.float64) for pts in captures_obj]
    img_fish = [pts.reshape(1, -1, 2).astype(np.float64) for pts in captures_img]

    K_fish = np.zeros((3, 3), dtype=np.float64)
    D_fish = np.zeros((4, 1), dtype=np.float64)

    try:
        ret_fish, K_fish, D_fish, _, _ = cv2.fisheye.calibrate(
            obj_fish,
            img_fish,
            image_size,
            K_fish,
            D_fish,
            flags=(cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
        print(f"  Reprojection error: {ret_fish:.4f} px")
    except cv2.error as e:
        print(f"  Failed: {e}")
        ret_fish = float("inf")

    # === 选择更优模型 ===
    if ret_normal == float("inf") and ret_fish == float("inf"):
        print("\nBoth models failed! Check board parameters or add more captures.")
        return

    if ret_fish < ret_normal:
        model_type = "fisheye"
        K, D, rpe = K_fish, D_fish, ret_fish
        print(f"\n>>> Selected: fisheye (error {ret_fish:.4f} < normal {ret_normal:.4f})")
    else:
        model_type = "normal"
        K, D, rpe = K_normal, D_normal, ret_normal
        print(f"\n>>> Selected: normal (error {ret_normal:.4f} < fisheye {ret_fish:.4f})")

    # === 保存 ===
    calibration = {
        "model_type": model_type,
        "camera_matrix": K.tolist(),
        "dist_coeffs": D.flatten().tolist(),
        "image_size": list(image_size),
        "reprojection_error": float(rpe),
        "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "board_config": {
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_length_mm": square_length_mm,
            "marker_length_mm": marker_length_mm,
        },
        "comparison": {
            "normal_error": float(ret_normal) if ret_normal != float("inf") else None,
            "fisheye_error": float(ret_fish) if ret_fish != float("inf") else None,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCalibration saved to: {output_path}")
    print(f"Model: {model_type}, Error: {rpe:.4f} px, Size: {image_size}")


def load_calibration(calibration_file):
    """加载标定参数文件。返回包含 camera_matrix, dist_coeffs, model_type, image_size 的字典。"""
    with open(calibration_file) as f:
        calib = json.load(f)

    calib["camera_matrix"] = np.array(calib["camera_matrix"], dtype=np.float64)
    calib["dist_coeffs"] = np.array(calib["dist_coeffs"], dtype=np.float64)
    if calib["model_type"] == "fisheye":
        calib["dist_coeffs"] = calib["dist_coeffs"].reshape(4, 1)

    return calib


def compute_undistort_maps(calibration, image_size=None):
    """计算畸变矫正映射表。

    Args:
        calibration: load_calibration() 的返回值
        image_size: (width, height)，None 则使用标定时的尺寸

    Returns:
        (map1, map2) 用于 cv2.remap()
    """
    K = calibration["camera_matrix"].copy()
    D = calibration["dist_coeffs"].copy()
    model_type = calibration["model_type"]
    calib_size = tuple(calibration["image_size"])

    if image_size is None:
        image_size = calib_size

    # 分辨率不同时按比例缩放内参
    if image_size != calib_size:
        sx = image_size[0] / calib_size[0]
        sy = image_size[1] / calib_size[1]
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    if model_type == "fisheye":
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, image_size, np.eye(3), balance=0.0, new_size=image_size
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, image_size, cv2.CV_16SC2
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K, D, image_size, alpha=0, newImgSize=image_size
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, image_size, cv2.CV_16SC2
        )

    return map1, map2


def verify_calibration(calibration_file, camera_index, width=640, height=480):
    """实时预览畸变矫正效果。左侧原图，右侧矫正后。"""
    calib = load_calibration(calibration_file)
    image_size = (width, height)
    map1, map2 = compute_undistort_maps(calib, image_size)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"Error: cannot open camera {camera_index}")
        return

    print(f"\nVerification - Camera {camera_index}")
    print(f"Model: {calib['model_type']}, Error: {calib['reprojection_error']:.4f} px")
    print("Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        h, w = frame.shape[:2]
        display = np.hstack([frame, undistorted])
        cv2.putText(display, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display, "Undistorted", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Calibration Verification", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Camera distortion calibration tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate-board
    p_board = subparsers.add_parser("generate-board", help="Generate ChArUco board image")
    p_board.add_argument("--output", default="charuco_board.png", help="Output path")
    p_board.add_argument("--squares-x", type=int, default=DEFAULT_SQUARES_X)
    p_board.add_argument("--squares-y", type=int, default=DEFAULT_SQUARES_Y)
    p_board.add_argument("--square-length", type=float, default=DEFAULT_SQUARE_LENGTH_MM, help="mm")
    p_board.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH_MM, help="mm")
    p_board.add_argument("--dpi", type=int, default=300)

    # calibrate
    p_calib = subparsers.add_parser("calibrate", help="Interactive camera calibration")
    p_calib.add_argument("--camera-index", type=int, required=True)
    p_calib.add_argument("--output", required=True, help="Output calibration JSON path")
    p_calib.add_argument("--num-captures", type=int, default=15)
    p_calib.add_argument("--width", type=int, default=640)
    p_calib.add_argument("--height", type=int, default=480)
    p_calib.add_argument("--squares-x", type=int, default=DEFAULT_SQUARES_X)
    p_calib.add_argument("--squares-y", type=int, default=DEFAULT_SQUARES_Y)
    p_calib.add_argument("--square-length", type=float, default=DEFAULT_SQUARE_LENGTH_MM)
    p_calib.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH_MM)

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify calibration with live preview")
    p_verify.add_argument("--calibration", required=True, help="Calibration JSON path")
    p_verify.add_argument("--camera-index", type=int, required=True)
    p_verify.add_argument("--width", type=int, default=640)
    p_verify.add_argument("--height", type=int, default=480)

    args = parser.parse_args()

    if args.command == "generate-board":
        generate_board(
            output_path=args.output,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length_mm=args.square_length,
            marker_length_mm=args.marker_length,
            dpi=args.dpi,
        )
    elif args.command == "calibrate":
        calibrate_camera(
            camera_index=args.camera_index,
            output_path=args.output,
            num_captures=args.num_captures,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length_mm=args.square_length,
            marker_length_mm=args.marker_length,
            width=args.width,
            height=args.height,
        )
    elif args.command == "verify":
        verify_calibration(
            calibration_file=args.calibration,
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
