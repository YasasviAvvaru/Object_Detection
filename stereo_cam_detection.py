import argparse
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from vision_utils import (
    Detection,
    FpsMeter,
    add_common_yolo_args,
    detections_from_result,
    draw_detections,
    focal_from_fov,
    open_camera,
    parse_classes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-camera stereo object detection and distance estimation.")
    add_common_yolo_args(parser)
    parser.add_argument("--left-camera", type=int, default=0, help="Left camera index.")
    parser.add_argument("--right-camera", type=int, default=1, help="Right camera index.")
    parser.add_argument("--baseline-m", type=float, default=0.10, help="Distance between camera centers in meters.")
    parser.add_argument("--focal-px", type=float, default=None, help="Calibrated focal length in pixels.")
    parser.add_argument(
        "--hfov",
        type=float,
        default=66.0,
        help="Horizontal field of view in degrees. Pi Camera Module 3 standard is about 66.",
    )
    parser.add_argument("--save-csv", default=None, help="Optional CSV log with stereo distances.")
    return parser


def y_overlap(a: Detection, b: Detection) -> float:
    ay1, ay2 = a.xyxy[1], a.xyxy[3]
    by1, by2 = b.xyxy[1], b.xyxy[3]
    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    denom = max(1, min(a.height_px, b.height_px))
    return overlap / denom


def match_stereo(left: List[Detection], right: List[Detection]) -> List[Tuple[Detection, Detection]]:
    matches: List[Tuple[Detection, Detection]] = []
    used_right = set()

    for ldet in sorted(left, key=lambda d: d.conf, reverse=True):
        lx, ly = ldet.center
        best_index: Optional[int] = None
        best_score = float("inf")

        for idx, rdet in enumerate(right):
            if idx in used_right or rdet.label != ldet.label:
                continue
            rx, ry = rdet.center
            disparity = lx - rx
            if disparity <= 1:
                continue
            overlap = y_overlap(ldet, rdet)
            if overlap < 0.35:
                continue
            score = abs(ly - ry) + abs(ldet.width_px - rdet.width_px) * 0.25 - overlap * 20
            if score < best_score:
                best_score = score
                best_index = idx

        if best_index is not None:
            used_right.add(best_index)
            matches.append((ldet, right[best_index]))

    return matches


def stereo_distance_m(left: Detection, right: Detection, focal_px: float, baseline_m: float) -> Optional[float]:
    lx, _ = left.center
    rx, _ = right.center
    disparity = lx - rx
    if disparity <= 1:
        return None
    return (focal_px * baseline_m) / disparity


def main() -> None:
    args = build_parser().parse_args()
    labels = parse_classes(args.classes)
    focal_px = args.focal_px or focal_from_fov(args.width, args.hfov)

    model = YOLO(args.model)
    left_cap = open_camera(args.left_camera, args.width, args.height, args.backend)
    right_cap = open_camera(args.right_camera, args.width, args.height, args.backend)
    fps_meter = FpsMeter()

    csv_file = None
    if args.save_csv:
        csv_file = open(args.save_csv, "w", encoding="utf-8", newline="")
        csv_file.write("timestamp,people_count,label,confidence,distance_m,left_x1,left_y1,left_x2,left_y2,right_x1,right_y1,right_x2,right_y2\n")

    print("Press q or Esc to quit.")
    print(f"Detecting: {', '.join(labels)} | focal_px={focal_px:.1f} | baseline_m={args.baseline_m:.3f}")

    try:
        while True:
            ok_left, left_frame = left_cap.read()
            ok_right, right_frame = right_cap.read()
            if not ok_left or not ok_right:
                break

            frames = [left_frame, right_frame]
            results = model.predict(
                frames,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )

            left_dets = detections_from_result(results[0], labels, args.conf)
            right_dets = detections_from_result(results[1], labels, args.conf)
            matches = match_stereo(left_dets, right_dets)
            people_count = sum(1 for ldet, _ in matches if ldet.label == "person")

            for ldet, rdet in matches:
                ldet.distance_m = stereo_distance_m(ldet, rdet, focal_px, args.baseline_m)
                ldet.source = "stereo"
                if csv_file and ldet.distance_m is not None:
                    lx1, ly1, lx2, ly2 = ldet.xyxy
                    rx1, ry1, rx2, ry2 = rdet.xyxy
                    csv_file.write(
                        f"{time.time():.3f},{people_count},{ldet.label},{ldet.conf:.4f},{ldet.distance_m:.4f},"
                        f"{lx1},{ly1},{lx2},{ly2},{rx1},{ry1},{rx2},{ry2}\n"
                    )

            fps = fps_meter.tick()
            draw_detections(left_frame, left_dets, fps=fps, people_count=people_count)
            draw_detections(right_frame, right_dets, fps=None, people_count=people_count)

            combined = np.hstack((left_frame, right_frame))
            cv2.imshow("Stereo Detection + Distance", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        left_cap.release()
        right_cap.release()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
