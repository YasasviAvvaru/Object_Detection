import argparse
import time

import cv2
from ultralytics import YOLO

from vision_utils import (
    FpsMeter,
    add_common_yolo_args,
    detections_from_result,
    draw_detections,
    focal_from_fov,
    monocular_distance_m,
    open_camera,
    parse_classes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast single-camera object detection and distance estimation.")
    add_common_yolo_args(parser)
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--focal-px", type=float, default=None, help="Calibrated focal length in pixels.")
    parser.add_argument(
        "--hfov",
        type=float,
        default=66.0,
        help="Horizontal field of view in degrees. Pi Camera Module 3 wide is about 102; standard is about 66.",
    )
    parser.add_argument("--save-csv", default=None, help="Optional CSV log with timestamp,label,confidence,distance_m,bbox.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    labels = parse_classes(args.classes)
    focal_px = args.focal_px or focal_from_fov(args.width, args.hfov)

    model = YOLO(args.model)
    cap = open_camera(args.camera, args.width, args.height, args.backend)
    fps_meter = FpsMeter()

    csv_file = None
    if args.save_csv:
        csv_file = open(args.save_csv, "w", encoding="utf-8", newline="")
        csv_file.write("timestamp,people_count,label,confidence,distance_m,x1,y1,x2,y2\n")

    print("Press q or Esc to quit.")
    print(f"Detecting: {', '.join(labels)} | focal_px={focal_px:.1f}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]

            detections = detections_from_result(result, labels, args.conf)
            people_count = sum(1 for det in detections if det.label == "person")
            for det in detections:
                det.distance_m = monocular_distance_m(det, focal_px)
                det.source = "single"
                if csv_file and det.distance_m is not None:
                    x1, y1, x2, y2 = det.xyxy
                    csv_file.write(
                        f"{time.time():.3f},{people_count},{det.label},{det.conf:.4f},{det.distance_m:.4f},{x1},{y1},{x2},{y2}\n"
                    )

            fps = fps_meter.tick()
            draw_detections(frame, detections, fps=fps, people_count=people_count)
            cv2.imshow("Single Camera Detection + Distance", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
