import argparse
import csv
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


COCO_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Average object dimensions used for monocular distance. Measure your own target
# objects and override these values when you need better distance accuracy.
OBJECT_DIMS_M = {
    "person": {"height": 1.70, "width": 0.45},
    "car": {"height": 1.50, "width": 1.80},
    "truck": {"height": 3.00, "width": 2.50},
    "bus": {"height": 3.20, "width": 2.55},
    "motorcycle": {"height": 1.20, "width": 0.80},
}


@dataclass
class Detection:
    label: str
    conf: float
    xyxy: Tuple[int, int, int, int]
    distance_m: Optional[float] = None
    source: str = ""

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def height_px(self) -> int:
        return max(0, self.xyxy[3] - self.xyxy[1])

    @property
    def width_px(self) -> int:
        return max(0, self.xyxy[2] - self.xyxy[0])


class FpsMeter:
    def __init__(self, smooth: float = 0.9) -> None:
        self.smooth = smooth
        self.last = time.perf_counter()
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        dt = max(now - self.last, 1e-6)
        instant = 1.0 / dt
        self.fps = instant if self.fps == 0.0 else self.fps * self.smooth + instant * (1.0 - self.smooth)
        self.last = now
        return self.fps


def parse_classes(raw: str) -> List[str]:
    aliases = {
        "people": "person",
        "human": "person",
        "humans": "person",
        "ct": "person",
        "count": "person",
        "vehicle": "car,truck,bus,motorcycle",
        "vehicles": "car,truck,bus,motorcycle",
    }
    labels: List[str] = []
    for part in raw.split(","):
        token = part.strip().lower()
        token = aliases.get(token, token)
        for label in token.split(","):
            label = label.strip()
            if label and label not in labels:
                labels.append(label)
    return labels


def class_ids_for(labels: Iterable[str]) -> List[int]:
    wanted = set(labels)
    return [class_id for class_id, name in COCO_CLASSES.items() if name in wanted]


def focal_from_fov(width_px: int, horizontal_fov_deg: float) -> float:
    return width_px / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))


def monocular_distance_m(det: Detection, focal_px: float) -> Optional[float]:
    dims = OBJECT_DIMS_M.get(det.label)
    if not dims:
        return None

    # Height is usually more stable for people. Width is often better for vehicles
    # because their visible height changes a lot with viewpoint.
    dimension_key = "height" if det.label == "person" else "width"
    real_size_m = dims[dimension_key]
    pixel_size = det.height_px if dimension_key == "height" else det.width_px
    if pixel_size <= 1:
        return None
    return (real_size_m * focal_px) / pixel_size


def detections_from_result(result, wanted_labels: Sequence[str], conf: float) -> List[Detection]:
    wanted = set(wanted_labels)
    detections: List[Detection] = []
    names: Dict[int, str] = result.names

    for box in result.boxes:
        class_id = int(box.cls[0])
        label = names.get(class_id, str(class_id))
        score = float(box.conf[0])
        if label not in wanted or score < conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append(Detection(label=label, conf=score, xyxy=(x1, y1, x2, y2)))
    return detections


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    fps: Optional[float] = None,
    people_count: Optional[int] = None,
) -> np.ndarray:
    colors = {
        "person": (0, 220, 0),
        "car": (255, 180, 0),
        "truck": (0, 180, 255),
        "bus": (255, 80, 80),
        "motorcycle": (180, 80, 255),
    }
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        color = colors.get(det.label, (255, 255, 255))
        distance = "?"
        if det.distance_m is not None and math.isfinite(det.distance_m):
            distance = f"{det.distance_m:.2f}m"
        caption = f"{det.label} {det.conf:.2f} {distance}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, caption, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if fps is not None:
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 255, 40), 2)
    if people_count is not None:
        cv2.putText(frame, f"People count: {people_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 255, 40), 2)
    return frame


class Picamera2Capture:
    def __init__(self, index: int, width: int, height: int) -> None:
        from picamera2 import Picamera2

        self.picam = Picamera2(camera_num=index)
        config = self.picam.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
        self.picam.configure(config)
        self.picam.start()

    def read(self):
        frame = self.picam.capture_array()
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self) -> None:
        self.picam.stop()


def open_camera(index: int, width: int, height: int, backend: str = "auto"):
    if backend in ("auto", "picamera2"):
        try:
            return Picamera2Capture(index, width, height)
        except Exception:
            if backend == "picamera2":
                raise

    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    return cap


def add_common_yolo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path. Use yolov8n.pt for speed.")
    parser.add_argument("--classes", default="person", help="Comma list. 'ct' and 'count' mean person count.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size. 320 is fast on Pi 5.")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height.")
    parser.add_argument("--device", default=None, help="YOLO device, for example cpu, mps, 0. Default lets Ultralytics choose.")
    parser.add_argument("--backend", choices=["auto", "opencv", "picamera2"], default="auto", help="Camera backend.")


def write_accuracy_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
