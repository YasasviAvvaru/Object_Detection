import cv2
from ultralytics import YOLO

# Load YOLO
model = YOLO("yolov8n.pt")

# Camera parameters
PERSON_HEIGHT_CM = 170
FOCAL_LENGTH = 1700   # Obtain through calibration

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])

            # COCO class 0 = person
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            bbox_height = y2 - y1

            if bbox_height > 0:
                distance_cm = (PERSON_HEIGHT_CM * FOCAL_LENGTH) / bbox_height

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                cv2.putText(
                    frame,
                    f"{distance_cm/100:.2f} m",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

    cv2.imshow("Distance Estimation", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()