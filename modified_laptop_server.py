import socket
import cv2
import pickle
import struct
import threading
import torch
import time
from ultralytics import YOLO

# =========================
# LOAD YOLO
# =========================
model = YOLO("yolov8n.pt")

device = 0 if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# =========================
# SERVER SETTINGS
# =========================
HOST = "0.0.0.0"
PORT = 9999

# =========================
# SOCKET SETUP
# =========================
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_socket.bind((HOST, PORT))

server_socket.listen(1)

print("Waiting for Raspberry Pi connection...")

conn, addr = server_socket.accept()

print(f"Connected by: {addr}")

# =========================
# SHARED FRAME
# =========================
latest_frame = None

# =========================
# RECEIVE THREAD
# =========================
def receive_frames():

    global latest_frame

    data = b""

    payload_size = struct.calcsize("Q")

    while True:

        try:

            while len(data) < payload_size:

                packet = conn.recv(4096)

                if not packet:
                    return

                data += packet

            packed_msg_size = data[:payload_size]

            data = data[payload_size:]

            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:

                packet = conn.recv(4096)

                if not packet:
                    return

                data += packet

            frame_data = data[:msg_size]

            data = data[msg_size:]

            encoded = pickle.loads(frame_data)

            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # OVERWRITE OLD FRAME
            latest_frame = frame

        except:
            return

# START THREAD
threading.Thread(target=receive_frames, daemon=True).start()

# =========================
# INFERENCE LOOP
# =========================
prev_time = time.time()

while True:

    if latest_frame is None:
        continue

    frame = latest_frame.copy()

    # SMALLER FRAME
    frame = cv2.resize(frame, (320, 240))

    # YOLO INFERENCE
    results = model(
        frame,
        imgsz=320,
        device=device,
        half=True,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # FPS
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

conn.close()

server_socket.close()

cv2.destroyAllWindows()
