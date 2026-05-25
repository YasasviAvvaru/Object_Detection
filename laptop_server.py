import socket
import cv2
import pickle
import struct
from ultralytics import YOLO

# LOAD YOLO MODEL
model = YOLO("yolov8n.pt")

# SERVER SETTINGS
HOST = "0.0.0.0"
PORT = 9999

# CREATE SOCKET
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# ALLOW REUSE
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Waiting for Raspberry Pi connection...")

conn, addr = server_socket.accept()

print(f"Connected by: {addr}")

data = b""

payload_size = struct.calcsize("Q")

while True:

    try:

        # RECEIVE SIZE OF FRAME
        while len(data) < payload_size:

            packet = conn.recv(4 * 1024)

            if not packet:
                print("Connection closed")
                break

            data += packet

        if len(data) < payload_size:
            break

        packed_msg_size = data[:payload_size]

        data = data[payload_size:]

        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # RECEIVE ACTUAL FRAME DATA
        while len(data) < msg_size:

            packet = conn.recv(4 * 1024)

            if not packet:
                break

            data += packet

        frame_data = data[:msg_size]

        data = data[msg_size:]

        # LOAD PICKLED JPEG BUFFER
        encoded = pickle.loads(frame_data)

        # DECODE JPEG -> IMAGE
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if frame is None:
            print("Frame decode failed")
            continue

        # RUN YOLO
        results = model(frame)

        # DRAW DETECTIONS
        annotated_frame = results[0].plot()

        # DISPLAY
        cv2.imshow("YOLO Detection", annotated_frame)

        # PRESS q TO EXIT
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:

        print("ERROR:", e)

        break

conn.close()

server_socket.close()

cv2.destroyAllWindows()