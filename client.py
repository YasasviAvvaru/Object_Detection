from picamera2 import Picamera2
import socket
import cv2
import pickle
import struct

# =========================
# LAPTOP IP
# =========================
HOST = "laptops IP"
PORT = 9999

# =========================
# SOCKET
# =========================
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print("Connecting to laptop...")

client_socket.connect((HOST, PORT))

print("Connected!")

# =========================
# CAMERA
# =========================
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (320, 240)}
)

picam2.configure(config)

picam2.start()

print("Camera started!")

# =========================
# SEND LOOP
# =========================
while True:

    frame = picam2.capture_array()

    # JPEG COMPRESS
    result, encoded = cv2.imencode(
        '.jpg',
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 35]
    )

    data = pickle.dumps(encoded)

    message = struct.pack("Q", len(data)) + data

    try:

        client_socket.sendall(message)

    except:
        break

client_socket.close()
