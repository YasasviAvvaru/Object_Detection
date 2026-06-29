import socket
import cv2
import pickle
import struct
# Import your custom modules from the repo
from vision_utils import detect_objects, estimate_distance 

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Replace with your Raspberry Pi's actual local IP address
client_socket.connect(('192.168.X.X', 8485)) 

data = b""
payload_size = struct.calcsize(">L")

while True:
    # Retrieve message size header
    while len(data) < payload_size:
        packet = client_socket.recv(4096)
        if not packet: break
        data += packet
    
    if not data:
        break
        
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    
    # Retrieve all packet data based on the message size
    while len(data) < msg_size:
        data += client_socket.recv(4096)
        
    frame_data = data[:msg_size]
    data = data[msg_size:]
    
    # Decode JPEG back to an OpenCV frame
    encoded_frame = pickle.loads(frame_data)
    frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
    
    # --- PROJECTION & CALCULATIONS (LAPTOP SIDE) ---
    # 1. Run your YOLO detection pipeline
    # detections = detect_objects(frame, model_size=320)
    
    # 2. Run your depth calculations (Single or Stereo matching)
    # processed_frame = estimate_distance(frame, detections)
    
    # Display the final output with calculated distances overlaid
    cv2.imshow('Laptop Processing Side', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
client_socket.close()



# from picamera2 import Picamera2
# import socket
# import cv2
# import pickle
# import struct

# # =========================
# # LAPTOP IP
# # =========================
# HOST = "laptops IP"
# PORT = 9999

# # =========================
# # SOCKET
# # =========================
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# print("Connecting to laptop...")

# client_socket.connect((HOST, PORT))

# print("Connected!")

# # =========================
# # CAMERA
# # =========================
# picam2 = Picamera2()

# config = picam2.create_preview_configuration(
#     main={"size": (320, 240)}
# )

# picam2.configure(config)

# picam2.start()

# print("Camera started!")

# # =========================
# # SEND LOOP
# # =========================
# while True:

#     frame = picam2.capture_array()

#     # JPEG COMPRESS
#     result, encoded = cv2.imencode(
#         '.jpg',
#         frame,
#         [int(cv2.IMWRITE_JPEG_QUALITY), 35]
#     )

#     data = pickle.dumps(encoded)

#     message = struct.pack("Q", len(data)) + data

#     try:

#         client_socket.sendall(message)

#     except:
#         break

# client_socket.close()
