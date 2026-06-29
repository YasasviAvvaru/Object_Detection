import socket
import struct
import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
PI_IP = '192.168.1.100'  # <--- CHANGE THIS TO YOUR PI'S IP ADDRESS
PORT = 8485

# Stereo Math Configuration
# Baseline is in METERS (e.g., 10 cm = 0.10)
baseline_m = 0.10  
# Calibrated focal length in pixels for a 320-width image
focal_px = 492.0   

def start_client():
    # 1. Connect to the Raspberry Pi
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((PI_IP, PORT))
        print(f"[*] Connected to Pi stream at {PI_IP}!")
    except Exception as e:
        print(f"[!] Could not connect to Pi. Is the server running? Error: {e}")
        return

    # 2. Load YOLO Model
    print("[*] Loading YOLO model...")
    model = YOLO('yolov8n.pt') 

    data = b""
    payload_size = struct.calcsize(">L")

    while True:
        # 3. Network Receive Loop (Handles TCP packet chunking)
        # Fetch the 4-byte size header
        while len(data) < payload_size:
            packet = client_socket.recv(4096)
            if not packet: break
            data += packet
            
        if not data:
            print("[*] Connection closed by server.")
            break
            
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        
        # Fetch the actual frame data based on the size header
        while len(data) < msg_size:
            data += client_socket.recv(4096)
            
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        # 4. Decode JPEG back to OpenCV Matrix
        np_arr = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # 5. Split the wide frame back into Left and Right
        height, width, _ = frame.shape
        half_width = width // 2
        frame_left = frame[:, :half_width]
        frame_right = frame[:, half_width:]
        
        # 6. Run YOLO on both halves (classes=[0] means 'person' only)
        res_left = model(frame_left, verbose=False, classes=[0])
        res_right = model(frame_right, verbose=False, classes=[0])
        
        # 7. Stereo Matching and Distance Math
        # Check if at least one person is detected in BOTH cameras
        if len(res_left[0].boxes) > 0 and len(res_right[0].boxes) > 0:
            
            # Get the first detected person's bounding box coordinates
            box_l = res_left[0].boxes[0].xyxy[0].cpu().numpy()
            box_r = res_right[0].boxes[0].xyxy[0].cpu().numpy()
            
            # Find the center X coordinate of the person in both frames
            center_x_l = (box_l[0] + box_l[2]) / 2
            center_x_r = (box_r[0] + box_r[2]) / 2
            
            # Calculate pixel disparity (difference in X positions)
            disparity_px = abs(center_x_l - center_x_r)
            
            # Prevent division by zero
            if disparity_px > 0:
                # The Golden Stereo Formula
                distance = (focal_px * baseline_m) / disparity_px
                
                # Draw the bounding box on the LEFT frame
                x1, y1, x2, y2 = int(box_l[0]), int(box_l[1]), int(box_l[2]), int(box_l[3])
                cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Overlay the distance text above the bounding box
                cv2.putText(frame_left, f"Dist: {distance:.2f}m", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 8. Display Output on Laptop Screen
        cv2.imshow("Laptop-Side Stereo Output", frame_left)
        
        # Press 'q' to quit safely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up on exit
    cv2.destroyAllWindows()
    client_socket.close()

if __name__ == "__main__":
    start_client()