import socket
import struct
import cv2
import time

def start_server():
    # 1. Setup the network socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Listen on all network interfaces on port 8485
    server_socket.bind(('0.0.0.0', 8485))
    server_socket.listen(1)
    print("[*] Stereo Server is listening on port 8485...")
    
    conn, addr = server_socket.accept()
    print(f"[*] Connected to laptop at: {addr}")

    # 2. Initialize both cameras
    # (Assuming standard USB webcams or standard V4L2 indices)
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    # Keep resolution low (320x240 each) so network streaming stays real-time
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    try:
        while True:
            ret_l, frame_l = cap_left.read()
            ret_r, frame_r = cap_right.read()
            
            # If a camera drops a frame, skip to the next loop
            if not ret_l or not ret_r:
                continue
            
            # 3. Stitch them side-by-side into a single 640x240 image
            stereo_frame = cv2.hconcat([frame_l, frame_r])
            
            # 4. Compress to JPEG to save network bandwidth (Quality: 80)
            result, encoded_frame = cv2.imencode('.jpg', stereo_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            data = encoded_frame.tobytes()
            
            # 5. Send frame size header, then the actual frame data
            size = len(data)
            conn.sendall(struct.pack(">L", size) + data)
            
    except Exception as e:
        print(f"[*] Streaming stopped: {e}")
    finally:
        # Clean up camera resources and network ports
        cap_left.release()
        cap_right.release()
        conn.close()
        server_socket.close()

if __name__ == "__main__":
    start_server()