# Object Detection using YOLOv8 + Raspberry Pi 🚁📷

A real-time object detection system using **YOLOv8**, where a **Raspberry Pi streams video frames** to a laptop/server for fast AI inference. Designed for robotics, drones, and edge-AI applications.

## Features

- Real-time object detection with YOLOv8
- Raspberry Pi camera streaming over sockets
- Laptop/server-side inference for higher FPS
- Lightweight and optimized architecture
- Easy to integrate with drones/robots
- OpenCV-based frame handling

---

## Project Structure

```bash
Object_Detection/
│── server.py        # Runs on laptop/PC (YOLO inference)
│── client.py        # Runs on Raspberry Pi (camera stream)
│── requirements.txt
│── README.md
```

---

## How It Works

1. Raspberry Pi captures frames using the camera.
2. Frames are serialized and sent over WiFi using sockets.
3. Laptop receives frames.
4. YOLOv8 performs object detection.
5. Detected objects are displayed in real time.

---

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- Socket Programming
- Raspberry Pi
- NumPy

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YasasviAvvaru/Object_Detection.git
cd Object_Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install ultralytics opencv-python numpy
```

---

## Running the Project

### Step 1: Run the server on Laptop/PC

```bash
python server.py
```

This starts the YOLO inference server.

---

### Step 2: Run the client on Raspberry Pi

Update the laptop IP address inside `client.py`:

```python
HOST = "YOUR_LAPTOP_IP"
```

Then run:

```bash
python client.py
```

---

## Example Workflow

```text
Raspberry Pi Camera
        ↓
 Frame Streaming via WiFi
        ↓
Laptop/PC Server
        ↓
 YOLOv8 Inference
        ↓
 Real-time Detection Output
```

---

## Applications

- Autonomous drones
- Surveillance systems
- Robotics
- Smart navigation
- Real-time AI vision systems

---

## Future Improvements

- Multi-threaded streaming
- FPS optimization
- TensorRT acceleration
- Tracking integration (DeepSORT/ByteTrack)
- ROS integration
- On-device inference on Raspberry Pi

---

## Sample Detection Classes

YOLOv8 can detect:
- Person
- Car
- Bicycle
- Dog
- Bottle
- Chair
- and many more...

---

## Requirements

- Raspberry Pi with camera module
- Laptop/PC with GPU (recommended)
- Stable WiFi connection
- Python 3.9+

---

## Author

Created by Yasasvi Avvaru

GitHub: https://github.com/YasasviAvvaru
