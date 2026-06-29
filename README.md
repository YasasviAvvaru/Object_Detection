# Object Detection + Distance Estimation

Real-time YOLO object detection for Raspberry Pi / laptop workflows using OpenCV, Ultralytics YOLO, and optional stereo distance estimation.

The project supports:

- Person detection and people counting (`person`, `ct`, or `count`)
- Vehicle detection for supported COCO classes (`car`, `truck`, `bus`, `motorcycle`)
- Single-camera distance estimation from object size
- Two-camera stereo distance estimation from disparity
- Pi-to-laptop streaming for running heavier detection on the laptop

## Files

```text
single_cam_detection.py        # local one-camera YOLO detection + distance
stereo_cam_detection.py        # local two-camera YOLO detection + stereo distance
vision_utils.py                # shared camera, detection, drawing, and distance helpers
evaluate_distance_accuracy.py  # computes MAE/RMSE/MAPE/accuracy from measured CSV data

server.py                      # one-camera Raspberry Pi streaming server
client.py                      # one-camera laptop receiver/viewer
Two_camera_server.py           # two-camera Raspberry Pi streaming server
Two_camera_client.py           # laptop stereo receiver + YOLO + distance display
```

## Install

Install Python packages:

```bash
pip install -r requirements.txt
```

For Raspberry Pi Camera Module support, install Picamera2 through Raspberry Pi OS packages:

```bash
sudo apt install python3-picamera2
```

If `yolov8n.pt` is not already present, Ultralytics will download it the first time you run a YOLO script.

## Network Setup

Use this setup when the Raspberry Pi captures frames and the laptop receives or processes them.

1. Connect the Raspberry Pi and laptop to the same Wi-Fi network. Avoid networks that block device-to-device traffic.
2. On the Raspberry Pi, find its IP address:

```bash
hostname -I
```

3. Put that IP address into the laptop client script:

```python
PI_IP = "192.168.x.x"
```

For `client.py`, update:

```python
client_socket.connect(("192.168.x.x", 8485))
```

Port `8485` is used by both streaming examples.

## One-Camera Streaming

Run this when one camera is connected to the Raspberry Pi and frames are viewed or processed on the laptop.

On the Raspberry Pi:

```bash
python server.py
```

On the laptop:

```bash
python client.py
```

Press `q` in the display window to quit.

## Two-Camera Streaming

Run this when two cameras are connected to the Raspberry Pi and stereo distance is computed on the laptop.

On the Raspberry Pi:

```bash
python Two_camera_server.py
```

On the laptop:

```bash
python Two_camera_client.py
```

Before running, edit `Two_camera_client.py`:

```python
PI_IP = "192.168.x.x"
baseline_m = 0.10
focal_px = 492.0
```

`baseline_m` is the distance between the two camera lens centers in meters. `focal_px` should be calibrated for the image width used by the server. The current two-camera server captures each camera at `320x240`, stitches them into one `640x240` frame, and sends JPEG frames at quality `80`.

## Local Single-Camera Detection

Run detection directly on the machine with the camera:

```bash
python single_cam_detection.py --backend auto --camera 0 --classes person --imgsz 320 --conf 0.35
```

Useful options:

```bash
python single_cam_detection.py --classes person,car,bus,motorcycle
python single_cam_detection.py --hfov 66
python single_cam_detection.py --focal-px 560
python single_cam_detection.py --save-csv single_log.csv
```

Pi Camera Module 3 approximate horizontal FOV:

- Standard lens: about `66` degrees
- Wide lens: about `102` degrees

Use `--focal-px` after calibration for better distance accuracy. `--hfov` is only an approximation.

## Local Two-Camera Detection

Run stereo detection directly on the machine with both cameras:

```bash
python stereo_cam_detection.py --backend auto --left-camera 0 --right-camera 1 --baseline-m 0.10 --classes person --imgsz 320
```

For good stereo accuracy:

- Mount both cameras horizontally at the same height.
- Keep both cameras pointing straight ahead.
- Measure the distance between lens centers and pass it as `--baseline-m`.
- Calibrate `--focal-px` if possible.
- Keep the two images aligned and rectified. Vertical shift or angled cameras will make distance noisy.

## Speed Tips

- Use `yolov8n.pt` for the fastest YOLO model.
- Use `--imgsz 320` for speed; try `416` or `640` for more accuracy.
- Use `--width 640 --height 480` for local camera scripts.
- Lower JPEG quality in the streaming server if the network is slow.
- Keep `--conf` around `0.35` to reduce false positives.
- On a CUDA laptop/server, pass `--device 0` to local YOLO scripts.

## Accuracy Measurement

Real distance accuracy needs ground-truth measurements. To compare one-camera and two-camera distance:

1. Put people or objects at known distances, for example `1m`, `2m`, `3m`, and `5m`.
2. Record the predicted distance from the scripts.
3. Create a CSV like this:

```csv
method,label,predicted_m,actual_m
single,person,2.34,2.00
stereo,person,2.08,2.00
```

4. Run:

```bash
python evaluate_distance_accuracy.py accuracy_samples.csv
```

Output includes:

- MAE in meters
- RMSE in meters
- MAPE percentage
- Distance accuracy percentage, computed as `100 - MAPE`

Expected behavior after calibration:

- Single camera is usually less accurate because it assumes average real-world object size.
- Two cameras are usually more accurate at close and medium range if baseline, focal length, and alignment are correct.

## Calibration Notes

For single-camera distance:

```text
focal_px = (known_distance_m * object_pixel_size) / real_object_size_m
```

For stereo distance:

```text
distance_m = (focal_px * baseline_m) / disparity_px
```

For better single-camera distance, adjust the real object dimensions in `vision_utils.py`. The default person height is `1.70 m`.
