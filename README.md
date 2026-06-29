# Object Detection + Distance Estimation

Real-time YOLO object detection for Raspberry Pi 5 / laptop workflows. The updated scripts detect:

- `person`
- `ct`, meaning count of people
- optional COCO vehicle classes such as `bus` and `motorcycle`

Distance is supported in two modes:

- Single camera: estimates distance from detected object size and focal length.
- Two cameras: estimates distance from stereo disparity using baseline and focal length.

## setting up connection
1) Make sure laptop and raspi are connected to same wifi network(not IITD WIFI).
2) in raspi -> hostname -I (find ip and put it in laptops client.py) 
3) in raspi terminal -> python server.py
4) in laptop terminal -> python client.py
5) for faster running 
    a)decrease imgsz to 320 in client.py
    b)decrease jpeg quality to 75 in server.py
## Files

```text
single_cam_detection.py        # one camera detection + distance
stereo_cam_detection.py        # two camera detection + stereo distance
vision_utils.py                # shared detection, drawing, distance helpers
evaluate_distance_accuracy.py  # computes MAE/RMSE/MAPE/accuracy from measured data
server.py / client.py          # older Pi-to-laptop streaming demo
```

## Install

```bash
pip install -r requirements.txt
```

For Raspberry Pi camera support, install Picamera2 through Raspberry Pi OS packages:

```bash
sudo apt install python3-picamera2
```

## Single Camera

```bash
python single_cam_detection.py --backend picamera2 --camera 0 --classes person --imgsz 320 --conf 0.35
```

Useful options:

```bash
python single_cam_detection.py --hfov 66
python single_cam_detection.py --focal-px 560
python single_cam_detection.py --save-csv single_log.csv
```

Pi Camera Module 3 approximate horizontal FOV:

- Standard lens: about `66` degrees
- Wide lens: about `102` degrees

Use `--focal-px` after calibration for better distance accuracy. `--hfov` is only an approximation.

## Two Cameras / Stereo

Mount both cameras horizontally, same height, looking straight ahead. Measure the distance between lens centers and pass it as `--baseline-m`.

```bash
python stereo_cam_detection.py --backend picamera2 --left-camera 0 --right-camera 1 --baseline-m 0.10 --classes person --imgsz 320
```

For good stereo accuracy, the two camera images should be aligned and calibrated. The current script assumes the cameras are roughly parallel and rectified. If the two images are vertically shifted or angled, stereo distance will be noisy.

## Speed Tips for Raspberry Pi 5

- Start with `yolov8n.pt`.
- Use `--imgsz 320` for best speed, `416` or `640` for more accuracy.
- Use `--width 640 --height 480`.
- Keep `--conf` around `0.35` to reduce false positives.
- If running on a laptop/server, use `--device 0` for CUDA GPU.

## Accuracy Measurement

You cannot know real accuracy without ground-truth measured distances. To compare one-camera and two-camera distance:

1. Put people at known distances, for example `1m, 2m, 3m, 5m`.
2. Record predicted distance from the scripts.
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
- distance accuracy percentage, computed as `100 - MAPE`

Expected behavior after calibration:

- Single camera: usually less accurate because it assumes average person height.
- Two cameras: usually more accurate at close/medium range if baseline, focal length, and alignment are correct.

## Notes

For best single-camera distance, calibrate focal length:

```text
focal_px = (known_distance_m * object_pixel_size) / real_object_size_m
```

For stereo:

```text
distance_m = (focal_px * baseline_m) / disparity_px
```

Measure your own average person height in `vision_utils.py` if your target users differ from the default `1.70 m`.
