How to Run
Connect iVCam or prepare a recorded video file.

Set IVCAM_INDEX to the correct index (e.g., 0, 1, 2). Or set SOURCE = "recorded_video.mp4".

Run the script:

python wheel_profiler_realtime.py

At startup:

The script shows the first frame. Click two points on a ruler placed next to the wheel. Press ENTER and type the real distance (mm).

If no ruler is available, press ESC to use a dummy scale (0.1 mm/px).

During live video:

Point the camera at the wheel profile.

Press b to register the baseline.

Rotate the wheel or move the camera. The script will continuously show ΔP, ΔB, ΔH in real-time.

🔧 Performance Optimization
FRAME_SKIP = 2: Processes every 2nd frame to keep FPS high. YOLO is computationally heavy. If you have a powerful GPU, set it to 1. If it lags, set it to 3 or 4.

Fallback Canny: If YOLO misses a detection (e.g., unusual lighting), the script automatically falls back to Canny edge detection, ensuring the pipeline never breaks.
