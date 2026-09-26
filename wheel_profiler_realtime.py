# wheel_profiler_realtime.py
# ====================================================================
#  REALTIME WHEEL WEAR MEASUREMENT SYSTEM
#  Based on "轨交明眸" (Rail Transit Smart Eye) - Video Stream Mode
#  Features: Baseline Registration, Real-time P/B/H, Wear Delta Tracking
# ====================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from ultralytics import YOLO
import sys
import time

# ============================================================
#  CONFIGURATION
# ============================================================
CALIB_FILE = "calibration.npz"
YOLO_MODEL_PATH = "yolov8n-seg.pt"  # Or your custom "best.pt"
IVCAM_INDEX = 1  # Change to your iVCam index
FRAME_SKIP = 2   # Process every 2nd frame to maintain real-time FPS

# Measurement standards
FLANGE_MEASURE_HEIGHT = 12.0  # mm (Height for flange thickness B)

class RealtimeWearProfiler:
    """
    Real-time video processing pipeline for wheel wear detection.
    Architecture: YOLO -> Morphology -> Scale -> P/B/H -> Baseline -> Wear Delta
    """
    
    def __init__(self, video_source=0):
        print("\n" + "="*70)
        print("🚆 RAIL TRANSIT SMART EYE - REALTIME WEAR PROFILER")
        print("   Video Mode: Continuous Measurement & Wear Tracking")
        print("="*70)

        # --- Load Camera Calibration ---
        if os.path.exists(CALIB_FILE):
            data = np.load(CALIB_FILE)
            self.mtx = data['mtx']
            self.dist = data['dist']
            print("✅ Camera calibration loaded.")
        else:
            print("❌ Calibration file missing! Run calibrate_camera.py first.")
            sys.exit(1)

        # --- Load YOLO Segmentation Model ---
        if os.path.exists(YOLO_MODEL_PATH):
            self.model = YOLO(YOLO_MODEL_PATH)
            print(f"✅ YOLO model loaded: {YOLO_MODEL_PATH}")
        else:
            print(f"⚠️  YOLO model not found. Falling back to Canny edge detection.")
            self.model = None

        # --- Scale Factor (mm/px) - set once at the beginning ---
        self.k = None

        # --- Baseline (New/Unworn Wheel Profile) ---
        self.baseline_contour_mm = None
        self.baseline_params = None  # {'P': float, 'B': float, 'H': float}
        self.baseline_set = False

        # --- Current measurements ---
        self.current_contour_mm = None
        self.current_params = None

        # --- Video Capture ---
        if isinstance(video_source, int):
            self.cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(video_source)  # Video file path
        
        if not self.cap.isOpened():
            print(f"❌ Cannot open video source: {video_source}")
            sys.exit(1)
        
        # Set high resolution if webcam
        if isinstance(video_source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
        print(f"📷 Video source ready: {self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS")
        print("   (Processing every {FRAME_SKIP} frames for performance)\n")

        # --- FPS tracker ---
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

    # ============================================================
    #  UNDISTORTION (Slide 2.2.1)
    # ============================================================
    def undistort_image(self, img):
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, self.mtx, self.dist, None, new_mtx)
        x, y, w, h = roi
        return dst[y:y+h, x:x+w]

    # ============================================================
    #  SCALE CALIBRATION (Run once at start) - Slide 2.2.4
    # ============================================================
    def calibrate_scale_once(self, frame):
        """User clicks two points on a ruler to set mm/px scale. Runs only once."""
        print("\n📏 SCALE CALIBRATION (One-time setup)")
        print("   Press [ESC] to skip and use dummy scale.")
        
        clone = frame.copy()
        points = []
        
        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                cv2.circle(clone, (x, y), 6, (0, 255, 255), -1)
                cv2.putText(clone, f"P{len(points)}", (x+10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.imshow("Set Scale", clone)

        cv2.imshow("Set Scale", clone)
        cv2.setMouseCallback("Set Scale", mouse_cb)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                self.k = 0.1  # Dummy fallback
                print(f"⚠️  Scale skipped. Using dummy k={self.k:.4f} mm/px")
                return
            if key == 13 and len(points) == 2:  # ENTER
                break

        cv2.destroyAllWindows()
        
        px_dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
        try:
            real_mm = float(input("Enter real distance between points (mm): "))
            self.k = real_mm / px_dist
            print(f"✅ Scale set: 1 pixel = {self.k:.4f} mm")
        except ValueError:
            self.k = 0.1
            print(f"❌ Invalid. Using dummy k={self.k:.4f} mm/px")

    # ============================================================
    #  CONTOUR EXTRACTION (YOLO + Morphology) - Slide 2.2.2 & 2.2.3
    # ============================================================
    def extract_contour(self, img):
        """Extract wheel contour using YOLO segmentation + morphological post-processing."""
        # --- YOLO Segmentation ---
        mask_binary = None
        if self.model is not None:
            results = self.model(img, conf=0.4, verbose=False)
            if results and len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                if len(masks) > 0:
                    mask_binary = (masks[0] * 255).astype(np.uint8)

        # --- Fallback: Canny if YOLO fails ---
        if mask_binary is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            kernel = np.ones((7, 7), np.uint8)
            mask_binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # --- Morphological Post-processing (Slide 2.2.3) ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)  # Close holes
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)        # Remove noise

        # --- Extract largest contour ---
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 500:
            return None

        # --- Smooth contour ---
        epsilon = 0.001 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        contour_points = approx.squeeze()
        
        if contour_points.ndim == 1:
            contour_points = contour_points.reshape(-1, 2)
        
        return contour_points

    # ============================================================
    #  ALIGNMENT & P/B/H CALCULATION
    # ============================================================
    def get_flange_tip(self, contour_mm):
        """Find the flange tip (lowest point in Y)."""
        idx = np.argmax(contour_mm[:, 1])
        return contour_mm[idx]

    def get_tread_line_y(self, contour_mm, scale):
        """Find the tread line (top flat surface)."""
        y_min = contour_mm[:, 1].min()
        threshold = y_min + 10.0 * scale
        tread_points = contour_mm[contour_mm[:, 1] <= threshold]
        if len(tread_points) == 0:
            return y_min
        return tread_points[:, 1].mean()

    def calculate_pbh(self, contour_px):
        """Compute P, B, H in mm."""
        if self.k is None:
            scale = 1.0
        else:
            scale = self.k
        
        contour_mm = contour_px * scale
        self.current_contour_mm = contour_mm

        # Flange tip
        flange_tip = self.get_flange_tip(contour_mm)
        tip_y = flange_tip[1]

        # Tread line
        tread_y = self.get_tread_line_y(contour_mm, scale)

        # H (Rim Width): width of tread points
        y_min = contour_mm[:, 1].min()
        tread_points = contour_mm[contour_mm[:, 1] <= y_min + 10.0 * scale]
        H = tread_points[:, 0].max() - tread_points[:, 0].min() if len(tread_points) > 0 else 0

        # P (Flange Height)
        P = tip_y - tread_y

        # B (Flange Thickness at FLANGE_MEASURE_HEIGHT above tread)
        measure_y = tread_y + FLANGE_MEASURE_HEIGHT
        near_points = contour_mm[np.abs(contour_mm[:, 1] - measure_y) < 3.0 * scale]
        if len(near_points) >= 2:
            B = near_points[:, 0].max() - near_points[:, 0].min()
        else:
            B = flange_tip[0] - tread_points[:, 0].min() if len(tread_points) > 0 else 0

        return {
            'P': P,
            'B': B,
            'H': H,
            'flange_tip': flange_tip,
            'tread_y': tread_y,
            'contour_mm': contour_mm
        }

    # ============================================================
    #  ALIGN CONTOUR FOR WEAR COMPARISON
    # ============================================================
    def align_contour(self, contour_mm, ref_point):
        """Translate contour so that ref_point is at origin (0,0)."""
        return contour_mm - ref_point

    def compute_wear(self, current, baseline, current_params, baseline_params):
        """
        Compute wear deltas (mm).
        Positive delta = material loss (wear).
        """
        if baseline is None or baseline_params is None:
            return None

        wear = {
            'delta_P': baseline_params['P'] - current_params['P'],
            'delta_B': baseline_params['B'] - current_params['B'],
            'delta_H': baseline_params['H'] - current_params['H'],
        }
        return wear

    # ============================================================
    #  VISUALIZATION OVERLAY
    # ============================================================
    def draw_overlay(self, frame, contour_px, params, baseline_set, wear=None):
        """Draw the real-time overlay with profile, parameters, and wear."""
        display = frame.copy()

        # 1. Draw current contour (Green)
        if contour_px is not None and len(contour_px) > 5:
            cv2.drawContours(display, [contour_px.astype(np.int32)], -1, (0, 255, 0), 2)

        # 2. Draw baseline contour (Blue) if set
        if self.baseline_contour_mm is not None and self.k is not None:
            baseline_px = (self.baseline_contour_mm / self.k).astype(np.int32)
            cv2.drawContours(display, [baseline_px], -1, (255, 0, 0), 2)

        # 3. Prepare text overlay
        text_lines = []
        if params:
            text_lines.append(f"P: {params['P']:.2f} mm")
            text_lines.append(f"B: {params['B']:.2f} mm")
            text_lines.append(f"H: {params['H']:.2f} mm")

        if baseline_set and wear:
            text_lines.append("--- WEAR (Loss) ---")
            text_lines.append(f"ΔP: +{wear['delta_P']:.2f} mm" if wear['delta_P'] > 0 else f"ΔP: {wear['delta_P']:.2f} mm")
            text_lines.append(f"ΔB: +{wear['delta_B']:.2f} mm" if wear['delta_B'] > 0 else f"ΔB: {wear['delta_B']:.2f} mm")
            text_lines.append(f"ΔH: +{wear['delta_H']:.2f} mm" if wear['delta_H'] > 0 else f"ΔH: {wear['delta_H']:.2f} mm")
        else:
            text_lines.append("---")
            text_lines.append("Press 'b' to set BASELINE")
            text_lines.append("Press 'r' to reset baseline")

        # 4. Display text on image
        y_pos = 30
        for line in text_lines:
            color = (0, 255, 255) if "WEAR" in line or "Δ" in line else (255, 255, 255)
            if "ΔP" in line or "ΔB" in line or "ΔH" in line:
                # Color code wear: Green = good (<0.5mm), Yellow = moderate, Red = severe
                val = float(line.split(":")[1].strip().split(" ")[0])
                if abs(val) < 0.5:
                    color = (0, 255, 0)  # Green
                elif abs(val) < 1.5:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
            cv2.putText(display, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25

        # 5. FPS display
        cv2.putText(display, f"FPS: {self.current_fps:.1f}", (display.shape[1]-150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # 6. Baseline status
        status = "BASELINE: SET" if baseline_set else "BASELINE: NOT SET"
        cv2.putText(display, status, (display.shape[1]-250, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if baseline_set else (0, 0, 255), 2)

        return display

    # ============================================================
    #  MAIN VIDEO LOOP
    # ============================================================
    def run(self):
        # --- Step 1: Capture first frame for scale calibration ---
        print("📸 Capturing initial frame for scale calibration...")
        ret, first_frame = self.cap.read()
        if not ret:
            print("❌ Cannot read from video source.")
            return
        first_frame = self.undistort_image(first_frame)
        self.calibrate_scale_once(first_frame)

        print("\n" + "="*70)
        print("▶️  REALTIME MEASUREMENT STARTED")
        print("   Key Controls:")
        print("   [b] - Set BASELINE (reference new/unworn wheel)")
        print("   [r] - Reset baseline")
        print("   [q] - Quit")
        print("="*70 + "\n")

        # Reset to beginning of video if file, else continue from frame 0
        if not isinstance(self.cap, cv2.VideoCapture) or not isinstance(IVCAM_INDEX, int):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_counter = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # If video file ends, loop
                if not isinstance(IVCAM_INDEX, int):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_counter += 1
            self.frame_count += 1

            # --- FPS calculation ---
            if time.time() - self.last_fps_time >= 1.0:
                self.current_fps = self.frame_count
                self.frame_count = 0
                self.last_fps_time = time.time()

            # --- Skip frames for performance ---
            if frame_counter % FRAME_SKIP != 0:
                # Still show the frame, but don't process YOLO
                display = self.undistort_image(frame)
                cv2.putText(display, "Processing... (frame skip)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                cv2.imshow("Wheel Wear Monitor", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            # --- Undistort ---
            img_undist = self.undistort_image(frame)

            # --- Extract Contour ---
            contour_px = self.extract_contour(img_undist)
            
            params = None
            wear = None
            
            if contour_px is not None:
                # --- Calculate P/B/H ---
                params = self.calculate_pbh(contour_px)
                self.current_params = params

                # --- Compute wear against baseline ---
                if self.baseline_set and self.baseline_params is not None:
                    wear = self.compute_wear(
                        self.current_contour_mm, 
                        self.baseline_contour_mm,
                        params, 
                        self.baseline_params
                    )

            # --- Draw overlay ---
            display = self.draw_overlay(img_undist, contour_px, params, self.baseline_set, wear)
            cv2.imshow("Wheel Wear Monitor", display)

            # --- Key Handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                # Set baseline to the current profile
                if self.current_contour_mm is not None and self.current_params is not None:
                    self.baseline_contour_mm = self.current_contour_mm.copy()
                    self.baseline_params = self.current_params.copy()
                    self.baseline_set = True
                    print(f"\n✅ BASELINE SET at frame {frame_counter}")
                    print(f"   P: {self.baseline_params['P']:.2f} mm")
                    print(f"   B: {self.baseline_params['B']:.2f} mm")
                    print(f"   H: {self.baseline_params['H']:.2f} mm")
                    print("   --- Now tracking wear ---\n")
                else:
                    print("⚠️  No contour detected. Cannot set baseline.")
            elif key == ord('r'):
                self.baseline_set = False
                self.baseline_contour_mm = None
                self.baseline_params = None
                print("🔄 Baseline reset.")

        # --- Cleanup ---
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Real-time wear monitoring stopped.")


# ============================================================
#  MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # For iVCam (webcam): pass IVCAM_INDEX (e.g., 0, 1, 2)
    # For recorded video file: pass "path/to/video.mp4"
    
    # Change this to your video source
    SOURCE = IVCAM_INDEX  # e.g., 1 for iVCam, or "test_video.mp4"
    
    profiler = RealtimeWearProfiler(video_source=SOURCE)
    profiler.run()