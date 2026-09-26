# wheel_profiler_v2.py
# ====================================================================
#  REVISED BASED ON "轨交明眸" (Rail Transit Smart Eye) PPT
#  Architecture: YOLO Segmentation -> Mask Post-processing -> Scale -> P/B/H
# ====================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from ultralytics import YOLO
import sys

# ============================================================
#  CONFIGURATION (Slide 2.1 - User & Platform Layer)
# ============================================================
CALIB_FILE = "calibration.npz"
YOLO_MODEL_PATH = "yolov8n-seg.pt"  # Or your custom trained best.pt
IVCAM_INDEX = 1  # Change to the index where iVCam is detected

# Wheel measurement standards (Slide 5.1)
FLANGE_MEASURE_HEIGHT = 12.0  # mm (Standard height for measuring thickness B)

class WheelProfilerV2:
    """
    Implements the 6-layer architecture from the PPT:
    Data Resource -> Technical Support -> Application -> System -> Presentation -> User
    """
    def __init__(self):
        print("\n" + "="*60)
        print("🚆 RAIL TRANSIT SMART EYE - WHEEL PROFILER V2")
        print("   Based on YOLO Segmentation + Morphological Post-processing")
        print("="*60)

        # --- Data Resource Layer ---
        # 1. Load Camera Calibration (Slide 2.2.1)
        if os.path.exists(CALIB_FILE):
            data = np.load(CALIB_FILE)
            self.mtx = data['mtx']
            self.dist = data['dist']
            print("✅ Camera calibration loaded.")
        else:
            print("❌ Calibration file missing! Run calibrate_camera.py first.")
            sys.exit(1)

        # 2. Load YOLO Segmentation Model (Slide 2.2.2 / 3.2)
        if os.path.exists(YOLO_MODEL_PATH):
            self.model = YOLO(YOLO_MODEL_PATH)
            print(f"✅ YOLO model loaded: {YOLO_MODEL_PATH}")
        else:
            print(f"⚠️  YOLO model not found at {YOLO_MODEL_PATH}.")
            print("   Pipeline will fallback to Canny edge detection if YOLO fails.")
            self.model = None

        # Scale factor (mm/px) - will be set later (Slide 2.2.4)
        self.k = None  
        self.contour_mm = None

        # --- Connect to iVCam (User Layer) ---
        self.cap = cv2.VideoCapture(IVCAM_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"❌ Cannot open iVCam at index {IVCAM_INDEX}.")
            print("   Try changing IVCAM_INDEX in the config.")
            sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        print(f"📷 Camera ready: {int(self.cap.get(3))}x{int(self.cap.get(4))}\n")

    # ============================================================
    #  TECHNICAL SUPPORT LAYER: Undistortion (Slide 2.2.1)
    # ============================================================
    def undistort_image(self, img):
        """Removes lens distortion to maintain a unified geometric framework."""
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, self.mtx, self.dist, None, new_mtx)
        x, y, w, h = roi
        return dst[y:y+h, x:x+w]

    # ============================================================
    #  APPLICATION LAYER: Capture & Scale Calibration
    # ============================================================
    def capture_image(self):
        """Grabs a single high-quality frame from the camera."""
        print("\n📸 Aim camera at the WHEEL SIDE PROFILE (flange must be visible).")
        print("   Press [SPACE] to capture, [ESC] to cancel.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            display = self.undistort_image(frame)
            cv2.putText(display, "Aim at wheel. Press SPACE to capture.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Capture", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                cv2.destroyAllWindows()
                return self.undistort_image(frame)
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None

    def calibrate_scale_from_reference(self, img):
        """
        Multi-segment joint estimation for Pixel-to-MM conversion (Slide 2.2.4).
        User clicks two points on a reference object (ruler) and enters the real length.
        """
        print("\n📏 SCALE CALIBRATION (Pixel <-> mm)")
        print("   Click on two points on a ruler (or known reference).")
        print("   Enter the real distance in mm. Press ENTER to confirm.")
        
        clone = img.copy()
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
                return False
            if key == 13 and len(points) == 2:  # ENTER
                break

        cv2.destroyAllWindows()
        
        px_dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
        try:
            real_mm = float(input(f"Enter real distance between points (mm): "))
            self.k = real_mm / px_dist
            print(f"✅ Scale set: 1 pixel = {self.k:.4f} mm")
            return True
        except ValueError:
            print("❌ Invalid input.")
            return False

    # ============================================================
    #  CORE PIPELINE: YOLO + Morphology (Slide 2.2.2 & 2.2.3)
    # ============================================================
    def extract_contour(self, img):
        """
        Implements the 'First Locate, Then Precisely Extract' two-stage pipeline.
        Stage 1: YOLO Semantic Segmentation (or Fallback).
        Stage 2: Morphological Post-processing (Closing -> Opening -> Filter).
        """
        # --- Stage 1: YOLO Segmentation (Slide 2.2.2) ---
        mask_binary = None
        if self.model is not None:
            results = self.model(img, conf=0.5, verbose=False)
            if results and len(results) > 0 and results[0].masks is not None:
                # Get the first detected mask (assumes the wheel is the main object)
                masks = results[0].masks.data.cpu().numpy()
                if len(masks) > 0:
                    # Convert to binary mask (0-255)
                    mask_binary = (masks[0] * 255).astype(np.uint8)
                    print("   ✅ YOLO mask detected.")

        # --- Stage 2: Fallback (if YOLO fails or no model) ---
        if mask_binary is None:
            print("   ⚠️  YOLO mask missing. Falling back to Classical Canny (Slide 2.2.2 Fallback).")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            # Morphological close to connect edges into a solid blob
            kernel = np.ones((7, 7), np.uint8)
            mask_binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # --- Stage 3: Post-processing (Slide 2.2.3) ---
        # 3a. Morphological Operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Closing (闭运算): Fill small holes, connect tiny fractures
        closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        # Opening (开运算): Remove discrete noise points
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # 3b. Connected Domain Filtering: Keep only the largest target
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            print("❌ No contours found in the mask.")
            return None
        
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 500:
            print("❌ Largest contour is too small (noise).")
            return None

        # 3c. Contour Smoothing (抑制局部尖刺) - Slide 2.2.3
        epsilon = 0.001 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        contour_points = approx.squeeze()
        
        if contour_points.ndim == 1:
            contour_points = contour_points.reshape(-1, 2)
            
        print(f"   ✅ Contour extracted with {len(contour_points)} points.")
        return contour_points

    # ============================================================
    #  APPLICATION LAYER: P/B/H Calculation (Slide 5.1)
    # ============================================================
    def calculate_pbh(self, contour_px):
        """Calculates P (Flange Height), B (Thickness), H (Rim Width) in mm."""
        if self.k is None:
            print("⚠️  Scale not set. Measurements are in pixels.")
            scale = 1.0
        else:
            scale = self.k

        contour_mm = contour_px * scale
        self.contour_mm = contour_mm

        # Find the bounding extremes
        x_min, x_max = contour_mm[:, 0].min(), contour_mm[:, 0].max()
        y_min, y_max = contour_mm[:, 1].min(), contour_mm[:, 1].max()

        # --- 1. Flange Tip (Highest Y in image -> physically lowest point) ---
        tip_idx = np.argmax(contour_mm[:, 1])
        flange_tip = contour_mm[tip_idx]

        # --- 2. Tread Line (The flat rolling surface at the top) ---
        # We take the top 10% of points in Y (minimum Y)
        tread_threshold = np.percentile(contour_mm[:, 1], 10)
        tread_points = contour_mm[contour_mm[:, 1] <= tread_threshold + 5.0 * scale]
        
        if len(tread_points) < 2:
            # Fallback: just use the min Y
            tread_y = y_min
        else:
            tread_y = tread_points[:, 1].mean()  # Average Y of the tread

        # --- 3. H (Rim Width): Horizontal span of the tread ---
        tread_x_min = tread_points[:, 0].min() if len(tread_points) > 0 else x_min
        tread_x_max = tread_points[:, 0].max() if len(tread_points) > 0 else x_max
        H = tread_x_max - tread_x_min

        # --- 4. P (Flange Height): Vertical distance from tip to tread line ---
        P = flange_tip[1] - tread_y

        # --- 5. B (Flange Thickness): Horizontal width at FLANGE_MEASURE_HEIGHT above tread ---
        measure_y = tread_y + FLANGE_MEASURE_HEIGHT
        # Find points near this specific Y level
        near_points = contour_mm[np.abs(contour_mm[:, 1] - measure_y) < 3.0 * scale]
        
        if len(near_points) >= 2:
            B = near_points[:, 0].max() - near_points[:, 0].min()
        else:
            # If not enough points, estimate using the flange tip x and tread inner edge
            B = flange_tip[0] - tread_x_min

        results = {
            'P (Flange Height) mm': P,
            'B (Flange Thickness) mm': B,
            'H (Rim Width) mm': H,
            'Scale (mm/px)': self.k,
            'Contour (mm)': contour_mm
        }
        return results

    # ============================================================
    #  PRESENTATION LAYER: Visualization & Report (Slide 5.1/5.2)
    # ============================================================
    def visualize_and_save(self, img, contour_px, results):
        """Generates the final report plot and saves CSV."""
        contour_mm = results['Contour (mm)']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- Plot 1: Original + Overlay, Plot 2: 2D Profile ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left: Original Image with detected contour
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.plot(contour_px[:, 0], contour_px[:, 1], 'g-', linewidth=2, label='YOLO+Post-process Contour')
        ax1.set_title("Step 1: Segmentation & Contour Extraction", fontsize=12)
        ax1.axis('off')
        ax1.legend()

        # Right: Physical Profile (mm) with P/B/H annotations
        ax2.plot(contour_mm[:, 0], contour_mm[:, 1], 'b-', linewidth=1.5)
        ax2.set_title("Step 2: Physical Profile & P/B/H Parameters", fontsize=12)
        ax2.set_xlabel("Width (mm)")
        ax2.set_ylabel("Height (mm)")
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Makes the profile look like a wheel (flange at bottom)

        # Annotations
        textstr = (f'P (Flange Height): {results["P (Flange Height) mm"]:.2f} mm\n'
                   f'B (Flange Thickness): {results["B (Flange Thickness) mm"]:.2f} mm\n'
                   f'H (Rim Width): {results["H (Rim Width) mm"]:.2f} mm')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10, 
                 verticalalignment='top', bbox=props)

        plt.tight_layout()
        plot_file = f"wheel_report_{timestamp}.png"
        plt.savefig(plot_file, dpi=150)
        print(f"💾 Report plot saved: {plot_file}")
        plt.show()

        # --- Save CSV for deeper analysis (Slide 5.2 - Batch Analysis) ---
        csv_file = f"wheel_contour_{timestamp}.csv"
        np.savetxt(csv_file, contour_mm, delimiter=',', header='x_mm,y_mm', comments='')
        print(f"💾 Contour CSV saved: {csv_file}")

    # ============================================================
    #  SYSTEM EXECUTION (Slide 2.1)
    # ============================================================
    def run(self):
        # Step 1: Capture Image
        img = self.capture_image()
        if img is None:
            print("❌ Capture cancelled.")
            return

        # Step 2: Scale Calibration (Slide 2.2.4)
        if self.k is None:
            print("\n⚠️  Scale not set. Calibrating using reference...")
            if not self.calibrate_scale_from_reference(img):
                print("❌ Scale calibration cancelled. Using dummy k=0.1 mm/px.")
                self.k = 0.1

        # Step 3: Extract Contour (YOLO + Morphology) - Slide 2.2.2 & 2.2.3
        contour_px = self.extract_contour(img)
        if contour_px is None:
            print("❌ Contour extraction failed.")
            return

        # Step 4: Calculate P/B/H - Slide 5.1
        results = self.calculate_pbh(contour_px)
        
        # Print results to console
        print("\n" + "="*60)
        print("📊 FINAL MEASUREMENT RESULTS (Slide 5.1)")
        print(f"   P (Flange Height)    : {results['P (Flange Height) mm']:.2f} mm")
        print(f"   B (Flange Thickness) : {results['B (Flange Thickness) mm']:.2f} mm")
        print(f"   H (Rim Width)        : {results['H (Rim Width) mm']:.2f} mm")
        print(f"   Scale Factor         : {self.k:.4f} mm/px")
        print("="*60)

        # Step 5: Visualization & Export (Slide 5.2)
        self.visualize_and_save(img, contour_px, results)

        # Release camera
        self.cap.release()
        print("\n✅ Wheel profiling complete. Ready for next measurement.")


# ============================================================
#  MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    profiler = WheelProfilerV2()
    profiler.run()