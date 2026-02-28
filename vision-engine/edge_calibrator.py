"""
Automatic Platform Edge Detection System
==========================================

Three-tier approach:
1. YOLO-seg: Semantic segmentation to find platform/floor area (PRIMARY)
2. Hough Transform: Line detection for platform edges (FALLBACK)
3. Manual Calibration: Interactive UI for manual selection (LAST RESORT)

Usage:
    python edge_calibrator.py --video sample-station.webm --camera platform_cam_1

    Or import and use:
    from edge_calibrator import CalibrationManager
    manager = CalibrationManager()
    platform_poly = manager.calibrate("sample-station.webm", "platform_cam_1")
"""

import cv2
import numpy as np
import json
import os
import argparse
from pathlib import Path
from ultralytics import YOLO


class YOLOSegEdgeDetector:
    """Primary method: Use YOLO segmentation to detect platform area"""

    def __init__(self, model_path="yolo26s-seg.pt"):
        """
        Initialize YOLO segmentation model

        Args:
            model_path: Path to YOLO-seg model
            Recommended: yolo26s-seg.pt (balanced) or yolo26m-seg.pt (higher accuracy)
            Available: yolo26n-seg, yolo26s-seg, yolo26m-seg, yolo26l-seg, yolo26x-seg
        """
        print(f"Loading YOLO-seg model: {model_path}")
        try:
            self.model = YOLO(model_path)
            self.available = True
            print("✓ YOLO-seg model loaded")
        except Exception as e:
            print(f"⚠ Could not load YOLO-seg: {e}")
            self.available = False

    def detect_platform_from_video(self, video_source, num_frames=10, visualize=False):
        """
        Detect platform area by analyzing multiple frames

        Args:
            video_source: Path to video file or camera ID
            num_frames: Number of frames to analyze
            visualize: Show detection process

        Returns:
            platform_polygon: List of (x, y) points defining platform boundary
        """
        if not self.available:
            return None

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"✗ Could not open video: {video_source}")
            return None

        # Sample frames from different parts of video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, max(total_frames-1, 0), num_frames, dtype=int)

        all_masks = []
        sample_frame = None

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if sample_frame is None:
                sample_frame = frame.copy()

            # Run YOLO segmentation
            results = self.model.predict(
                source=frame,
                conf=0.3,
                verbose=False
            )

            if len(results) == 0 or results[0].masks is None:
                continue

            # Extract masks for floor/ground/platform classes
            # Common COCO classes: floor (59), pavement (11), ground, etc.
            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            # Look for floor-like classes (adjust based on your model's classes)
            # COCO: 0=person, 1=bicycle, ... Check what your model detects
            floor_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            for mask, cls in zip(masks, classes):
                # Resize mask to frame size
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0])
                )
                # Accumulate all floor-like areas
                floor_mask = cv2.bitwise_or(floor_mask, mask_resized)

            if floor_mask.sum() > 0:
                all_masks.append(floor_mask)

                if visualize:
                    vis = frame.copy()
                    vis[floor_mask > 0] = vis[floor_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                    cv2.imshow("YOLO-seg Detection", vis)
                    cv2.waitKey(100)

        cap.release()
        if visualize:
            cv2.destroyAllWindows()

        if not all_masks:
            print("✗ No floor/platform detected in frames")
            return None

        # Combine masks from multiple frames (voting)
        combined_mask = np.mean(all_masks, axis=0)
        combined_mask = (combined_mask > 0.5).astype(np.uint8) * 255

        # === SMOOTH THE MASK TO REMOVE JAGGED EDGES ===
        # Apply Gaussian blur to smooth the mask
        combined_mask = cv2.GaussianBlur(combined_mask, (15, 15), 0)
        _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

        # Morphological operations to smooth edges
        kernel_size = 25  # Larger kernel = more smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Closing: removes small holes and smooths contours
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Opening: removes small noise and smooths further
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        # ===============================================

        # Find contours of platform area
        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            print("✗ No contours found in segmentation mask")
            return None

        # Get largest contour (main platform area)
        largest_contour = max(contours, key=cv2.contourArea)

        # Simplify polygon (reduce number of points & smooth spikes)
        # Increase epsilon for more aggressive smoothing (0.01 -> 0.02)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_poly = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Optional: Use convex hull for maximum smoothness (removes ALL concavities)
        # Uncomment next line if you want perfectly smooth boundary (no indentations)
        # approx_poly = cv2.convexHull(largest_contour)

        # Convert to list of tuples
        platform_poly = [(int(pt[0][0]), int(pt[0][1])) for pt in approx_poly]

        # Visualize result
        if visualize and sample_frame is not None:
            vis = sample_frame.copy()
            cv2.polylines(vis, [np.array(platform_poly)], True, (0, 255, 0), 3)
            cv2.putText(vis, "Detected Platform Boundary", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Result", vis)
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"✓ YOLO-seg detected platform with {len(platform_poly)} vertices")
        return platform_poly


class HoughEdgeDetector:
    """Fallback method: Use Hough Transform to detect straight edges"""

    def detect_platform_from_video(self, video_source, num_frames=30, visualize=False):
        """
        Detect platform edge using line detection

        Args:
            video_source: Path to video file
            num_frames: Number of frames to analyze
            visualize: Show detection process

        Returns:
            platform_polygon: List of (x, y) points
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"✗ Could not open video: {video_source}")
            return None

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        all_edge_y = []
        frame_count = 0
        sample_frame = None

        # Sample frames uniformly
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, max(total_frames-1, 0), num_frames, dtype=int)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if sample_frame is None:
                sample_frame = frame.copy()

            # Focus on bottom half of frame (where platform edge usually is)
            roi_y_start = h // 3
            roi = frame[roi_y_start:h, 0:w]

            # Convert to grayscale and detect edges
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Detect lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=80,
                minLineLength=100,
                maxLineGap=50
            )

            if lines is None:
                continue

            # Filter for horizontal lines (platform edge)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

                # Keep lines within 15 degrees of horizontal
                if angle < 15 or angle > 165:
                    # Convert back to full frame coordinates
                    actual_y = y1 + roi_y_start
                    all_edge_y.append(actual_y)

            frame_count += 1

        cap.release()

        if not all_edge_y:
            print("✗ No horizontal edges detected")
            return None

        # Use median y-coordinate for stability
        stable_y = int(np.median(all_edge_y))

        # Create platform polygon (rectangle from edge to bottom)
        platform_poly = [
            (50, stable_y),           # Top-left (with margin)
            (w - 50, stable_y),       # Top-right (with margin)
            (w - 50, h - 50),         # Bottom-right (with margin)
            (50, h - 50)              # Bottom-left (with margin)
        ]

        # Visualize
        if visualize and sample_frame is not None:
            vis = sample_frame.copy()
            cv2.polylines(vis, [np.array(platform_poly)], True, (0, 255, 255), 3)
            cv2.line(vis, (0, stable_y), (w, stable_y), (255, 0, 0), 2)
            cv2.putText(vis, "Hough Transform Edge Detection", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Result", vis)
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"✓ Hough detected platform edge at y={stable_y}")
        return platform_poly


class ManualCalibrator:
    """Last resort: Interactive manual calibration"""

    def __init__(self):
        self.points = []
        self.frame = None
        self.window_name = "Manual Platform Calibration"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"  Point {len(self.points)}: ({x}, {y})")

    def calibrate(self, video_source):
        """
        Interactive calibration: Click to define platform boundary

        Instructions:
        - Click points around platform boundary (at least 3 points)
        - Press 's' to save
        - Press 'r' to reset
        - Press 'q' to cancel

        Returns:
            platform_polygon: List of (x, y) points
        """
        cap = cv2.VideoCapture(video_source)
        ret, self.frame = cap.read()
        cap.release()

        if not ret:
            print("✗ Could not read video frame")
            return None

        print("\n" + "="*60)
        print("MANUAL CALIBRATION MODE")
        print("="*60)
        print("Instructions:")
        print("  1. Click points around the platform boundary")
        print("  2. You need at least 3 points to form a polygon")
        print("  3. Press 's' to SAVE the calibration")
        print("  4. Press 'r' to RESET and start over")
        print("  5. Press 'q' to CANCEL")
        print("="*60 + "\n")

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            display = self.frame.copy()

            # Draw points
            for i, pt in enumerate(self.points):
                cv2.circle(display, pt, 6, (0, 255, 0), -1)
                cv2.circle(display, pt, 8, (255, 255, 255), 2)
                cv2.putText(display, str(i+1), (pt[0]+12, pt[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw polygon
            if len(self.points) >= 2:
                cv2.polylines(display, [np.array(self.points)],
                             len(self.points) > 2, (0, 255, 0), 2)

            # Instructions overlay
            cv2.rectangle(display, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
            cv2.rectangle(display, (0, 0), (display.shape[1], 80), (255, 255, 255), 2)

            cv2.putText(display, "Click to add points | 's' = Save | 'r' = Reset | 'q' = Cancel",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Points: {len(self.points)} (need at least 3)",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if len(self.points) >= 3:
                    print(f"✓ Saved {len(self.points)} points")
                    cv2.destroyAllWindows()
                    return self.points
                else:
                    print("⚠ Need at least 3 points!")

            elif key == ord('r'):
                self.points = []
                print("Reset points")

            elif key == ord('q'):
                print("✗ Calibration cancelled")
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()
        return None


class CalibrationManager:
    """
    Unified calibration manager with three-tier approach:
    1. Try YOLO-seg (best for semantic understanding)
    2. Fall back to Hough Transform (good for clear edges)
    3. Fall back to manual calibration (always works)
    """

    def __init__(self, config_dir="configs"):
        """
        Initialize calibration manager

        Args:
            config_dir: Directory to store calibration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

    def get_config_path(self, camera_id):
        """Get config file path for a specific camera"""
        return self.config_dir / f"{camera_id}_calibration.json"

    def load_calibration(self, camera_id):
        """
        Load existing calibration for a camera

        Returns:
            platform_polygon or None if not found
        """
        config_path = self.get_config_path(camera_id)

        if not config_path.exists():
            return None

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            print(f"✓ Loaded calibration for camera '{camera_id}'")
            return config['platform_polygon']

        except Exception as e:
            print(f"⚠ Error loading calibration: {e}")
            return None

    def save_calibration(self, camera_id, platform_poly, video_source, method):
        """Save calibration to file"""
        config = {
            'camera_id': camera_id,
            'platform_polygon': platform_poly,
            'video_source': video_source,
            'detection_method': method,
            'num_vertices': len(platform_poly)
        }

        config_path = self.get_config_path(camera_id)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Saved calibration to {config_path}")

    def calibrate(self, video_source, camera_id, method='auto', visualize=False, force_recalibrate=False):
        """
        Calibrate platform boundary for a camera

        Args:
            video_source: Path to video file or camera ID
            camera_id: Unique identifier for this camera
            method: 'auto' (try all), 'yolo', 'hough', 'manual'
            visualize: Show detection process
            force_recalibrate: Ignore existing calibration

        Returns:
            platform_polygon: List of (x, y) points
        """
        print("\n" + "="*60)
        print(f"CALIBRATING CAMERA: {camera_id}")
        print("="*60)

        # Check for existing calibration
        if not force_recalibrate:
            existing = self.load_calibration(camera_id)
            if existing is not None:
                print("Using existing calibration. Use --force to recalibrate.")
                return existing

        platform_poly = None
        detection_method = None

        # Method 1: YOLO-seg (PRIMARY)
        if method in ['auto', 'yolo']:
            print("\n[1/3] Trying YOLO-seg segmentation...")
            try:
                detector = YOLOSegEdgeDetector()
                platform_poly = detector.detect_platform_from_video(
                    video_source,
                    num_frames=10,
                    visualize=visualize
                )
                if platform_poly:
                    detection_method = 'yolo-seg'
                    print("✓ Success with YOLO-seg!")
            except Exception as e:
                print(f"⚠ YOLO-seg failed: {e}")

        # Method 2: Hough Transform (FALLBACK)
        if platform_poly is None and method in ['auto', 'hough']:
            print("\n[2/3] Trying Hough Transform edge detection...")
            try:
                detector = HoughEdgeDetector()
                platform_poly = detector.detect_platform_from_video(
                    video_source,
                    num_frames=30,
                    visualize=visualize
                )
                if platform_poly:
                    detection_method = 'hough-transform'
                    print("✓ Success with Hough Transform!")
            except Exception as e:
                print(f"⚠ Hough Transform failed: {e}")

        # Method 3: Manual Calibration (LAST RESORT)
        if platform_poly is None and method in ['auto', 'manual']:
            print("\n[3/3] Automatic detection failed. Starting manual calibration...")
            try:
                calibrator = ManualCalibrator()
                platform_poly = calibrator.calibrate(video_source)
                if platform_poly:
                    detection_method = 'manual'
                    print("✓ Manual calibration complete!")
            except Exception as e:
                print(f"✗ Manual calibration failed: {e}")

        # Save calibration
        if platform_poly:
            self.save_calibration(camera_id, platform_poly, video_source, detection_method)
            print(f"\n{'='*60}")
            print(f"✓ CALIBRATION COMPLETE")
            print(f"  Method: {detection_method}")
            print(f"  Vertices: {len(platform_poly)}")
            print(f"  Camera: {camera_id}")
            print(f"{'='*60}\n")
            return platform_poly
        else:
            print("\n✗ Calibration failed with all methods")
            return None

    def visualize_calibration(self, video_source, camera_id):
        """Show calibration overlay on video"""
        platform_poly = self.load_calibration(camera_id)

        if platform_poly is None:
            print(f"✗ No calibration found for camera '{camera_id}'")
            return

        cap = cv2.VideoCapture(video_source)

        print("\nShowing calibration overlay...")
        print("Press 'q' to quit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue

            # Draw platform boundary
            cv2.polylines(frame, [np.array(platform_poly)], True, (0, 255, 0), 3)

            # Draw vertices
            for i, pt in enumerate(platform_poly):
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)

            # Info overlay
            cv2.putText(frame, f"Camera: {camera_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Platform Boundary ({len(platform_poly)} vertices)", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Platform Calibration", frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Automatic Platform Edge Detection & Calibration"
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--camera', type=str, required=True,
                       help='Camera ID (e.g., platform_cam_1)')
    parser.add_argument('--method', type=str, default='auto',
                       choices=['auto', 'yolo', 'hough', 'manual'],
                       help='Detection method (default: auto)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show detection process')
    parser.add_argument('--force', action='store_true',
                       help='Force recalibration even if config exists')
    parser.add_argument('--show', action='store_true',
                       help='Show calibration overlay on video')

    args = parser.parse_args()

    manager = CalibrationManager()

    if args.show:
        # Just show existing calibration
        manager.visualize_calibration(args.video, args.camera)
    else:
        # Run calibration
        platform_poly = manager.calibrate(
            video_source=args.video,
            camera_id=args.camera,
            method=args.method,
            visualize=args.visualize,
            force_recalibrate=args.force
        )

        if platform_poly:
            print("\nTo view the calibration overlay, run:")
            print(f"  python edge_calibrator.py --video {args.video} --camera {args.camera} --show")


if __name__ == "__main__":
    main()
