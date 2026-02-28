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

    def detect_platform_from_video(self, video_source, num_frames=10, visualize=False, use_inverse=True):
        """
        Detect platform area by analyzing multiple frames

        Args:
            video_source: Path to video file or camera ID
            num_frames: Number of frames to analyze
            visualize: Show detection process
            use_inverse: Use inverse detection (detect people, then platform is the rest)

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
        all_inverse_masks = []  # For inverse detection
        sample_frame = None

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if sample_frame is None:
                sample_frame = frame.copy()

            # Run YOLO segmentation with lower confidence to catch more detections
            results = self.model.predict(
                source=frame,
                conf=0.25,  # Lower confidence threshold
                verbose=False
            )

            if len(results) == 0 or results[0].masks is None:
                continue

            # Extract masks for floor/ground/platform classes
            # FILTER OUT PEOPLE: COCO class 0 = person
            # We want background/platform, not foreground objects
            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            # Debug: show detected classes
            if frame_idx == frame_indices[0]:  # Only print for first frame
                unique_classes = np.unique(classes)
                print(f"  Detected classes: {unique_classes}")
                print(f"  (Class 0 = person, will be filtered out)")

            # Method 1: Direct detection (exclude foreground objects)
            # Classes to EXCLUDE (people, vehicles, furniture, etc.)
            # COCO classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 4=bus, 5=train, 6=truck
            # 7-25=various objects, 56=chair, 57=couch, etc.
            exclude_classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 24, 25, 26, 27, 28, 56, 57, 58, 59, 60, 61, 62}

            floor_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            people_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            for mask, cls in zip(masks, classes):
                # Resize mask to frame size
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0])
                )

                # Collect people masks separately
                if int(cls) == 0:  # Person class
                    people_mask = cv2.bitwise_or(people_mask, mask_resized)
                # Skip other foreground objects
                elif int(cls) not in exclude_classes:
                    # Accumulate background/floor-like areas
                    floor_mask = cv2.bitwise_or(floor_mask, mask_resized)

            # Method 2: Inverse detection (platform = bottom half - people)
            if use_inverse:
                h, w = frame.shape[:2]
                # Create a mask for the bottom 70% of the frame (where platform is)
                platform_region = np.zeros((h, w), dtype=np.uint8)
                platform_region[int(h * 0.3):, :] = 255

                # Platform = region - people - other objects
                inverse_mask = cv2.bitwise_and(platform_region, cv2.bitwise_not(people_mask))
                all_inverse_masks.append(inverse_mask)

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

        # Try direct detection first, fall back to inverse detection
        combined_mask = None

        if all_masks:
            print(f"  Direct detection: Found masks in {len(all_masks)} frames")
            # Combine masks from multiple frames (voting)
            combined_mask = np.mean(all_masks, axis=0)
            combined_mask = (combined_mask > 0.5).astype(np.uint8) * 255

        # If direct detection failed or found very little, use inverse detection
        if (combined_mask is None or combined_mask.sum() < 1000) and all_inverse_masks:
            print(f"  Using inverse detection: Found masks in {len(all_inverse_masks)} frames")
            combined_mask = np.mean(all_inverse_masks, axis=0)
            combined_mask = (combined_mask > 0.3).astype(np.uint8) * 255  # Lower threshold for inverse

        if combined_mask is None or combined_mask.sum() == 0:
            print("✗ No floor/platform detected in frames")
            return None

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

        # Find the ACTUAL PLATFORM EDGE LINE using contour analysis
        # The platform edge is the boundary line between platform and tracks

        # Strategy: Find the longest continuous edge of the contour
        # For diagonal edges, we need to detect the actual boundary shape

        # Get all contour points
        contour_points = largest_contour.reshape(-1, 2)

        # Find the convex hull to get the outer boundary
        hull = cv2.convexHull(largest_contour, returnPoints=True)
        hull_points = hull.reshape(-1, 2)

        # Analyze the hull to find the most prominent edge (longest side)
        # Calculate distances between consecutive hull points
        edges = []
        for i in range(len(hull_points)):
            pt1 = hull_points[i]
            pt2 = hull_points[(i + 1) % len(hull_points)]
            length = np.linalg.norm(pt2 - pt1)
            angle = np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
            edges.append({
                'start': tuple(pt1.astype(int)),
                'end': tuple(pt2.astype(int)),
                'length': length,
                'angle': angle,
                'midpoint_y': (pt1[1] + pt2[1]) / 2
            })

        # Sort by length to find longest edges
        edges.sort(key=lambda e: e['length'], reverse=True)

        # The platform edge is typically:
        # 1. One of the longest edges
        # 2. In the upper-middle portion of the frame
        # 3. Has a significant angle (for diagonal edges) or is horizontal

        # Take the top 40% longest edges and filter by position
        top_edges = edges[:max(int(len(edges) * 0.4), 2)]

        # Prefer edges in the upper-middle region (where platform edges typically are)
        frame_height = sample_frame.shape[0] if sample_frame is not None else 1000
        preferred_edges = [e for e in top_edges if 0.1 * frame_height < e['midpoint_y'] < 0.7 * frame_height]

        if not preferred_edges:
            preferred_edges = top_edges

        # Select the longest edge in preferred region
        platform_edge = preferred_edges[0]

        # Create edge line with intermediate points for better accuracy
        # Sample points along the longest edge from the original contour
        start_pt = np.array(platform_edge['start'])
        end_pt = np.array(platform_edge['end'])

        # Find all contour points near this edge
        edge_points = []
        edge_line_vec = end_pt - start_pt
        edge_length = np.linalg.norm(edge_line_vec)
        edge_unit_vec = edge_line_vec / edge_length if edge_length > 0 else edge_line_vec

        for pt in contour_points:
            # Project point onto edge line
            v = pt - start_pt
            projection_length = np.dot(v, edge_unit_vec)

            # Check if point is close to the edge line
            if 0 <= projection_length <= edge_length:
                projected_pt = start_pt + projection_length * edge_unit_vec
                dist_to_edge = np.linalg.norm(pt - projected_pt)

                if dist_to_edge < 20:  # Within 20 pixels of edge line
                    edge_points.append(pt)

        # If we found points along the edge, use them; otherwise use hull edge
        if len(edge_points) >= 2:
            edge_points = np.array(edge_points)
            # Sort by position along the edge
            projections = [np.dot(pt - start_pt, edge_unit_vec) for pt in edge_points]
            sorted_indices = np.argsort(projections)
            edge_points = edge_points[sorted_indices]

            # Simplify using Douglas-Peucker
            edge_contour = edge_points.reshape(-1, 1, 2).astype(np.int32)
            epsilon = 0.015 * cv2.arcLength(edge_contour, False)
            simplified_edge = cv2.approxPolyDP(edge_contour, epsilon, False)

            platform_poly = [(int(pt[0][0]), int(pt[0][1])) for pt in simplified_edge]
        else:
            # Fallback: use hull edge endpoints
            platform_poly = [platform_edge['start'], platform_edge['end']]

        # Visualize result
        if visualize and sample_frame is not None:
            vis = sample_frame.copy()
            # Draw edge line (not closed polygon)
            cv2.polylines(vis, [np.array(platform_poly)], False, (0, 255, 0), 3)
            # Draw edge points
            for pt in platform_poly:
                cv2.circle(vis, pt, 6, (0, 0, 255), -1)
            cv2.putText(vis, "Detected Platform Edge Line", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis, f"{len(platform_poly)} points", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Result", vis)
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"✓ YOLO-seg detected platform edge with {len(platform_poly)} points")
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

        all_lines = []  # Store all detected lines
        sample_frame = None

        # Sample frames uniformly
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, max(total_frames-1, 0), num_frames, dtype=int)

        print(f"  Analyzing {num_frames} frames for edge lines...")

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if sample_frame is None:
                sample_frame = frame.copy()

            # Focus on middle portion of frame (where platform edge usually is)
            roi_y_start = h // 4
            roi_y_end = int(h * 0.75)
            roi = frame[roi_y_start:roi_y_end, 0:w]

            # Convert to grayscale and detect edges
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Apply preprocessing for better edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

            # Detect lines with adjusted parameters for diagonal detection
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=60,        # Lower threshold to catch more lines
                minLineLength=150,   # Longer lines only
                maxLineGap=30
            )

            if lines is None:
                continue

            # Collect ALL strong lines (not just horizontal)
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Convert to full frame coordinates
                y1_full = y1 + roi_y_start
                y2_full = y2 + roi_y_start

                # Calculate line properties
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                # Store line with metadata
                all_lines.append({
                    'start': (x1, y1_full),
                    'end': (x2, y2_full),
                    'length': length,
                    'angle': angle,
                    'midpoint_y': (y1_full + y2_full) / 2
                })

        cap.release()

        if not all_lines:
            print("✗ No edge lines detected")
            return None

        # Find the most prominent edge line
        # Strategy: Find the longest line that appears consistently

        # Group similar lines (same angle and position)
        line_groups = []
        for line in all_lines:
            # Find if this line belongs to an existing group
            grouped = False
            for group in line_groups:
                # Check if angle and position are similar
                angle_diff = abs(line['angle'] - group['avg_angle'])
                y_diff = abs(line['midpoint_y'] - group['avg_y'])

                if angle_diff < 15 and y_diff < 50:  # Similar line
                    group['lines'].append(line)
                    group['total_length'] += line['length']
                    group['avg_angle'] = np.mean([l['angle'] for l in group['lines']])
                    group['avg_y'] = np.mean([l['midpoint_y'] for l in group['lines']])
                    grouped = True
                    break

            if not grouped:
                # Create new group
                line_groups.append({
                    'lines': [line],
                    'total_length': line['length'],
                    'avg_angle': line['angle'],
                    'avg_y': line['midpoint_y']
                })

        # Sort groups by total length (strongest/most consistent lines)
        line_groups.sort(key=lambda g: g['total_length'], reverse=True)

        if not line_groups:
            print("✗ No consistent edge lines found")
            return None

        # Take the strongest group (most prominent edge)
        best_group = line_groups[0]
        print(f"  Found edge line group: {len(best_group['lines'])} detections, angle={best_group['avg_angle']:.1f}°")

        # Get the longest line from the best group
        longest_line = max(best_group['lines'], key=lambda l: l['length'])

        # Create edge line (just 2 endpoints)
        platform_poly = [
            longest_line['start'],
            longest_line['end']
        ]

        print(f"  Edge line: {platform_poly[0]} → {platform_poly[1]}")

        # Visualize
        if visualize and sample_frame is not None:
            vis = sample_frame.copy()

            # Draw the detected edge line
            cv2.line(vis, platform_poly[0], platform_poly[1], (0, 255, 255), 4)

            # Draw endpoints
            cv2.circle(vis, platform_poly[0], 8, (0, 0, 255), -1)
            cv2.circle(vis, platform_poly[0], 10, (255, 255, 255), 2)
            cv2.circle(vis, platform_poly[1], 8, (0, 0, 255), -1)
            cv2.circle(vis, platform_poly[1], 10, (255, 255, 255), 2)

            # Display info
            cv2.putText(vis, "Hough Transform Edge Detection", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(vis, f"Angle: {best_group['avg_angle']:.1f}°", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(vis, f"2 points (diagonal edge line)", (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Result", vis)
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"✓ Hough detected platform edge line: {len(platform_poly)} points, angle={best_group['avg_angle']:.1f}°")
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

    def load_calibration(self, camera_id, target_dimensions=None):
        """
        Load existing calibration for a camera

        Args:
            camera_id: Camera identifier
            target_dimensions: (width, height) to scale coords to, or None for original

        Returns:
            platform_polygon scaled to target dimensions, or None if not found
        """
        config_path = self.get_config_path(camera_id)

        if not config_path.exists():
            return None

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Support both old and new config formats
            if 'platform_edge' in config:
                # New format with normalized coordinates
                normalized = config['platform_edge']['normalized']

                if target_dimensions:
                    # Scale to target dimensions
                    width, height = target_dimensions
                    scaled_poly = [
                        (int(x * width), int(y * height))
                        for x, y in normalized
                    ]
                    print(f"✓ Loaded calibration for '{camera_id}' (scaled to {width}x{height})")
                    return scaled_poly
                else:
                    # Use original absolute coordinates
                    print(f"✓ Loaded calibration for camera '{camera_id}'")
                    return config['platform_edge']['absolute']

            elif 'platform_polygon' in config:
                # Old format (absolute coordinates only)
                print(f"✓ Loaded calibration for camera '{camera_id}' (legacy format)")
                return config['platform_polygon']

            else:
                return None

        except Exception as e:
            print(f"⚠ Error loading calibration: {e}")
            return None

    def save_calibration(self, camera_id, platform_poly, video_source, method, video_dims=None):
        """
        Save calibration to file with NORMALIZED coordinates

        Args:
            camera_id: Camera identifier
            platform_poly: List of (x, y) absolute pixel coordinates
            video_source: Path to video file
            method: Detection method used
            video_dims: (width, height) tuple, or None to read from video
        """
        # Get video dimensions if not provided
        if video_dims is None:
            cap = cv2.VideoCapture(video_source)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            width, height = video_dims

        # Normalize coordinates (convert to 0.0-1.0 range)
        normalized_poly = [
            [round(x / width, 4), round(y / height, 4)]
            for x, y in platform_poly
        ]

        config = {
            'camera_id': camera_id,
            'video_dimensions': {'width': width, 'height': height},
            'platform_edge': {
                'normalized': normalized_poly,  # 0.0-1.0 range (resolution-independent)
                'absolute': platform_poly        # Original pixel coordinates
            },
            'video_source': video_source,
            'detection_method': method,
            'num_points': len(platform_poly)
        }

        config_path = self.get_config_path(camera_id)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✓ Saved calibration to {config_path}")
        print(f"  Original resolution: {width}x{height}")
        print(f"  Normalized coordinates: {normalized_poly}")

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
                    num_frames=15,  # Increased from 10 for better accuracy
                    visualize=visualize,
                    use_inverse=True  # Enable inverse detection
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

            # Draw platform edge line (not closed)
            cv2.polylines(frame, [np.array(platform_poly)], False, (0, 255, 0), 4)

            # Draw edge points
            for i, pt in enumerate(platform_poly):
                cv2.circle(frame, pt, 6, (0, 0, 255), -1)
                cv2.circle(frame, pt, 8, (255, 255, 255), 2)

            # Info overlay
            cv2.putText(frame, f"Camera: {camera_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Platform Edge Line ({len(platform_poly)} points)", (10, 60),
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
