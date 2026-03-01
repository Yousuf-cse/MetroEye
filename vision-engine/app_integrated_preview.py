"""
Suspicious Behavior Detection - COMPLETE INTEGRATED VERSION
============================================================

Features:
1. ✓ Auto-calibration (edge detection) OR Manual calibration (click polygon)
2. ✓ YOLO pose detection + tracking
3. ✓ Feature extraction (speed, edge distance, torso angle, dwell time)
4. ✓ Brain logic (aggregation + rule-based scoring + LLM analysis)
5. ✓ API-ready structure (commented out, ready to enable)

Usage:
    # Auto-calibration (automatic edge detection)
    python app_integrated_preview.py --video sample-station.webm --camera platform_cam_1

    # Manual calibration (click polygon points)
    python app_integrated_preview.py --video sample-station.webm --camera platform_cam_1 --calibration manual

    # Force recalibration
    python app_integrated_preview.py --video sample-station.webm --camera platform_cam_1 --recalibrate

    # Visualize calibration process
    python app_integrated_preview.py --video sample-station.webm --camera platform_cam_1 --visualize-calibration
"""

import time
import csv
import os
import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ============ Import auto-calibration system ============
from edge_calibrator import CalibrationManager
# ========================================================

# ============ Import brain modules ============
from brain.feature_aggregator import FeatureAggregator
from brain.rule_based_scorer import RuleBasedScorer
from brain.llm_analyzer import LLMAnalyzer
# ==============================================

# ---------------- DEFAULT CONFIG ----------------
# These can be overridden via command-line arguments

VIDEO_SOURCE = "sample-station.webm"
CAMERA_ID = "platform_cam_1"
MODEL_PATH = "yolo26n-pose.pt"
CONF_THRESH = 0.3
IOU_THRESH = 0.5
WINDOW_SECONDS = 4.0
OUTPUT_CSV = "features_dump.csv"

# PLATFORM_POLY will be auto-detected or manually calibrated!
PLATFORM_POLY = None

# Backend API endpoint (ready for future integration)
BACKEND_URL = "http://localhost:8000/api/analyze"
ENABLE_API = False  # Set to True when backend is ready

# ---------------- UTILITIES ----------------
# (All your existing utility functions stay the same)

def bbox_center(bbox):
    x1,y1,x2,y2 = bbox
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def point_to_polygon_distance(point, polygon):
    px, py = point
    min_d = float('inf')
    for i in range(len(polygon)):
        x1,y1 = polygon[i]
        x2,y2 = polygon[(i+1)%len(polygon)]
        dx = x2 - x1
        dy = y2 - y1
        if dx == dy == 0:
            d = np.hypot(px-x1, py-y1)
        else:
            t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy)/(dx*dx + dy*dy)))
            projx = x1 + t*dx
            projy = y1 + t*dy
            d = np.hypot(px-projx, py-projy)
        min_d = min(min_d, d)
    return min_d

def compute_torso_angle(kp):
    try:
        L_sh = kp[5][:2]
        R_sh = kp[6][:2]
        L_hip = kp[11][:2]
        R_hip = kp[12][:2]
        sh_mid = ((L_sh[0]+R_sh[0])/2, (L_sh[1]+R_sh[1])/2)
        hip_mid = ((L_hip[0]+R_hip[0])/2, (L_hip[1]+R_hip[1])/2)
        vec = np.array(hip_mid) - np.array(sh_mid)
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        return float(angle)
    except:
        return None

# ---------------- MAIN FUNCTION ----------------

def main():
    global PLATFORM_POLY

    # ============ PARSE ARGUMENTS ============
    parser = argparse.ArgumentParser(description='Suspicious Behavior Detection with Auto-Calibration')
    parser.add_argument('--video', type=str, default=VIDEO_SOURCE,
                       help='Video source (file or RTSP stream)')
    parser.add_argument('--camera', type=str, default=CAMERA_ID,
                       help='Camera ID (unique identifier)')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                       help='YOLO pose model path')
    parser.add_argument('--calibration', type=str, default='auto',
                       choices=['auto', 'manual', 'yolo', 'hough'],
                       help='Calibration method: auto (try all), manual (click points), yolo (segmentation), hough (edge detection)')
    parser.add_argument('--recalibrate', action='store_true',
                       help='Force platform recalibration (ignore saved config)')
    parser.add_argument('--visualize-calibration', action='store_true',
                       help='Show calibration detection process')
    parser.add_argument('--enable-api', action='store_true',
                       help='Enable sending alerts to backend API')
    parser.add_argument('--api-url', type=str, default=BACKEND_URL,
                       help='Backend API endpoint URL')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("SUSPICIOUS BEHAVIOR DETECTION SYSTEM")
    print("="*60)

    # ============ STEP 1: PLATFORM CALIBRATION ============
    print("\n[STEP 1] Platform Calibration")
    print("-" * 60)

    calibration_manager = CalibrationManager()

    PLATFORM_POLY = calibration_manager.calibrate(
        video_source=args.video,
        camera_id=args.camera,
        method=args.calibration,
        visualize=args.visualize_calibration,
        force_recalibrate=args.recalibrate
    )

    if PLATFORM_POLY is None:
        print("✗ Failed to calibrate platform boundary. Exiting.")
        return

    print(f"✓ Platform boundary: {len(PLATFORM_POLY)} vertices")
    print(f"  Vertices: {PLATFORM_POLY[:3]}..." if len(PLATFORM_POLY) > 3 else f"  Vertices: {PLATFORM_POLY}")
    # ======================================================

    # ============ STEP 2: INITIALIZE YOLO ============
    print("\n[STEP 2] Loading YOLO Model")
    print("-" * 60)

    try:
        model = YOLO(args.model)
        print(f"✓ Loaded model: {args.model}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    # =================================================

    # ============ STEP 3: INITIALIZE BRAIN LOGIC ============
    print("\n[STEP 3] Initializing Brain Logic")
    print("-" * 60)

    aggregator = FeatureAggregator(window_seconds=WINDOW_SECONDS)
    scorer = RuleBasedScorer()
    llm_analyzer = LLMAnalyzer()
    print("✓ Feature Aggregator initialized")
    print("✓ Rule-Based Scorer initialized")
    print("✓ LLM Analyzer initialized")
    # ========================================================

    # ============ STEP 4: SETUP CSV OUTPUT ============
    print("\n[STEP 4] Setting up output CSV")
    print("-" * 60)

    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp","camera_id","track_id",
                "x1","y1","x2","y2",
                "torso_angle","speed_px_s",
                "dist_to_edge_px","dwell_time_s"
            ])
        print(f"✓ Created CSV: {OUTPUT_CSV}")
    else:
        print(f"✓ Appending to CSV: {OUTPUT_CSV}")
    # ==================================================

    # ============ TRACKING STATE ============
    track_history = defaultdict(lambda: deque())
    last_positions = {}
    first_seen = {}

    # Alert tracking (to avoid spam)
    recent_alerts = {}
    ALERT_COOLDOWN = 10.0  # seconds between alerts for same person
    # ========================================

    # ============ STEP 5: START VIDEO PROCESSING ============
    print("\n[STEP 5] Starting video processing")
    print("-" * 60)
    print(f"Video source: {args.video}")
    print(f"Camera ID: {args.camera}")
    if args.enable_api:
        print(f"API enabled: {args.api_url}")
    print("Press 'q' to quit\n")

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"✗ Could not open video: {args.video}")
        return

    frame_count = 0
    start_time = time.time()
    # ========================================================

    # ============ MAIN LOOP ================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n✓ End of video or stream closed")
            break

        ts = time.time()
        frame_count += 1

        # ========== YOLO DETECTION ==========
        results = model.track(
            source=frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            persist=True,
            verbose=False
        )

        if len(results) == 0:
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        annotated_frame = r.plot()
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None
        keypoints = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
        # ====================================

        # Process each detected person
        if ids is None:
            cv2.imshow("Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for i, box in enumerate(boxes):
            track_id = int(ids[i])
            x1,y1,x2,y2 = map(int, box)

            center = bbox_center((x1,y1,x2,y2))

            # ========== FEATURE EXTRACTION ==========
            # Update temporal history
            track_history[track_id].append((ts, center))
            while (ts - track_history[track_id][0][0]) > WINDOW_SECONDS:
                track_history[track_id].popleft()

            if track_id not in first_seen:
                first_seen[track_id] = ts

            # Compute speed
            speed = None
            if track_id in last_positions:
                last_ts, last_center = last_positions[track_id]
                dt = ts - last_ts if ts - last_ts > 0 else 1e-6
                speed = float(np.hypot(
                    center[0]-last_center[0],
                    center[1]-last_center[1]
                ) / dt)

            last_positions[track_id] = (ts, center)

            # Compute dwell time
            dwell = ts - first_seen[track_id]

            # Distance to platform edge
            dist_edge = point_to_polygon_distance(center, PLATFORM_POLY)

            # Torso angle
            torso_angle = None
            if keypoints is not None and i < len(keypoints):
                torso_angle = compute_torso_angle(keypoints[i])
            # =========================================

            # ============ BRAIN LOGIC INTEGRATION ============

            # Package features for brain
            frame_features = {
                'bbox_center': center,
                'torso_angle': torso_angle if torso_angle is not None else 90.0,
                'speed': speed if speed is not None else 0.0,
                'dist_to_edge': dist_edge
            }

            # Add to aggregator
            aggregator.add_frame_features(track_id, ts, frame_features)

            # Get aggregated features (if window is full)
            agg_features = aggregator.get_aggregated_features(track_id)

            if agg_features is not None:
                # Calculate risk score
                risk_score, _ = scorer.compute_risk(agg_features)
                risk_level = scorer.get_risk_level(risk_score)

                # Display risk on video
                color = {
                    'normal': (0, 255, 0),      # Green
                    'low': (0, 255, 255),       # Yellow
                    'medium': (0, 165, 255),    # Orange
                    'high': (0, 100, 255),      # Dark orange
                    'critical': (0, 0, 255)     # Red
                }.get(risk_level, (255, 255, 255))

                cv2.putText(annotated_frame,
                           f"Risk: {risk_score} ({risk_level})",
                           (x1, y1-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # ============ THRESHOLD-BASED ALERT ROUTING ============
                # HIGH RISK (≥85): Skip LLM for immediate response
                # MEDIUM RISK (50-84): Use LLM for detailed analysis
                # LOW RISK (<50): No alert
                # =======================================================

                CRITICAL_THRESHOLD = 85  # Skip LLM if risk >= this
                MEDIUM_THRESHOLD = 50    # Minimum risk to generate alert

                if risk_score >= MEDIUM_THRESHOLD:
                    last_alert_time = recent_alerts.get(track_id, 0)

                    if (ts - last_alert_time) > ALERT_COOLDOWN:

                        # DECISION: Use LLM or fast path?
                        if risk_score >= CRITICAL_THRESHOLD:
                            # ⚡ FAST PATH: Critical risk - skip LLM for speed
                            print(f"\n⚡ CRITICAL ALERT (Fast Path): Track #{track_id} - Risk {risk_score}")

                            # Generate simple rule-based message
                            alert_parts = [f"CRITICAL: Person #{track_id}"]
                            if agg_features.get('min_dist_to_edge', 999) < 100:
                                alert_parts.append(f"very close to edge ({agg_features['min_dist_to_edge']:.0f}px)")
                            if agg_features.get('dwell_time_near_edge', 0) > 5:
                                alert_parts.append(f"dwelling {agg_features['dwell_time_near_edge']:.1f}s")

                            alert_message = " - ".join(alert_parts) + f" - Risk: {risk_score}/100"
                            recommended_action = "driver_alert" if risk_score >= 90 else "control_room"

                            llm_result = {
                                'risk_level': risk_level,
                                'confidence': 0.95,
                                'reasoning': 'FAST PATH: Rule-based detection (LLM skipped for critical urgency)',
                                'alert_message': alert_message,
                                'recommended_action': recommended_action,
                                'llm_used': False  # Important: marks this as fast-path
                            }

                            print(f"   {alert_message}")
                            print(f"   Action: {recommended_action} (LLM SKIPPED)\n")

                        else:
                            # 🧠 LLM PATH: Medium risk - use LLM for context
                            print(f"\n🧠 MEDIUM ALERT (LLM Path): Track #{track_id} - Risk {risk_score}")
                            print(f"   Calling LLM for detailed analysis...")

                            llm_result = llm_analyzer.analyze(
                                features=agg_features,
                                risk_score=risk_score,
                                track_id=track_id,
                                camera_id=args.camera
                            )

                            llm_result['llm_used'] = True  # Mark as LLM-analyzed

                            print(f"   {llm_result['alert_message']}")
                            print(f"   Action: {llm_result['recommended_action']} (LLM analyzed)\n")

                        # Send to backend API (if enabled)
                        if args.enable_api:
                            try:
                                import requests
                                requests.post(args.api_url, json={
                                    'track_id': track_id,
                                    'camera_id': args.camera,
                                    'timestamp': ts,
                                    'risk_score': risk_score,
                                    'risk_level': risk_level,
                                    'features': agg_features,
                                    'llm_analysis': llm_result
                                }, timeout=5)
                            except Exception as e:
                                print(f"⚠ Failed to send alert to API: {e}")

                        # Update cooldown
                        recent_alerts[track_id] = ts

            # =================================================

            # ========== CSV WRITING ==========
            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    ts, args.camera, track_id,
                    x1,y1,x2,y2,
                    torso_angle, speed,
                    dist_edge, dwell
                ])
            # =================================

            # ========== VISUALIZATION ==========
            cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(annotated_frame, f"ID:{track_id}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Show key metrics
            info_y = y1 + 20
            if speed is not None:
                cv2.putText(annotated_frame, f"Speed: {speed:.1f}px/s", (x1, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                info_y += 15

            cv2.putText(annotated_frame, f"Edge: {dist_edge:.0f}px", (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            info_y += 15

            cv2.putText(annotated_frame, f"Dwell: {dwell:.1f}s", (x1, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            # ===================================

        # Draw platform polygon
        cv2.polylines(annotated_frame, [np.array(PLATFORM_POLY)], True, (0,255,0), 2)

        # Add info overlay
        cv2.putText(annotated_frame, f"Frame: {frame_count} | Camera: {args.camera}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display frame
        cv2.imshow("Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n✓ User quit")
            break

    # ============ CLEANUP ============
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    print(f"Frames processed: {frame_count}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Average FPS: {fps:.1f}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Platform vertices: {len(PLATFORM_POLY)}")
    print("="*60 + "\n")
    # =================================


if __name__ == "__main__":
    main()
