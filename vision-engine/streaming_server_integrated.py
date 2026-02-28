"""
FastAPI Streaming Server with FULL BRAIN LOGIC Integration
===========================================================

Combines streaming_server.py + app_integrated_preview.py

Features:
- MJPEG streaming
- YOLO pose detection + tracking
- Feature extraction (speed, edge distance, torso angle, dwell time)
- Brain logic (aggregation + risk scoring + LLM analysis)
- Calibration integration
- Sends RICH data to Node.js (with risk scores!)

Usage:
    python streaming_server_integrated.py
"""

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import asyncio
import numpy as np
from typing import Dict
import threading
import time
import logging
import httpx
from collections import defaultdict, deque

# Import calibration
from calibration_endpoints import router as calibration_router
from edge_calibrator import CalibrationManager

# Import brain modules
try:
    from brain.feature_aggregator import FeatureAggregator
    from brain.rule_based_scorer import RuleBasedScorer
    from brain.llm_analyzer import LLMAnalyzer
    BRAIN_AVAILABLE = True
except ImportError:
    logger.warning("Brain modules not found - running without risk scoring")
    BRAIN_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MetroEye Vision Engine API (Integrated)",
    description="Video streaming with intelligent behavior detection",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include calibration router
app.include_router(calibration_router)

# Store latest frames for each camera (thread-safe)
latest_frames: Dict[str, np.ndarray] = {}
frame_locks: Dict[str, threading.Lock] = {}

# Node.js backend configuration
NODE_BACKEND_URL = "http://localhost:8000"
http_client = httpx.Client(timeout=1.0)

# Configuration
WINDOW_SECONDS = 4.0
ALERT_COOLDOWN = 10.0


def update_frame(camera_id: str, frame: np.ndarray):
    """Update the latest frame for a camera (thread-safe)"""
    if camera_id not in frame_locks:
        frame_locks[camera_id] = threading.Lock()

    with frame_locks[camera_id]:
        latest_frames[camera_id] = frame.copy()


def get_latest_frame(camera_id: str) -> np.ndarray:
    """Get the latest frame for a camera (thread-safe)"""
    if camera_id not in frame_locks:
        return None

    with frame_locks[camera_id]:
        return latest_frames.get(camera_id, None)


# ============ UTILITY FUNCTIONS (from app_integrated_preview.py) ============

def bbox_center(bbox):
    """Calculate center of bounding box"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_to_polygon_distance(point, polygon):
    """Calculate distance from point to polygon edge"""
    px, py = point
    min_d = float('inf')
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        dx = x2 - x1
        dy = y2 - y1
        if dx == dy == 0:
            d = np.hypot(px - x1, py - y1)
        else:
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
            projx = x1 + t * dx
            projy = y1 + t * dy
            d = np.hypot(px - projx, py - projy)
        min_d = min(min_d, d)
    return min_d


def compute_torso_angle(kp):
    """Compute torso angle from pose keypoints"""
    try:
        L_sh = kp[5][:2]
        R_sh = kp[6][:2]
        L_hip = kp[11][:2]
        R_hip = kp[12][:2]
        sh_mid = ((L_sh[0] + R_sh[0]) / 2, (L_sh[1] + R_sh[1]) / 2)
        hip_mid = ((L_hip[0] + R_hip[0]) / 2, (L_hip[1] + R_hip[1]) / 2)
        vec = np.array(hip_mid) - np.array(sh_mid)
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        return float(angle)
    except:
        return None


def extract_tracking_data_with_features(result, camera_id: str, track_history,
                                        last_positions, first_seen, platform_poly,
                                        aggregator, scorer, recent_alerts, ts):
    """
    Extract RICH tracking data with brain logic

    Returns both objects array AND alert if high risk detected
    """
    objects = []
    alert = None

    if result.boxes is None:
        return {"camera_id": camera_id, "timestamp": ts, "objects": [], "alert": None}

    boxes = result.boxes.cpu().numpy()
    ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
    keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None

    if ids is None:
        return {"camera_id": camera_id, "timestamp": ts, "objects": [], "alert": None}

    for i, box in enumerate(boxes):
        track_id = int(ids[i])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])

        center = bbox_center((x1, y1, x2, y2))

        # Update temporal history
        track_history[track_id].append((ts, center))
        while (ts - track_history[track_id][0][0]) > WINDOW_SECONDS:
            track_history[track_id].popleft()

        if track_id not in first_seen:
            first_seen[track_id] = ts

        # Compute speed
        speed = 0.0
        if track_id in last_positions:
            last_ts, last_center = last_positions[track_id]
            dt = ts - last_ts if ts - last_ts > 0 else 1e-6
            speed = float(np.hypot(
                center[0] - last_center[0],
                center[1] - last_center[1]
            ) / dt)

        last_positions[track_id] = (ts, center)

        # Compute dwell time
        dwell = ts - first_seen[track_id]

        # Distance to platform edge
        dist_edge = point_to_polygon_distance(center, platform_poly) if platform_poly else 999.0

        # Torso angle
        torso_angle = None
        if keypoints is not None and i < len(keypoints):
            torso_angle = compute_torso_angle(keypoints[i])

        # Default risk score
        risk_score = 0
        risk_level = "normal"

        # ============ BRAIN LOGIC INTEGRATION ============
        if BRAIN_AVAILABLE and aggregator:
            # Package features
            frame_features = {
                'bbox_center': center,
                'torso_angle': torso_angle if torso_angle is not None else 90.0,
                'speed': speed,
                'dist_to_edge': dist_edge
            }

            # Add to aggregator
            aggregator.add_frame_features(track_id, ts, frame_features)

            # Get aggregated features
            agg_features = aggregator.get_aggregated_features(track_id)

            if agg_features is not None and scorer:
                # Calculate risk score
                risk_score, _ = scorer.compute_risk(agg_features)
                risk_level = scorer.get_risk_level(risk_score)

                # Generate alert if risky and not in cooldown
                if risk_level in ['medium', 'high', 'critical']:
                    last_alert_time = recent_alerts.get(track_id, 0)

                    if (ts - last_alert_time) > ALERT_COOLDOWN:
                        alert = {
                            "track_id": track_id,
                            "camera_id": camera_id,
                            "risk_score": risk_score,
                            "risk_level": risk_level,
                            "timestamp": ts,
                            "features": agg_features
                        }
                        recent_alerts[track_id] = ts
                        logger.warning(f"🚨 HIGH RISK DETECTED: Track {track_id} - {risk_level} ({risk_score})")

        # Package object data
        obj = {
            "track_id": track_id,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": confidence,
            "class": "person",
            # RICH FEATURES
            "speed": speed,
            "dist_to_edge": dist_edge,
            "dwell_time": dwell,
            "torso_angle": torso_angle,
            "risk_score": risk_score,
            "risk_level": risk_level
        }

        if keypoints is not None and i < len(keypoints):
            obj["keypoints"] = keypoints[i].tolist()

        objects.append(obj)

    return {
        "camera_id": camera_id,
        "timestamp": ts,
        "frame_count": len(objects),
        "objects": objects,
        "alert": alert
    }


def send_tracking_data(camera_id: str, tracking_data: dict):
    """Send tracking data to Node.js backend (non-blocking)"""
    try:
        response = http_client.post(
            f"{NODE_BACKEND_URL}/api/tracking/{camera_id}",
            json=tracking_data
        )
        if response.status_code != 200:
            logger.warning(f"Failed to send tracking data: {response.status_code}")

        # Send alert if present
        if tracking_data.get('alert'):
            http_client.post(
                f"{NODE_BACKEND_URL}/api/alerts",
                json=tracking_data['alert']
            )
    except Exception as e:
        logger.debug(f"Error sending tracking data: {e}")


async def frame_generator(camera_id: str):
    """Generator that yields frames as MJPEG stream"""
    logger.info(f"Starting stream for camera: {camera_id}")

    while True:
        frame = get_latest_frame(camera_id)

        if frame is None:
            await asyncio.sleep(0.033)
            continue

        # Encode frame as JPEG with optimized settings
        _, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, 50,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        await asyncio.sleep(0.02)


@app.get("/stream/{camera_id}")
async def stream_camera(camera_id: str):
    """Stream endpoint for a specific camera"""
    return StreamingResponse(
        frame_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/frame/{camera_id}")
async def get_frame(camera_id: str):
    """Get single latest frame"""
    frame = get_latest_frame(camera_id)

    if frame is None:
        return Response(content=b'', status_code=404, media_type="text/plain")

    _, buffer = cv2.imencode('.jpg', frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "active_cameras": list(latest_frames.keys()),
        "timestamp": time.time(),
        "brain_enabled": BRAIN_AVAILABLE,
        "frame_count": {cam: 1 if get_latest_frame(cam) is not None else 0
                       for cam in latest_frames.keys()}
    }


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "MetroEye Vision Engine (Integrated)",
        "version": "2.0.0",
        "brain_logic": BRAIN_AVAILABLE,
        "endpoints": {
            "streaming": "/stream/{camera_id}",
            "frame": "/frame/{camera_id}",
            "calibration_auto": "/calibrate/auto/{camera_id}",
            "health": "/health"
        },
        "docs": "/docs"
    }


# ============ INTEGRATED DETECTION LOOP ============

def run_detection_integrated(camera_id: str, video_source: str):
    """
    Main detection loop with FULL BRAIN LOGIC

    Combines YOLO detection + feature extraction + risk scoring
    """
    from ultralytics import YOLO

    logger.info(f"Starting integrated detection for {camera_id}: {video_source}")

    # Load YOLO model
    model = YOLO('yolo26n-pose.pt')

    # Initialize brain modules
    aggregator = None
    scorer = None
    if BRAIN_AVAILABLE:
        aggregator = FeatureAggregator(window_seconds=WINDOW_SECONDS)
        scorer = RuleBasedScorer()
        logger.info("✓ Brain modules initialized")

    # Load calibration
    calibration_manager = CalibrationManager()
    platform_poly = calibration_manager.load_calibration(camera_id)

    if platform_poly is None:
        logger.warning(f"No calibration found for {camera_id} - auto-calibrating...")
        platform_poly = calibration_manager.calibrate(video_source, camera_id, method='auto')

    if platform_poly:
        logger.info(f"✓ Platform calibrated: {len(platform_poly)} vertices")
    else:
        logger.warning("Running without platform calibration")

    # Tracking state
    track_history = defaultdict(lambda: deque())
    last_positions = {}
    first_seen = {}
    recent_alerts = {}

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        logger.error(f"Could not open video source: {video_source}")
        return

    frame_counter = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            logger.warning(f"End of video for {camera_id}, looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        ts = time.time()

        # Run YOLO detection
        results = model.track(frame, persist=True, conf=0.3)

        # Draw annotations
        annotated_frame = results[0].plot()

        # Draw platform polygon if available
        if platform_poly:
            cv2.polylines(annotated_frame, [np.array(platform_poly)], True, (0, 255, 0), 2)

        # Update frame for streaming
        update_frame(camera_id, annotated_frame)

        # Extract and send RICH tracking data every 10 frames
        frame_counter += 1
        if frame_counter % 10 == 0:
            tracking_data = extract_tracking_data_with_features(
                results[0], camera_id, track_history, last_positions,
                first_seen, platform_poly, aggregator, scorer, recent_alerts, ts
            )

            # Send to Node.js backend (non-blocking)
            if tracking_data['objects']:
                send_tracking_data(camera_id, tracking_data)

        time.sleep(0.001)

    cap.release()


def start_detection_thread(camera_id: str, video_source: str):
    """Start detection in background thread"""
    detection_thread = threading.Thread(
        target=run_detection_integrated,
        args=(camera_id, video_source),
        daemon=True
    )
    detection_thread.start()
    logger.info(f"Detection thread started for {camera_id}")


if __name__ == "__main__":
    import uvicorn

    # Start detection for multiple cameras
    cameras = [
        ("camera_1", "sample-station.webm"),
        ("camera_2", "sample2.0.mp4"),
        # ("camera_3", "rtsp://your-camera-url"),
    ]

    logger.info("=" * 50)
    logger.info("Starting INTEGRATED detection threads...")
    for camera_id, video_source in cameras:
        start_detection_thread(camera_id, video_source)
        logger.info(f"  ✅ {camera_id}: {video_source}")
    logger.info("=" * 50)

    # Start FastAPI server
    logger.info("MetroEye Vision Engine API (Integrated)")
    logger.info("=" * 50)
    logger.info("📹 Video streams:")
    for camera_id, _ in cameras:
        logger.info(f"    http://localhost:5000/stream/{camera_id}")
    logger.info(f"📊 API docs: http://localhost:5000/docs")
    logger.info(f"🏥 Health check: http://localhost:5000/health")
    logger.info(f"🔗 Node.js backend: {NODE_BACKEND_URL}")
    logger.info(f"🧠 Brain logic: {'ENABLED' if BRAIN_AVAILABLE else 'DISABLED'}")
    logger.info("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=5000)
