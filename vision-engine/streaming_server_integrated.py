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

# Import brain modules (WITH LLM for threshold-based routing)
try:
    from brain.feature_aggregator import FeatureAggregator
    from brain.rule_based_scorer import RuleBasedScorer
    from brain.llm_analyzer import LLMAnalyzer  # NOW IMPORTED for threshold routing
    BRAIN_AVAILABLE = True
except ImportError:
    logger.warning("Brain modules not found - running without risk scoring")
    BRAIN_AVAILABLE = False

# Import face recognition, emotion detection, and advanced features
try:
    from face_recognition_service import FaceRecognitionService
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    logger.warning("Face recognition service not found - running without face tracking")
    FACE_RECOGNITION_AVAILABLE = False

try:
    from emotion_recognition_service import EmotionRecognitionService
    EMOTION_RECOGNITION_AVAILABLE = True
except ImportError:
    logger.warning("Emotion recognition service not found - running without emotion detection")
    EMOTION_RECOGNITION_AVAILABLE = False

try:
    from brain.advanced_features import AdvancedFeatureExtractor
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    logger.warning("Advanced features not found - running without advanced behavioral analysis")
    ADVANCED_FEATURES_AVAILABLE = False

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

# Threshold configuration for LLM routing
CRITICAL_THRESHOLD = 85  # Skip LLM if risk >= this (fast path)
MEDIUM_THRESHOLD = 20    # Minimum risk to generate alert (LOWERED FOR TESTING)


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
                                        aggregator, scorer, llm_analyzer, recent_alerts, ts,
                                        frame=None, face_service=None, emotion_service=None,
                                        advanced_features_extractor=None, location="Unknown"):
    """
    Extract RICH tracking data with brain logic + threshold-based LLM routing
    NOW INCLUDES: Face recognition, emotion detection, and advanced behavioral features

    Returns both objects array AND alert if high risk detected
    Alert includes LLM analysis for medium risk, or fast-path for critical risk
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

        # ============ FACE RECOGNITION INTEGRATION ============
        persistent_id = None
        face_embedding = None
        face_confidence = None

        if face_service and frame is not None:
            try:
                bbox = [x1, y1, x2, y2]
                face_embedding = face_service.extract_face_embedding(frame, bbox)

                if face_embedding is not None:
                    # Get or create persistent ID
                    persistent_id = face_service.get_or_create_track_id(
                        face_embedding,
                        camera_id,
                        location
                    )

                    # Map YOLO track ID to persistent ID
                    face_service.map_yolo_to_persistent_id(
                        camera_id,
                        track_id,
                        face_embedding,
                        location
                    )

                    # Get face match confidence
                    face_confidence = face_service.get_match_confidence(persistent_id, face_embedding)

                    logger.debug(f"Face detected: Track {track_id} -> {persistent_id} (conf: {face_confidence:.2f})")
            except Exception as e:
                logger.debug(f"Face recognition failed for track {track_id}: {e}")

        # ============ EMOTION DETECTION INTEGRATION ============
        emotion = None
        emotion_confidence = None
        emotion_risk_assessment = None

        if emotion_service and frame is not None:
            try:
                bbox = [x1, y1, x2, y2]
                emotion, emotion_confidence = emotion_service.detect_emotion(frame, bbox)

                if emotion and persistent_id:
                    # Track emotion history
                    emotion_service.track_emotion(persistent_id, emotion, emotion_confidence)

                    # Assess emotional risk with context
                    emotion_context = {
                        'near_edge': dist_edge < 100,
                        'loitering': dwell > 10,
                        'aggressive_movement': speed > 50
                    }
                    emotion_risk_assessment = emotion_service.assess_emotional_risk(
                        emotion,
                        emotion_confidence,
                        emotion_context
                    )

                    logger.debug(f"Emotion detected: {persistent_id} - {emotion} ({emotion_confidence:.2f})")
            except Exception as e:
                logger.debug(f"Emotion detection failed for track {track_id}: {e}")

        # ============ ADVANCED FEATURES INTEGRATION ============
        advanced_features = {}

        if advanced_features_extractor and keypoints is not None and i < len(keypoints):
            try:
                kp = keypoints[i]

                # Extract advanced behavioral indicators
                advanced_features['shoulder_hunch'] = advanced_features_extractor.compute_shoulder_hunch(kp)
                advanced_features['closed_body_posture'] = advanced_features_extractor.detect_closed_body_posture(kp)
                advanced_features['hand_to_face'] = advanced_features_extractor.compute_hand_to_face_frequency(track_id, kp, ts)
                advanced_features['head_down'] = advanced_features_extractor.compute_head_down_ratio(track_id, kp, ts)
                advanced_features['edge_transgression_count'] = advanced_features_extractor.compute_edge_transgression_count(
                    track_id, center, platform_poly, ts
                ) if platform_poly else 0
                advanced_features['head_yaw'] = advanced_features_extractor.compute_head_yaw(kp)
                advanced_features['social_isolation'] = advanced_features_extractor.compute_social_isolation_metric(
                    track_id, center, [bbox_center(obj['bbox']) for obj in objects]
                ) if objects else 0

                logger.debug(f"Advanced features: Track {track_id} - {advanced_features}")
            except Exception as e:
                logger.debug(f"Advanced feature extraction failed for track {track_id}: {e}")

        # Default risk score
        risk_score = 0
        risk_level = "normal"

        # ============ BRAIN LOGIC INTEGRATION ============
        if BRAIN_AVAILABLE and aggregator:
            # Package features (including advanced features)
            frame_features = {
                'bbox_center': center,
                'torso_angle': torso_angle if torso_angle is not None else 90.0,
                'speed': speed,
                'dist_to_edge': dist_edge
            }

            # Add advanced features to frame_features
            if advanced_features:
                frame_features.update(advanced_features)

            # Add to aggregator
            aggregator.add_frame_features(track_id, ts, frame_features)

            # Get aggregated features
            agg_features = aggregator.get_aggregated_features(track_id)

            if agg_features is not None and scorer:
                # Calculate base risk score
                risk_score, _ = scorer.compute_risk(agg_features)

                # ============ ENHANCE RISK WITH EMOTION ============
                if emotion_risk_assessment:
                    emotion_risk_score = emotion_risk_assessment.get('risk_score', 0)
                    # Add emotion risk (weighted 30%)
                    risk_score = int(risk_score * 0.7 + emotion_risk_score * 0.3)
                    logger.debug(f"Risk enhanced with emotion: base={risk_score*0.7:.0f} + emotion={emotion_risk_score*0.3:.0f} = {risk_score}")

                # ============ ENHANCE RISK WITH ADVANCED FEATURES ============
                if advanced_features:
                    # Add risk from advanced indicators
                    risk_boost = 0
                    shoulder_hunch = advanced_features.get('shoulder_hunch')
                    if shoulder_hunch is not None and shoulder_hunch > 0.7:
                        risk_boost += 10
                    if advanced_features.get('closed_body_posture', False):
                        risk_boost += 15
                    hand_to_face = advanced_features.get('hand_to_face')
                    if hand_to_face is not None and hand_to_face > 0.5:
                        risk_boost += 8
                    head_down = advanced_features.get('head_down')
                    if head_down is not None and head_down > 0.6:
                        risk_boost += 7
                    edge_count = advanced_features.get('edge_transgression_count')
                    if edge_count is not None and edge_count >= 3:
                        risk_boost += 20
                    social_isolation = advanced_features.get('social_isolation')
                    if social_isolation is not None and social_isolation > 0.8:
                        risk_boost += 12

                    risk_score = min(100, risk_score + risk_boost)
                    if risk_boost > 0:
                        logger.debug(f"Risk boosted by {risk_boost} from advanced features: total={risk_score}")

                risk_level = scorer.get_risk_level(risk_score)

                # ============ THRESHOLD-BASED ALERT ROUTING ============
                # Generate alert if risk >= MEDIUM_THRESHOLD
                if risk_score >= MEDIUM_THRESHOLD:
                    last_alert_time = recent_alerts.get(track_id, 0)

                    if (ts - last_alert_time) > ALERT_COOLDOWN:

                        # DECISION: Use LLM or fast path?
                        if risk_score >= CRITICAL_THRESHOLD:
                            # ⚡ FAST PATH: Critical risk - skip LLM for speed
                            logger.warning(f"⚡ CRITICAL ALERT (Fast Path): Track {track_id} - Risk {risk_score}")

                            # Generate simple rule-based message
                            alert_parts = [f"CRITICAL: Person #{track_id}"]
                            if agg_features.get('min_dist_to_edge', 999) < 100:
                                alert_parts.append(f"very close to edge ({agg_features['min_dist_to_edge']:.0f}px)")
                            if agg_features.get('dwell_time_near_edge', 0) > 5:
                                alert_parts.append(f"dwelling {agg_features['dwell_time_near_edge']:.1f}s")

                            alert_message = " - ".join(alert_parts) + f" - Risk: {risk_score}/100"
                            recommended_action = "driver_alert" if risk_score >= 90 else "control_room"

                            alert = {
                                "track_id": track_id,
                                "camera_id": camera_id,
                                "risk_score": risk_score,
                                "risk_level": risk_level,
                                "timestamp": ts,
                                "features": agg_features,
                                "llm_analysis": {
                                    "risk_level": risk_level,
                                    "confidence": 0.95,
                                    "reasoning": "FAST PATH: Rule-based detection (LLM skipped for critical urgency)",
                                    "alert_message": alert_message,
                                    "recommended_action": recommended_action,
                                    "llm_used": False
                                }
                            }

                        else:
                            # 🧠 LLM PATH: Medium risk - use LLM for context
                            logger.warning(f"🧠 MEDIUM ALERT (LLM Path): Track {track_id} - Risk {risk_score}")

                            if llm_analyzer:
                                try:
                                    logger.info(f"   Calling LLM for detailed analysis...")
                                    llm_result = llm_analyzer.analyze(
                                        features=agg_features,
                                        risk_score=risk_score,
                                        track_id=track_id,
                                        camera_id=camera_id
                                    )
                                    llm_result['llm_used'] = True

                                    alert = {
                                        "track_id": track_id,
                                        "camera_id": camera_id,
                                        "risk_score": risk_score,
                                        "risk_level": risk_level,
                                        "timestamp": ts,
                                        "features": agg_features,
                                        "llm_analysis": llm_result
                                    }
                                    logger.info(f"   ✓ LLM analysis complete: {llm_result['alert_message'][:50]}...")

                                except Exception as e:
                                    # Fallback to fast path if LLM fails
                                    logger.error(f"   ✗ LLM failed: {e}, using fast path fallback")
                                    alert_message = f"Person #{track_id} detected with risk {risk_score}/100"
                                    alert = {
                                        "track_id": track_id,
                                        "camera_id": camera_id,
                                        "risk_score": risk_score,
                                        "risk_level": risk_level,
                                        "timestamp": ts,
                                        "features": agg_features,
                                        "llm_analysis": {
                                            "risk_level": risk_level,
                                            "confidence": 0.7,
                                            "reasoning": "LLM unavailable - fallback alert",
                                            "alert_message": alert_message,
                                            "recommended_action": "monitor",
                                            "llm_used": False
                                        }
                                    }
                            else:
                                # LLM not available - use fast path
                                logger.warning(f"   ⚠ LLM not available, using fast path")
                                alert_message = f"Person #{track_id} detected with risk {risk_score}/100"
                                alert = {
                                    "track_id": track_id,
                                    "camera_id": camera_id,
                                    "risk_score": risk_score,
                                    "risk_level": risk_level,
                                    "timestamp": ts,
                                    "features": agg_features,
                                    "llm_analysis": {
                                        "risk_level": risk_level,
                                        "confidence": 0.7,
                                        "reasoning": "LLM not initialized",
                                        "alert_message": alert_message,
                                        "recommended_action": "monitor",
                                        "llm_used": False
                                    }
                                }

                        recent_alerts[track_id] = ts

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
            "risk_level": risk_level,
            # FACE RECOGNITION DATA
            "persistent_id": persistent_id,
            "face_confidence": face_confidence,
            # EMOTION DATA
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
            "emotion_risk": emotion_risk_assessment.get('risk_score') if emotion_risk_assessment else None,
            # ADVANCED FEATURES
            "advanced_features": advanced_features if advanced_features else None
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

        # Send person data for persistent tracking (face recognition + emotion)
        for obj in tracking_data.get('objects', []):
            if obj.get('persistent_id'):
                person_data = {
                    "persistent_id": obj['persistent_id'],
                    "camera_id": camera_id,
                    "location": f"Platform {camera_id}",
                    "yolo_track_id": obj['track_id'],
                    "bbox": obj['bbox'],
                    "emotion": obj.get('emotion'),
                    "emotion_confidence": obj.get('emotion_confidence'),
                    "risk_score": obj.get('risk_score', 0),
                    "timestamp": tracking_data['timestamp']
                }

                try:
                    person_response = http_client.post(
                        f"{NODE_BACKEND_URL}/api/persons/update",
                        json=person_data
                    )
                    if person_response.status_code not in [200, 201]:
                        logger.debug(f"Person update failed: {person_response.status_code}")
                except Exception as e:
                    logger.debug(f"Error updating person {obj['persistent_id']}: {e}")

        # Send alert if present (using NEW smart routing endpoint)
        if tracking_data.get('alert'):
            alert_response = http_client.post(
                f"{NODE_BACKEND_URL}/api/alerts/from-detection",
                json=tracking_data['alert']
            )
            if alert_response.status_code == 201:
                result = alert_response.json()
                logger.info(f"✅ Alert created via {result.get('path', 'unknown')} path")
            else:
                logger.warning(f"⚠️ Alert creation failed: {alert_response.status_code}")
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

        await asyncio.sleep(0.05)  # 20 FPS instead of 50 FPS (reduce lag)


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
        "face_recognition_enabled": FACE_RECOGNITION_AVAILABLE,
        "emotion_recognition_enabled": EMOTION_RECOGNITION_AVAILABLE,
        "advanced_features_enabled": ADVANCED_FEATURES_AVAILABLE,
        "frame_count": {cam: 1 if get_latest_frame(cam) is not None else 0
                       for cam in latest_frames.keys()}
    }


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "MetroEye Vision Engine (Integrated)",
        "version": "3.0.0",
        "features": {
            "brain_logic": BRAIN_AVAILABLE,
            "face_recognition": FACE_RECOGNITION_AVAILABLE,
            "emotion_detection": EMOTION_RECOGNITION_AVAILABLE,
            "advanced_behavioral_analysis": ADVANCED_FEATURES_AVAILABLE
        },
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
    llm_analyzer = None
    if BRAIN_AVAILABLE:
        aggregator = FeatureAggregator(window_seconds=WINDOW_SECONDS)
        scorer = RuleBasedScorer()
        try:
            llm_analyzer = LLMAnalyzer()
            logger.info("✓ Brain modules initialized (with LLM)")
        except Exception as e:
            logger.warning(f"⚠ LLM analyzer failed to initialize: {e}")
            logger.info("✓ Brain modules initialized (without LLM)")

    # Initialize face recognition service
    face_service = None
    if FACE_RECOGNITION_AVAILABLE:
        try:
            face_service = FaceRecognitionService(backend_url=NODE_BACKEND_URL)
            logger.info("✓ Face recognition service initialized")
        except Exception as e:
            logger.warning(f"⚠ Face recognition failed to initialize: {e}")

    # Initialize emotion recognition service
    emotion_service = None
    if EMOTION_RECOGNITION_AVAILABLE:
        try:
            emotion_service = EmotionRecognitionService()
            logger.info("✓ Emotion recognition service initialized")
        except Exception as e:
            logger.warning(f"⚠ Emotion recognition failed to initialize: {e}")

    # Initialize advanced features extractor
    advanced_features_extractor = None
    if ADVANCED_FEATURES_AVAILABLE:
        try:
            advanced_features_extractor = AdvancedFeatureExtractor()
            logger.info("✓ Advanced features extractor initialized")
        except Exception as e:
            logger.warning(f"⚠ Advanced features failed to initialize: {e}")

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

        # Draw real-time features on frame (for every frame, not just every 10)
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for i, box in enumerate(boxes):
                track_id = int(ids[i])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center = bbox_center((x1, y1, x2, y2))

                # Calculate current features for display
                dist_edge = point_to_polygon_distance(center, platform_poly) if platform_poly else 999.0
                speed = 0.0
                if track_id in last_positions:
                    last_ts, last_center = last_positions[track_id]
                    dt = ts - last_ts if ts - last_ts > 0 else 1e-6
                    speed = float(np.hypot(center[0] - last_center[0], center[1] - last_center[1]) / dt)

                # Get aggregated risk if available
                risk_score = 0
                if BRAIN_AVAILABLE and aggregator:
                    agg_features = aggregator.get_aggregated_features(track_id)
                    if agg_features is not None and scorer:
                        risk_score, _ = scorer.compute_risk(agg_features)

                # Draw feature text overlay
                y_offset = int(y1) - 10

                # Risk score (color-coded)
                risk_color = (0, 255, 0) if risk_score < 30 else (0, 165, 255) if risk_score < 70 else (0, 0, 255)
                cv2.putText(annotated_frame, f"Risk: {risk_score}",
                           (int(x1), y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, risk_color, 2)
                y_offset -= 25

                # Edge distance
                if dist_edge < 999:
                    edge_color = (0, 0, 255) if dist_edge < 50 else (0, 165, 255) if dist_edge < 100 else (0, 255, 0)
                    cv2.putText(annotated_frame, f"Edge: {dist_edge:.0f}px",
                               (int(x1), y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, edge_color, 2)
                    y_offset -= 20

                # Speed
                if speed > 10:
                    cv2.putText(annotated_frame, f"Spd: {speed:.0f}px/s",
                               (int(x1), y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (255, 255, 255), 2)

        # Update frame for streaming
        update_frame(camera_id, annotated_frame)

        # Extract and send RICH tracking data every 3 frames (faster updates for frontend)
        frame_counter += 1
        if frame_counter % 3 == 0:
            tracking_data = extract_tracking_data_with_features(
                results[0], camera_id, track_history, last_positions,
                first_seen, platform_poly, aggregator, scorer, llm_analyzer, recent_alerts, ts,
                frame=frame,  # Pass original frame for face/emotion detection
                face_service=face_service,
                emotion_service=emotion_service,
                advanced_features_extractor=advanced_features_extractor,
                location=f"Platform {camera_id}"  # Location for person tracking
            )

            # Send to Node.js backend (non-blocking)
            if tracking_data['objects']:
                send_tracking_data(camera_id, tracking_data)
            else:
                logger.debug(f"{camera_id}: No objects to send (frame {frame_counter})")

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
        # ("camera_3", "sample-station1.mp4"),
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
    logger.info("=" * 50)
    logger.info("🎯 Feature Status:")
    logger.info(f"  🧠 Brain Logic: {'✅ ENABLED' if BRAIN_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  👤 Face Recognition: {'✅ ENABLED' if FACE_RECOGNITION_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  😊 Emotion Detection: {'✅ ENABLED' if EMOTION_RECOGNITION_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  🎭 Advanced Behavioral Analysis: {'✅ ENABLED' if ADVANCED_FEATURES_AVAILABLE else '❌ DISABLED'}")
    logger.info("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=5000)
