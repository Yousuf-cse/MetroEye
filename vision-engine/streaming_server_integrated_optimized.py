"""
FastAPI Streaming Server with OPTIMIZED PERFORMANCE
====================================================

PERFORMANCE OPTIMIZATIONS:
- Async LLM calls (non-blocking)
- Per-camera face/emotion processing intervals (configurable 20-30 frames)
- Reduced processing frequency
- Smart frame dropping
- Performance monitoring
- Configurable feature toggles per camera

Usage:
    python streaming_server_integrated_optimized.py
"""

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import asyncio
import numpy as np
from typing import Dict, Optional
import threading
import time
import logging
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import queue
import multiprocessing as mp
import signal
import atexit
import sys

# Configure multiprocessing for Windows (use spawn method)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set
    pass

# Import calibration
from calibration_endpoints import router as calibration_router
from edge_calibrator import CalibrationManager

# Import new multiprocessing modules
from shared_frame_buffer import SharedFrameBuffer, create_frame_buffer
from camera_process import start_camera_process

# Import brain modules (WITH LLM for threshold-based routing)
try:
    from brain.feature_aggregator import FeatureAggregator
    from brain.rule_based_scorer import RuleBasedScorer
    from brain.llm_analyzer import LLMAnalyzer
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False

# Import face recognition, emotion detection, and advanced features
try:
    from face_recognition_service import FaceRecognitionService
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    from emotion_recognition_service import EmotionRecognitionService
    EMOTION_RECOGNITION_AVAILABLE = True
except ImportError:
    EMOTION_RECOGNITION_AVAILABLE = False

try:
    from brain.advanced_features import AdvancedFeatureExtractor
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import PA announcement system (after logger setup)
try:
    from pa_announcement_system import PAAnnouncement
    PA_SYSTEM_AVAILABLE = True
    logger.info("✓ PA announcement system available")
except ImportError:
    PA_SYSTEM_AVAILABLE = False
    logger.warning("⚠ PA announcement system not available")

# ============ EDGE PROXIMITY TRACKER ============

class EdgeProximityTracker:
    """
    Tracks how long each person has been near the platform edge
    Triggers PA announcements when dwell time exceeds threshold
    """

    def __init__(self, distance_threshold: float = 100.0, dwell_threshold: float = 5.0, announcement_cooldown: float = 15.0):
        """
        Args:
            distance_threshold: Max distance from edge to be considered "near" (pixels)
            dwell_threshold: Minimum time near edge before alert (seconds)
            announcement_cooldown: Min seconds between announcements for same person
        """
        self.distance_threshold = distance_threshold
        self.dwell_threshold = dwell_threshold
        self.announcement_cooldown = announcement_cooldown

        # Track when each person first got near the edge
        self.near_edge_since = {}  # track_id -> timestamp when they got near edge

        # Track when we last triggered an announcement for each person
        self.last_announcement_time = {}  # track_id -> timestamp of last announcement

    def update(self, track_id: int, dist_to_edge: float, timestamp: float) -> dict:
        """
        Update proximity tracking for a person

        Args:
            track_id: Person track ID
            dist_to_edge: Current distance from edge (pixels)
            timestamp: Current timestamp

        Returns:
            dict: Alert info if announcement should be triggered, None otherwise
        """
        # Check if person is near the edge
        is_near_edge = dist_to_edge < self.distance_threshold

        if is_near_edge:
            # Person is near edge
            if track_id not in self.near_edge_since:
                # First time near edge, start tracking
                self.near_edge_since[track_id] = timestamp
                logger.info(f"⚠️ Track {track_id} entered danger zone ({dist_to_edge:.0f}px from edge)")

            # Calculate how long they've been near edge
            time_near_edge = timestamp - self.near_edge_since[track_id]

            # Check if we should trigger an announcement
            if time_near_edge >= self.dwell_threshold:
                # Check cooldown
                last_announcement = self.last_announcement_time.get(track_id, 0)
                if (timestamp - last_announcement) >= self.announcement_cooldown:
                    # Trigger announcement!
                    self.last_announcement_time[track_id] = timestamp

                    return {
                        'should_announce': True,
                        'time_near_edge': time_near_edge,
                        'distance': dist_to_edge,
                        'reason': f'Person has been near edge for {time_near_edge:.1f}s'
                    }
        else:
            # Person moved away from edge, reset tracking
            if track_id in self.near_edge_since:
                time_near_edge = timestamp - self.near_edge_since[track_id]
                logger.info(f"✓ Track {track_id} left danger zone (was there for {time_near_edge:.1f}s)")
                del self.near_edge_since[track_id]

        return {'should_announce': False}

    def cleanup_old_tracks(self, active_tracks: set, timestamp: float):
        """Remove tracking for tracks that no longer exist"""
        to_remove = []
        for track_id in list(self.near_edge_since.keys()):
            if track_id not in active_tracks:
                to_remove.append(track_id)
        for track_id in to_remove:
            del self.near_edge_since[track_id]
            if track_id in self.last_announcement_time:
                del self.last_announcement_time[track_id]

# Create FastAPI app
app = FastAPI(
    title="MetroEye Vision Engine API (Optimized)",
    description="High-performance video streaming with intelligent behavior detection",
    version="3.0.0-optimized"
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

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# LLM task queue (non-blocking)
llm_queue = queue.Queue(maxsize=10)

# Configuration
WINDOW_SECONDS = 4.0
ALERT_COOLDOWN = 10.0

# Threshold configuration for LLM routing
CRITICAL_THRESHOLD = 85
MEDIUM_THRESHOLD = 50  # Medium risk triggers LLM analysis (changed from 20 - too low!)

# ============ PER-CAMERA CONFIGURATION ============
class CameraConfig:
    """Configuration for each camera with performance tuning"""
    def __init__(
        self,
        camera_id: str,
        face_recognition_enabled: bool = True,  # Enable/disable face recognition for this camera
        emotion_detection_enabled: bool = True,  # Enable/disable emotion detection for this camera
        face_detection_interval: int = 25,  # Process face every N frames (20-30)
        emotion_detection_interval: int = 25,  # Process emotion every N frames (20-30)
        advanced_features_enabled: bool = True,
        tracking_data_interval: int = 10,  # Send tracking data every N frames (reduced from 3)
        stream_fps: int = 20,  # Target streaming FPS
        jpeg_quality: int = 50  # JPEG compression quality
    ):
        self.camera_id = camera_id
        self.face_recognition_enabled = face_recognition_enabled
        self.emotion_detection_enabled = emotion_detection_enabled
        self.face_detection_interval = face_detection_interval
        self.emotion_detection_interval = emotion_detection_interval
        self.advanced_features_enabled = advanced_features_enabled
        self.tracking_data_interval = tracking_data_interval
        self.stream_fps = stream_fps
        self.jpeg_quality = jpeg_quality

        # Performance tracking
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        self.last_perf_log = time.time()

    def should_process_face(self) -> bool:
        """Check if should process face detection this frame"""
        if not self.face_recognition_enabled:
            return False
        return self.frame_count % self.face_detection_interval == 0

    def should_process_emotion(self) -> bool:
        """Check if should process emotion detection this frame"""
        if not self.emotion_detection_enabled:
            return False
        return self.frame_count % self.emotion_detection_interval == 0

    def should_send_tracking_data(self) -> bool:
        """Check if should send tracking data this frame"""
        return self.frame_count % self.tracking_data_interval == 0

    def log_performance(self):
        """Log performance metrics every 5 seconds"""
        if time.time() - self.last_perf_log > 5.0 and self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            max_time = max(self.processing_times)

            features = []
            if self.face_recognition_enabled:
                features.append(f"Face@{self.face_detection_interval}f")
            if self.emotion_detection_enabled:
                features.append(f"Emotion@{self.emotion_detection_interval}f")
            features_str = ", ".join(features) if features else "None"

            logger.info(
                f"📊 {self.camera_id} Performance: "
                f"Avg={avg_time*1000:.1f}ms, Max={max_time*1000:.1f}ms, "
                f"FPS={1/avg_time:.1f} | Features: {features_str}"
            )
            self.last_perf_log = time.time()


# Store camera configurations
camera_configs: Dict[str, CameraConfig] = {}

# Store camera processes and shared memory
camera_processes: Dict[str, mp.Process] = {}
camera_shared_memory: Dict[str, SharedFrameBuffer] = {}
camera_tracking_queues: Dict[str, mp.Queue] = {}
camera_control_queues: Dict[str, mp.Queue] = {}


# ============ ASYNC LLM PROCESSOR ============
class AsyncLLMProcessor:
    """Non-blocking LLM processor that runs in background"""

    def __init__(self):
        self.llm_analyzer = None
        self.running = False
        self.worker_thread = None
        self.pending_requests = {}  # track_id -> future

    def initialize(self, llm_analyzer):
        """Initialize with LLM analyzer"""
        self.llm_analyzer = llm_analyzer

    def start(self):
        """Start background worker"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("🚀 Async LLM processor started")

    def _worker_loop(self):
        """Background worker that processes LLM requests"""
        while self.running:
            try:
                # Get request from queue (with timeout)
                request = llm_queue.get(timeout=0.1)

                if request is None:
                    continue

                track_id = request['track_id']
                features = request['features']
                risk_score = request['risk_score']
                camera_id = request['camera_id']
                callback = request['callback']

                try:
                    # Call LLM (this blocks, but in background thread)
                    logger.debug(f"🧠 Processing LLM for track {track_id}...")
                    llm_result = self.llm_analyzer.analyze(
                        features=features,
                        risk_score=risk_score,
                        track_id=track_id,
                        camera_id=camera_id
                    )
                    llm_result['llm_used'] = True

                    # Call callback with result
                    callback(llm_result, None)
                    logger.debug(f"✅ LLM complete for track {track_id}")

                except Exception as e:
                    logger.error(f"❌ LLM error for track {track_id}: {e}")
                    callback(None, e)

                llm_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"LLM worker error: {e}")

    def analyze_async(self, track_id, features, risk_score, camera_id, callback):
        """Queue LLM analysis (non-blocking)"""
        if not self.llm_analyzer:
            return False

        try:
            request = {
                'track_id': track_id,
                'features': features,
                'risk_score': risk_score,
                'camera_id': camera_id,
                'callback': callback
            }
            llm_queue.put_nowait(request)
            return True
        except queue.Full:
            logger.warning(f"⚠️ LLM queue full, skipping analysis for track {track_id}")
            return False

    def stop(self):
        """Stop background worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)


# Global async LLM processor
async_llm_processor = AsyncLLMProcessor()


# ============ ASYNC HTTP CLIENT ============
class AsyncHTTPClient:
    """Non-blocking HTTP client for sending data"""

    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.send_queue = queue.Queue(maxsize=100)
        self.running = False
        self.worker_thread = None

    def start(self):
        """Start background sender"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.worker_thread.start()
        logger.info("🚀 Async HTTP client started")

    def _sender_loop(self):
        """Background worker that sends HTTP requests"""
        import httpx
        client = httpx.Client(timeout=1.0)

        while self.running:
            try:
                request = self.send_queue.get(timeout=0.1)

                if request is None:
                    continue

                endpoint = request['endpoint']
                data = request['data']

                try:
                    response = client.post(f"{self.backend_url}{endpoint}", json=data)
                    if response.status_code not in [200, 201]:
                        logger.debug(f"HTTP {endpoint} failed: {response.status_code}")
                except Exception as e:
                    logger.debug(f"HTTP error {endpoint}: {e}")

                self.send_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"HTTP sender error: {e}")

        client.close()

    def send_async(self, endpoint: str, data: dict):
        """Queue HTTP request (non-blocking)"""
        try:
            self.send_queue.put_nowait({'endpoint': endpoint, 'data': data})
        except queue.Full:
            logger.warning(f"⚠️ HTTP queue full, dropping request to {endpoint}")

    def stop(self):
        """Stop background sender"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)


# Global async HTTP client
async_http_client = AsyncHTTPClient(NODE_BACKEND_URL)


# ============ TRACKING DATA CONSUMER ============
class TrackingDataConsumer:
    """Background worker that consumes tracking data from camera processes"""

    def __init__(self):
        self.running = False
        self.worker_thread = None

    def start(self):
        """Start background consumer"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self.worker_thread.start()
        logger.info("🚀 Tracking data consumer started")

    def _consumer_loop(self):
        """Background worker that consumes tracking data from queues"""
        while self.running:
            try:
                # Check all camera tracking queues
                for camera_id, tracking_queue in camera_tracking_queues.items():
                    try:
                        # Non-blocking get with small timeout
                        tracking_data = tracking_queue.get(timeout=0.01)

                        # Send tracking data to backend
                        if tracking_data and tracking_data.get('objects'):
                            send_tracking_data_async(camera_id, tracking_data)

                        # Send alert if present
                        if tracking_data and tracking_data.get('alert'):
                            alert = tracking_data['alert']

                            # Check if alert needs LLM analysis (medium risk)
                            if alert.get('needs_llm', False):
                                logger.info(f"🧠 {camera_id}: Medium risk alert for track {alert['track_id']}, queuing for LLM analysis...")

                                # Define callback to send alert after LLM analysis (with proper closure)
                                def make_llm_callback(alert_dict, cam_id):
                                    def callback(llm_result, error):
                                        if error:
                                            # LLM failed, use fallback
                                            logger.error(f"❌ {cam_id}: LLM failed for track {alert_dict['track_id']}: {error}")
                                            alert_dict['llm_analysis'] = {
                                                'risk_level': alert_dict.get('risk_level', 'medium'),
                                                'confidence': 0.5,
                                                'reasoning': f"LLM analysis failed: {error}. Using rule-based assessment.",
                                                'alert_message': f"Alert for track #{alert_dict['track_id']} - Risk {alert_dict.get('risk_score', 0)}/100",
                                                'recommended_action': 'monitor',
                                                'llm_used': False
                                            }
                                        else:
                                            # LLM succeeded
                                            alert_dict['llm_analysis'] = llm_result

                                        alert_dict['needs_llm'] = False
                                        logger.info(f"✅ {cam_id}: Alert ready for track {alert_dict['track_id']}, sending to backend")
                                        async_http_client.send_async("/api/alerts/from-detection", alert_dict)
                                    return callback

                                # Queue LLM analysis (non-blocking)
                                async_llm_processor.analyze_async(
                                    features=alert['features'],
                                    risk_score=alert['risk_score'],
                                    track_id=alert['track_id'],
                                    camera_id=camera_id,
                                    callback=make_llm_callback(alert, camera_id)
                                )
                            else:
                                # Critical risk or LLM already processed - send immediately
                                logger.info(f"📢 {camera_id}: Sending alert for track {alert['track_id']}, risk={alert['risk_score']}")
                                async_http_client.send_async("/api/alerts/from-detection", alert)

                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.debug(f"Error processing tracking data for {camera_id}: {e}")

                time.sleep(0.01)  # Small sleep to prevent CPU spinning

            except Exception as e:
                logger.error(f"❌ Tracking consumer error: {e}")
                time.sleep(0.1)

    def stop(self):
        """Stop background consumer"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)


# Global tracking data consumer
tracking_data_consumer = TrackingDataConsumer()


def update_frame(camera_id: str, frame: np.ndarray):
    """Update the latest frame for a camera (thread-safe) - Legacy, kept for compatibility"""
    if camera_id not in frame_locks:
        frame_locks[camera_id] = threading.Lock()

    with frame_locks[camera_id]:
        latest_frames[camera_id] = frame.copy()


def get_latest_frame(camera_id: str) -> Optional[np.ndarray]:
    """Get the latest frame for a camera from shared memory"""
    # Try shared memory first (multi-process)
    if camera_id in camera_shared_memory:
        return camera_shared_memory[camera_id].read_frame_nocopy()

    # Fallback to legacy dict-based storage (single-process)
    if camera_id not in frame_locks:
        return None

    with frame_locks[camera_id]:
        return latest_frames.get(camera_id, None)


def encode_frame(frame: np.ndarray, quality: int = 50) -> bytes:
    """
    Encode frame to JPEG (synchronous, runs in thread pool)

    This function is designed to be called via asyncio.to_thread()
    """
    try:
        _, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ])
        return buffer.tobytes()
    except Exception as e:
        logger.error(f"JPEG encoding error: {e}")
        return b''


# ============ UTILITY FUNCTIONS ============

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


def extract_tracking_data_optimized(
    result, camera_id: str, config: CameraConfig,
    track_history, last_positions, first_seen, platform_poly,
    aggregator, scorer, recent_alerts, pending_llm_alerts, ts,
    frame=None, face_service=None, emotion_service=None,
    advanced_features_extractor=None, location="Unknown",
    pa_system=None, edge_tracker=None
):
    """
    OPTIMIZED tracking data extraction with:
    - Conditional face/emotion processing based on camera config
    - Async LLM calls
    - Reduced processing overhead
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

    # Check if we should process face/emotion this frame
    should_process_face = config.should_process_face()
    should_process_emotion = config.should_process_emotion()

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

        # ============ CONDITIONAL FACE RECOGNITION ============
        persistent_id = None
        face_embedding = None
        face_confidence = None

        if should_process_face and face_service and frame is not None:
            try:
                bbox = [x1, y1, x2, y2]
                face_embedding = face_service.extract_face_embedding(frame, bbox)

                if face_embedding is not None:
                    persistent_id = face_service.get_or_create_track_id(
                        face_embedding, camera_id, location
                    )
                    face_service.map_yolo_to_persistent_id(
                        camera_id, track_id, face_embedding, location
                    )
                    face_confidence = face_service.get_match_confidence(persistent_id, face_embedding)
                    logger.debug(f"👤 Face: Track {track_id} -> {persistent_id}")
            except Exception as e:
                logger.debug(f"Face recognition error: {e}")

        # ============ CONDITIONAL EMOTION DETECTION ============
        emotion = None
        emotion_confidence = None
        emotion_risk_assessment = None

        if should_process_emotion and emotion_service and frame is not None:
            try:
                bbox = [x1, y1, x2, y2]
                emotion, emotion_confidence = emotion_service.detect_emotion(frame, bbox)

                if emotion and persistent_id:
                    emotion_service.track_emotion(persistent_id, emotion, emotion_confidence)
                    emotion_context = {
                        'near_edge': dist_edge < 100,
                        'loitering': dwell > 10,
                        'aggressive_movement': speed > 50
                    }
                    emotion_risk_assessment = emotion_service.assess_emotional_risk(
                        emotion, emotion_confidence, emotion_context
                    )
                    logger.debug(f"😊 Emotion: {persistent_id} - {emotion}")
            except Exception as e:
                logger.debug(f"Emotion detection error: {e}")

        # ============ CONDITIONAL ADVANCED FEATURES ============
        advanced_features = {}

        if config.advanced_features_enabled and advanced_features_extractor and keypoints is not None and i < len(keypoints):
            try:
                kp = keypoints[i]
                # Only compute lightweight features
                advanced_features['shoulder_hunch'] = advanced_features_extractor.compute_shoulder_hunch(kp)
                advanced_features['head_down'] = advanced_features_extractor.compute_head_down_ratio(track_id, kp, ts)
                advanced_features['edge_transgression_count'] = advanced_features_extractor.compute_edge_transgression_count(
                    track_id, center, platform_poly, ts
                ) if platform_poly else 0
            except Exception as e:
                logger.debug(f"Advanced features error: {e}")

        # Default risk score
        risk_score = 0
        risk_level = "normal"

        # ============ BRAIN LOGIC ============
        if BRAIN_AVAILABLE and aggregator:
            frame_features = {
                'bbox_center': center,
                'torso_angle': torso_angle if torso_angle is not None else 90.0,
                'speed': speed,
                'dist_to_edge': dist_edge
            }

            if advanced_features:
                frame_features.update(advanced_features)

            aggregator.add_frame_features(track_id, ts, frame_features)
            agg_features = aggregator.get_aggregated_features(track_id)

            if agg_features is not None and scorer:
                risk_score, _ = scorer.compute_risk(agg_features)

                # Enhance with emotion
                if emotion_risk_assessment:
                    emotion_risk_score = emotion_risk_assessment.get('risk_score', 0)
                    risk_score = int(risk_score * 0.7 + emotion_risk_score * 0.3)

                # Enhance with advanced features
                if advanced_features:
                    risk_boost = 0
                    shoulder_hunch = advanced_features.get('shoulder_hunch')
                    if shoulder_hunch is not None and shoulder_hunch > 0.7:
                        risk_boost += 10
                    head_down = advanced_features.get('head_down')
                    if head_down is not None and head_down > 0.6:
                        risk_boost += 7
                    edge_count = advanced_features.get('edge_transgression_count')
                    if edge_count is not None and edge_count >= 3:
                        risk_boost += 20
                    risk_score = min(100, risk_score + risk_boost)

                risk_level = scorer.get_risk_level(risk_score)

                # ============ EDGE PROXIMITY PA ANNOUNCEMENT ============
                # Check if person has been near edge for too long
                if edge_tracker and pa_system and dist_edge < 200:  # Only check if reasonably close
                    proximity_status = edge_tracker.update(track_id, dist_edge, ts)

                    if proximity_status.get('should_announce'):
                        # Calculate normalized risk score (0-1 scale)
                        risk_score_normalized = risk_score / 100.0

                        logger.warning(f"🔊 PA ANNOUNCEMENT TRIGGERED for Track {track_id}")
                        logger.warning(f"   Risk Score: {risk_score}/100 ({risk_score_normalized:.2f})")
                        logger.warning(f"   Distance: {dist_edge:.0f}px")
                        logger.warning(f"   Time near edge: {proximity_status['time_near_edge']:.1f}s")

                        # Trigger PA announcement in background thread (non-blocking)
                        def trigger_pa_announcement():
                            try:
                                if risk_score_normalized >= 0.95:
                                    # Level 3: Emergency - pre-cached announcement
                                    logger.info(f"🚨🚨 LEVEL 3 EMERGENCY PA (risk: {risk_score_normalized:.2f})")
                                    pa_system.play_level3_announcement()
                                elif risk_score_normalized >= 0.80:
                                    # Level 2: Critical - pre-cached announcement
                                    logger.info(f"🚨 LEVEL 2 CRITICAL PA (risk: {risk_score_normalized:.2f})")
                                    pa_system.play_level2_announcement()
                                elif risk_score_normalized >= 0.40:
                                    # Level 1: Gentle intervention - streaming
                                    logger.info(f"⚠️ LEVEL 1 GENTLE PA (risk: {risk_score_normalized:.2f})")
                                    pa_system.play_level1_announcement(platform=camera_id)
                                else:
                                    logger.info(f"ℹ️ Risk {risk_score_normalized:.2f} below PA threshold")
                            except Exception as e:
                                logger.error(f"❌ PA announcement failed: {e}")

                        # Run in background to avoid blocking detection
                        import threading
                        threading.Thread(target=trigger_pa_announcement, daemon=True).start()

                # ============ OPTIMIZED ALERT ROUTING ============
                if risk_score >= MEDIUM_THRESHOLD:
                    last_alert_time = recent_alerts.get(track_id, 0)

                    if (ts - last_alert_time) > ALERT_COOLDOWN:

                        if risk_score >= CRITICAL_THRESHOLD:
                            # ⚡ FAST PATH: Critical risk
                            logger.warning(f"⚡ CRITICAL: Track {track_id} - Risk {risk_score}")

                            alert_parts = [f"CRITICAL: Person #{track_id}"]
                            if agg_features.get('min_dist_to_edge', 999) < 100:
                                alert_parts.append(f"very close to edge")

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
                                    "reasoning": "FAST PATH: Critical urgency",
                                    "alert_message": alert_message,
                                    "recommended_action": recommended_action,
                                    "llm_used": False
                                }
                            }

                        else:
                            # 🧠 ASYNC LLM PATH: Medium risk
                            logger.warning(f"🧠 MEDIUM: Track {track_id} - Risk {risk_score} (queuing LLM)")

                            # Check if LLM result already available
                            if track_id in pending_llm_alerts:
                                llm_result = pending_llm_alerts[track_id]
                                alert = {
                                    "track_id": track_id,
                                    "camera_id": camera_id,
                                    "risk_score": risk_score,
                                    "risk_level": risk_level,
                                    "timestamp": ts,
                                    "features": agg_features,
                                    "llm_analysis": llm_result
                                }
                                logger.info(f"✅ Using cached LLM result for track {track_id}")
                                del pending_llm_alerts[track_id]
                            else:
                                # Queue async LLM call
                                def llm_callback(llm_result, error):
                                    if llm_result:
                                        pending_llm_alerts[track_id] = llm_result
                                        # Send alert asynchronously
                                        alert_data = {
                                            "track_id": track_id,
                                            "camera_id": camera_id,
                                            "risk_score": risk_score,
                                            "risk_level": risk_level,
                                            "timestamp": ts,
                                            "features": agg_features,
                                            "llm_analysis": llm_result
                                        }
                                        async_http_client.send_async("/api/alerts/from-detection", alert_data)
                                        logger.info(f"📤 Alert sent async for track {track_id}")

                                success = async_llm_processor.analyze_async(
                                    track_id, agg_features, risk_score, camera_id, llm_callback
                                )

                                if not success:
                                    # Fallback if queue full
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
                                            "reasoning": "LLM queue full - fast path",
                                            "alert_message": f"Person #{track_id} - Risk {risk_score}/100",
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
            "speed": speed,
            "dist_to_edge": dist_edge,
            "dwell_time": dwell,
            "torso_angle": torso_angle,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "persistent_id": persistent_id,
            "face_confidence": face_confidence,
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
            "emotion_risk": emotion_risk_assessment.get('risk_score') if emotion_risk_assessment else None,
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


def send_tracking_data_async(camera_id: str, tracking_data: dict):
    """Send tracking data using async client (non-blocking)"""
    # Send tracking data
    async_http_client.send_async(f"/api/tracking/{camera_id}", tracking_data)

    # Send person data
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
            async_http_client.send_async("/api/persons/update", person_data)

    # Send alert if present (immediate alerts only, async LLM alerts sent in callback)
    if tracking_data.get('alert'):
        async_http_client.send_async("/api/alerts/from-detection", tracking_data['alert'])


async def frame_generator(camera_id: str):
    """Generator that yields frames as MJPEG stream with async encoding"""
    logger.info(f"Starting stream for camera: {camera_id}")

    config = camera_configs.get(camera_id)
    if not config:
        logger.error(f"No config found for {camera_id}")
        return

    sleep_time = 1.0 / config.stream_fps

    while True:
        frame = get_latest_frame(camera_id)

        if frame is None:
            await asyncio.sleep(0.033)
            continue

        # Make a copy since we're encoding asynchronously
        # (frame from shared memory is read-only)
        frame_copy = frame.copy()

        # Encode frame asynchronously (non-blocking!)
        frame_bytes = await asyncio.to_thread(
            encode_frame,
            frame_copy,
            config.jpeg_quality
        )

        if not frame_bytes:
            await asyncio.sleep(0.033)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        await asyncio.sleep(sleep_time)


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
        "llm_queue_size": llm_queue.qsize(),
        "http_queue_size": async_http_client.send_queue.qsize(),
        "frame_count": {cam: 1 if get_latest_frame(cam) is not None else 0
                       for cam in latest_frames.keys()}
    }


@app.get("/performance/{camera_id}")
async def get_performance(camera_id: str):
    """Get performance metrics for a camera"""
    config = camera_configs.get(camera_id)
    if not config:
        return {"error": "Camera not found"}

    avg_time = sum(config.processing_times) / len(config.processing_times) if config.processing_times else 0

    return {
        "camera_id": camera_id,
        "avg_processing_time_ms": avg_time * 1000,
        "current_fps": 1 / avg_time if avg_time > 0 else 0,
        "frame_count": config.frame_count,
        "config": {
            "face_recognition_enabled": config.face_recognition_enabled,
            "emotion_detection_enabled": config.emotion_detection_enabled,
            "face_interval": config.face_detection_interval,
            "emotion_interval": config.emotion_detection_interval,
            "advanced_features_enabled": config.advanced_features_enabled,
            "tracking_interval": config.tracking_data_interval,
            "stream_fps": config.stream_fps,
            "jpeg_quality": config.jpeg_quality
        }
    }


@app.get("/config/{camera_id}")
async def get_camera_config(camera_id: str):
    """Get current configuration for a camera"""
    config = camera_configs.get(camera_id)
    if not config:
        return {"error": "Camera not found"}

    return {
        "camera_id": camera_id,
        "face_recognition_enabled": config.face_recognition_enabled,
        "emotion_detection_enabled": config.emotion_detection_enabled,
        "face_detection_interval": config.face_detection_interval,
        "emotion_detection_interval": config.emotion_detection_interval,
        "advanced_features_enabled": config.advanced_features_enabled,
        "tracking_data_interval": config.tracking_data_interval,
        "stream_fps": config.stream_fps,
        "jpeg_quality": config.jpeg_quality
    }


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "MetroEye Vision Engine (Optimized)",
        "version": "3.0.0-optimized",
        "features": {
            "brain_logic": BRAIN_AVAILABLE,
            "face_recognition": FACE_RECOGNITION_AVAILABLE,
            "emotion_detection": EMOTION_RECOGNITION_AVAILABLE,
            "advanced_behavioral_analysis": ADVANCED_FEATURES_AVAILABLE,
            "async_llm": True,
            "async_http": True,
            "per_camera_config": True
        },
        "endpoints": {
            "streaming": "/stream/{camera_id}",
            "frame": "/frame/{camera_id}",
            "config": "/config/{camera_id}",
            "performance": "/performance/{camera_id}",
            "health": "/health"
        },
        "docs": "/docs"
    }


# ============ OPTIMIZED DETECTION LOOP ============

def run_detection_optimized(camera_id: str, video_source: str, config: CameraConfig):
    """
    OPTIMIZED detection loop with:
    - Per-camera configuration
    - Conditional feature processing
    - Performance monitoring
    - Async communication
    - Adaptive frame skipping
    """
    from ultralytics import YOLO

    logger.info(f"🎬 Starting optimized detection for {camera_id}: {video_source}")

    # Load YOLO model with optimizations
    model = YOLO('yolo26n-pose.pt')

    # Performance tracking
    max_processing_time = 1.0 / 15  # Target max 15 FPS processing (66ms)
    skip_heavy_processing = False

    # Initialize brain modules
    aggregator = None
    scorer = None
    if BRAIN_AVAILABLE:
        aggregator = FeatureAggregator(window_seconds=WINDOW_SECONDS)
        scorer = RuleBasedScorer()
        logger.info("✓ Brain modules initialized")

    # Initialize face recognition service
    face_service = None
    if FACE_RECOGNITION_AVAILABLE and config.face_recognition_enabled:
        try:
            face_service = FaceRecognitionService(backend_url=NODE_BACKEND_URL)
            logger.info(f"✓ Face recognition ENABLED (every {config.face_detection_interval} frames)")
        except Exception as e:
            logger.warning(f"⚠ Face recognition failed: {e}")
    elif not config.face_recognition_enabled:
        logger.info(f"⊗ Face recognition DISABLED for {camera_id}")

    # Initialize emotion recognition service
    emotion_service = None
    if EMOTION_RECOGNITION_AVAILABLE and config.emotion_detection_enabled:
        try:
            emotion_service = EmotionRecognitionService()
            logger.info(f"✓ Emotion recognition ENABLED (every {config.emotion_detection_interval} frames)")
        except Exception as e:
            logger.warning(f"⚠ Emotion recognition failed: {e}")
    elif not config.emotion_detection_enabled:
        logger.info(f"⊗ Emotion recognition DISABLED for {camera_id}")

    # Initialize advanced features extractor
    advanced_features_extractor = None
    if ADVANCED_FEATURES_AVAILABLE and config.advanced_features_enabled:
        try:
            advanced_features_extractor = AdvancedFeatureExtractor()
            logger.info("✓ Advanced features extractor")
        except Exception as e:
            logger.warning(f"⚠ Advanced features failed: {e}")

    # Load calibration
    calibration_manager = CalibrationManager()
    platform_poly = calibration_manager.load_calibration(camera_id)

    if platform_poly is None:
        logger.warning(f"No calibration found for {camera_id} - auto-calibrating...")
        platform_poly = calibration_manager.calibrate(video_source, camera_id, method='auto')

    if platform_poly:
        logger.info(f"✓ Platform calibrated: {len(platform_poly)} vertices")

    # Initialize PA announcement system
    pa_system = None
    if PA_SYSTEM_AVAILABLE:
        try:
            import os
            pa_system = PAAnnouncement(
                elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
                use_elevenlabs=True,
                model="eleven_flash_v2_5"  # Ultra-low latency (~75ms)
            )
            logger.info("🔊 PA announcement system initialized with ElevenLabs streaming")
        except Exception as e:
            logger.warning(f"⚠ PA system initialization failed: {e}")
            pa_system = None

    # Initialize edge proximity tracker
    edge_tracker = EdgeProximityTracker(
        distance_threshold=100.0,  # pixels - how close is "too close"
        dwell_threshold=5.0,  # seconds - how long at edge triggers alert
        announcement_cooldown=15.0  # seconds between announcements
    )
    logger.info("✓ Edge proximity tracker initialized (100px, 5s dwell, 15s cooldown)")

    # Tracking state
    track_history = defaultdict(lambda: deque())
    last_positions = {}
    first_seen = {}
    recent_alerts = {}
    pending_llm_alerts = {}  # Store async LLM results

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        logger.error(f"Could not open video source: {video_source}")
        return

    # Set buffer size to 1 to reduce lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    logger.info(f"✅ {camera_id} ready - streaming at {config.stream_fps} FPS")

    consecutive_fails = 0
    yolo_skip_counter = 0

    while True:
        loop_start = time.time()

        ret, frame = cap.read()

        if not ret:
            consecutive_fails += 1
            if consecutive_fails > 30:
                logger.warning(f"End of video for {camera_id}, looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                consecutive_fails = 0
            continue

        consecutive_fails = 0
        config.frame_count += 1
        ts = time.time()

        # Adaptive YOLO skipping - skip every other frame if processing is slow
        yolo_skip_counter += 1
        should_run_yolo = True

        if skip_heavy_processing and yolo_skip_counter % 2 == 0:
            should_run_yolo = False

        # Run YOLO detection (with adaptive skipping)
        if should_run_yolo:
            results = model.track(frame, persist=True, conf=0.3, verbose=False)
        else:
            # Skip YOLO, use last annotated frame
            if get_latest_frame(camera_id) is not None:
                continue

        # Draw annotations (lightweight)
        try:
            annotated_frame = results[0].plot()

            # Draw platform polygon (only if not skipping)
            if platform_poly and not skip_heavy_processing:
                cv2.polylines(annotated_frame, [np.array(platform_poly)], True, (0, 255, 0), 2)

            # Draw lightweight overlays - ONLY risk score
            if results[0].boxes is not None and results[0].boxes.id is not None and not skip_heavy_processing:
                boxes = results[0].boxes.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()

                for i, box in enumerate(boxes):
                    track_id = int(ids[i])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Only draw risk score (lightweight)
                    if BRAIN_AVAILABLE and aggregator:
                        agg_features = aggregator.get_aggregated_features(track_id)
                        if agg_features is not None and scorer:
                            risk_score, _ = scorer.compute_risk(agg_features)
                            risk_color = (0, 255, 0) if risk_score < 30 else (0, 165, 255) if risk_score < 70 else (0, 0, 255)
                            cv2.putText(annotated_frame, f"R:{risk_score}",
                                       (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.5, risk_color, 1)

            # Update frame for streaming
            update_frame(camera_id, annotated_frame)
        except Exception as e:
            logger.debug(f"Frame annotation error: {e}")
            # Use raw frame if annotation fails
            update_frame(camera_id, frame)

        # Extract and send tracking data (conditional)
        if config.should_send_tracking_data():
            try:
                tracking_data = extract_tracking_data_optimized(
                    results[0], camera_id, config, track_history, last_positions,
                    first_seen, platform_poly, aggregator, scorer, recent_alerts,
                    pending_llm_alerts, ts,
                    frame=frame,
                    face_service=face_service,
                    emotion_service=emotion_service,
                    advanced_features_extractor=advanced_features_extractor,
                    location=f"Platform {camera_id}",
                    pa_system=pa_system,
                    edge_tracker=edge_tracker
                )

                if tracking_data['objects']:
                    send_tracking_data_async(camera_id, tracking_data)

                # Periodic cleanup of edge tracker (every 100 frames)
                if edge_tracker and config.frame_count % 100 == 0:
                    active_tracks = {obj['track_id'] for obj in tracking_data['objects']}
                    edge_tracker.cleanup_old_tracks(active_tracks, ts)

            except Exception as e:
                logger.error(f"❌ {camera_id}: Tracking data extraction error: {e}")
                # Continue processing even if tracking fails

        # Track performance
        processing_time = time.time() - loop_start
        config.processing_times.append(processing_time)
        config.log_performance()

        # Adaptive performance management
        if processing_time > max_processing_time:
            if not skip_heavy_processing:
                skip_heavy_processing = True
                logger.warning(f"⚠️ {camera_id}: Enabling performance mode (processing too slow: {processing_time*1000:.1f}ms)")
        elif processing_time < max_processing_time * 0.7:
            if skip_heavy_processing:
                skip_heavy_processing = False
                logger.info(f"✅ {camera_id}: Disabling performance mode (processing fast enough: {processing_time*1000:.1f}ms)")

        # Minimal sleep to prevent CPU overload
        time.sleep(0.001)

    cap.release()


def start_camera_process_new(camera_id: str, video_source: str, config: CameraConfig):
    """Start detection in separate process (multi-process architecture)"""
    camera_configs[camera_id] = config

    try:
        # Get frame dimensions from first frame of video
        cap = cv2.VideoCapture(video_source)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.error(f"❌ {camera_id}: Could not read frame to determine dimensions")
            return

        height, width = frame.shape[:2]
        logger.info(f"📏 {camera_id}: Frame dimensions: {width}x{height}")

        # Create shared memory buffer
        shm_buffer = create_frame_buffer(camera_id, width, height)
        camera_shared_memory[camera_id] = shm_buffer

        # Convert CameraConfig to dict for pickling
        config_dict = {
            'face_recognition_enabled': config.face_recognition_enabled,
            'emotion_detection_enabled': config.emotion_detection_enabled,
            'face_detection_interval': config.face_detection_interval,
            'emotion_detection_interval': config.emotion_detection_interval,
            'advanced_features_enabled': config.advanced_features_enabled,
            'tracking_data_interval': config.tracking_data_interval,
            'stream_fps': config.stream_fps,
            'jpeg_quality': config.jpeg_quality
        }

        # Start camera process
        process, tracking_q, control_q = start_camera_process(
            camera_id=camera_id,
            video_source=video_source,
            config=config_dict,
            shared_frame_buffer=shm_buffer
        )

        camera_processes[camera_id] = process
        camera_tracking_queues[camera_id] = tracking_q
        camera_control_queues[camera_id] = control_q

        logger.info(f"✅ {camera_id}: Camera process started (PID: {process.pid})")

    except Exception as e:
        logger.error(f"❌ {camera_id}: Failed to start camera process: {e}")
        import traceback
        logger.error(traceback.format_exc())


def cleanup_all_processes():
    """Cleanup all camera processes and shared memory"""
    logger.info("🧹 Cleaning up all camera processes...")

    # Stop background workers
    tracking_data_consumer.stop()
    async_http_client.stop()
    async_llm_processor.stop()

    # Stop all processes
    for camera_id, control_q in camera_control_queues.items():
        try:
            control_q.put("stop")
            logger.info(f"Sent stop command to {camera_id}")
        except Exception as e:
            logger.error(f"Error sending stop to {camera_id}: {e}")

    # Wait for processes to finish
    for camera_id, process in camera_processes.items():
        try:
            process.join(timeout=3.0)
            if process.is_alive():
                logger.warning(f"Force terminating {camera_id}")
                process.terminate()
                process.join(timeout=1.0)
            logger.info(f"✅ {camera_id} process stopped")
        except Exception as e:
            logger.error(f"Error stopping {camera_id}: {e}")

    # Cleanup shared memory
    for camera_id, shm_buffer in camera_shared_memory.items():
        try:
            shm_buffer.close()
            shm_buffer.unlink()
            logger.info(f"✅ {camera_id} shared memory cleaned")
        except Exception as e:
            logger.error(f"Error cleaning shared memory for {camera_id}: {e}")

    logger.info("✅ Cleanup complete")


if __name__ == "__main__":
    import uvicorn
    import atexit
    import signal

    # Register cleanup handlers
    atexit.register(cleanup_all_processes)

    def signal_handler(signum, frame_info):
        """Handle Ctrl+C gracefully"""
        logger.info("\n🛑 Received shutdown signal, cleaning up...")
        cleanup_all_processes()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start async processors
    async_http_client.start()
    tracking_data_consumer.start()

    # Initialize LLM processor
    if BRAIN_AVAILABLE:
        try:
            llm_analyzer = LLMAnalyzer()
            async_llm_processor.initialize(llm_analyzer)
            async_llm_processor.start()
        except Exception as e:
            logger.warning(f"⚠ LLM analyzer unavailable: {e}")

    # ============ CONFIGURE CAMERAS ============
    # Per-Camera Configuration Guide:
    #
    # face_recognition_enabled: True/False - Enable/disable face recognition for this camera
    # emotion_detection_enabled: True/False - Enable/disable emotion detection for this camera
    # face_detection_interval: 20-30 frames - Process face every N frames (lower = more frequent, higher CPU)
    # emotion_detection_interval: 20-30 frames - Process emotion every N frames (lower = more frequent, higher CPU)
    # advanced_features_enabled: True/False - Enable/disable advanced behavioral analysis
    # tracking_data_interval: 5-15 frames - Send tracking data every N frames (lower = more updates, higher network)
    # stream_fps: 15-30 - Target streaming frame rate (lower = better performance, less smooth)
    # jpeg_quality: 30-70 - JPEG compression quality (lower = better performance, worse quality)
    #
    # Performance Tips:
    # - For laggy streams: Set face/emotion to False or increase intervals to 40-50
    # - For high-security areas: Enable all features, reduce intervals to 20-25
    # - For low-traffic areas: Disable face/emotion, increase tracking_interval to 15
    # - For bandwidth issues: Lower stream_fps to 15 and jpeg_quality to 40

    cameras = [
        {
            "camera_id": "camera_1",
            "video_source": "sample-station.webm",
            "config": CameraConfig(
                camera_id="camera_1",
                face_recognition_enabled=False,    # Enable face recognition for this camera
                emotion_detection_enabled=False,   # Enable emotion detection for this camera
                face_detection_interval=30,       # Face every 30 frames (~1.5s at 20fps) - INCREASED for performance
                emotion_detection_interval=30,    # Emotion every 30 frames - INCREASED for performance
                advanced_features_enabled=False,  # DISABLED for performance - can enable later if smooth
                tracking_data_interval=20,        # Send data every 15 frames - INCREASED for performance
                stream_fps=20,                    # Stream at 20 FPS
                jpeg_quality=40                   # Medium quality
            )
        },
        {
            "camera_id": "camera_2",
            "video_source": "sample4.mp4",
            "config": CameraConfig(
                camera_id="camera_2",
                face_recognition_enabled=False,   # DISABLE face recognition for this camera
                emotion_detection_enabled=False,  # DISABLE emotion detection for this camera
                face_detection_interval=30,       # (Ignored if disabled)
                emotion_detection_interval=30,    # (Ignored if disabled)
                advanced_features_enabled=False,  # DISABLED for maximum performance
                tracking_data_interval=20,        # Send data every 15 frames
                stream_fps=20,
                jpeg_quality=40                  # Slightly lower quality for performance
            )
        },
 {
            "camera_id": "camera_3",
            "video_source": 0,  # 0 = Default webcam, change to RTSP URL if using IP camera
            # For IP camera: "rtsp://username:password@ip:port/stream"
            # For USB camera index: 0, 1, 2, etc.
            "config": CameraConfig(
                camera_id="camera_3",
                face_recognition_enabled=False,    # ❌ DISABLED for performance - enable after GPU optimization
                emotion_detection_enabled=False,   # ❌ DISABLED for performance - enable after GPU optimization
                face_detection_interval=25,        # (Ignored if disabled)
                emotion_detection_interval=25,     # (Ignored if disabled)
                advanced_features_enabled=False,   # ❌ DISABLED for performance - enable after GPU optimization
                tracking_data_interval=20,         # Send updates every 20 frames - matches other cameras
                stream_fps=20,                     # 20 FPS - matches other cameras for consistency
                jpeg_quality=40                    # Medium quality - matches other cameras
            )
        },
    ]

    logger.info("=" * 70)
    logger.info("🚀 STARTING OPTIMIZED METROYE VISION ENGINE (MULTI-PROCESS)")
    logger.info("=" * 70)

    for cam in cameras:
        # Use new multi-process architecture
        start_camera_process_new(cam["camera_id"], cam["video_source"], cam["config"])

        # Show feature status
        features_status = []
        if cam['config'].face_recognition_enabled:
            features_status.append(f"Face@{cam['config'].face_detection_interval}f")
        else:
            features_status.append("Face=OFF")

        if cam['config'].emotion_detection_enabled:
            features_status.append(f"Emotion@{cam['config'].emotion_detection_interval}f")
        else:
            features_status.append("Emotion=OFF")

        if cam['config'].advanced_features_enabled:
            features_status.append("Advanced=ON")
        else:
            features_status.append("Advanced=OFF")

        logger.info(f"  📹 {cam['camera_id']}: {', '.join(features_status)}")

    # Give processes a moment to start
    time.sleep(2)

    logger.info("=" * 70)
    logger.info("🎯 Feature Status:")
    logger.info(f"  🧠 Brain Logic: {'✅' if BRAIN_AVAILABLE else '❌'}")
    logger.info(f"  👤 Face Recognition: {'✅' if FACE_RECOGNITION_AVAILABLE else '❌'}")
    logger.info(f"  😊 Emotion Detection: {'✅' if EMOTION_RECOGNITION_AVAILABLE else '❌'}")
    logger.info(f"  🎭 Advanced Features: {'✅' if ADVANCED_FEATURES_AVAILABLE else '❌'}")
    logger.info(f"  ⚡ Async LLM: ✅")
    logger.info(f"  📤 Async HTTP: ✅")
    logger.info(f"  🔄 Multi-Process: ✅ ({len(camera_processes)} camera processes)")
    logger.info(f"  💾 Shared Memory: ✅ (Zero-copy frame transfer)")
    logger.info("=" * 70)
    logger.info("🌐 Endpoints:")
    logger.info(f"  📹 Streams: http://localhost:5000/stream/{{camera_id}}")
    logger.info(f"  📊 Performance: http://localhost:5000/performance/{{camera_id}}")
    logger.info(f"  🏥 Health: http://localhost:5000/health")
    logger.info(f"  📖 Docs: http://localhost:5000/docs")
    logger.info("=" * 70)
    logger.info("💡 TIP: Use Ctrl+C to gracefully shutdown all camera processes")
    logger.info("=" * 70)

    # Run server with single worker (important for multiprocessing!)
    uvicorn.run(app, host="0.0.0.0", port=5000, workers=1)
