"""
Camera Process - Isolated Process for Each Camera
==================================================

Runs in separate process to eliminate GIL contention.
Each camera has dedicated CPU time and resources.

Features:
- Producer-consumer frame reading
- YOLO detection and tracking
- Feature extraction
- Shared memory frame output
- Inter-process communication for tracking data
"""

import multiprocessing as mp
from multiprocessing import Queue
import time
import logging
import numpy as np
from collections import defaultdict, deque
from typing import Optional, Dict
import queue
import sys
import traceback

# Import detection modules
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from brain.feature_aggregator import FeatureAggregator
    from brain.rule_based_scorer import RuleBasedScorer
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False

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

# Import utilities
from frame_reader import FrameReader
from shared_frame_buffer import SharedFrameBuffer
from edge_calibrator import CalibrationManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import feature extraction functions (will be moved here)
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


def extract_tracking_data_optimized(result, camera_id: str, track_history, last_positions, first_seen,
                                    platform_poly, aggregator, scorer, recent_alerts, ts, frame=None,
                                    face_service=None, emotion_service=None, advanced_features_extractor=None,
                                    frame_count=0, config=None):
    """
    Extract tracking data with features, risk scoring, and alert generation
    Optimized for per-camera configuration
    """
    objects = []
    alert = None

    # Threshold configuration
    MEDIUM_THRESHOLD = 60  # Medium risk triggers LLM analysis (raised from 40 - CPU bottleneck)
    CRITICAL_THRESHOLD = 85
    ALERT_COOLDOWN = 10.0

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
        WINDOW_SECONDS = 4.0
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

        # ============ FACE RECOGNITION (if enabled) ============
        persistent_id = None
        face_confidence = None

        should_process_face = (config and config.get('face_recognition_enabled', False) and
                              frame_count % config.get('face_detection_interval', 25) == 0)

        if should_process_face and face_service and frame is not None:
            try:
                bbox = [x1, y1, x2, y2]
                face_embedding = face_service.extract_face_embedding(frame, bbox)

                if face_embedding is not None:
                    persistent_id = face_service.get_or_create_track_id(
                        face_embedding, camera_id, "Unknown"
                    )
                    face_service.map_yolo_to_persistent_id(
                        camera_id, track_id, face_embedding, "Unknown"
                    )
                    face_confidence = face_service.get_match_confidence(persistent_id, face_embedding)
                    logger.debug(f"Face detected: Track {track_id} -> {persistent_id}")
            except Exception as e:
                logger.debug(f"Face recognition failed for track {track_id}: {e}")

        # ============ EMOTION DETECTION (if enabled) ============
        emotion = None
        emotion_confidence = None
        emotion_risk_assessment = None

        should_process_emotion = (config and config.get('emotion_detection_enabled', False) and
                                 frame_count % config.get('emotion_detection_interval', 25) == 0)

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
                    logger.debug(f"Emotion: {persistent_id} - {emotion}")
            except Exception as e:
                logger.debug(f"Emotion detection failed for track {track_id}: {e}")

        # ============ ADVANCED FEATURES (if enabled) ============
        advanced_features = {}

        should_process_advanced = config and config.get('advanced_features_enabled', False)

        if should_process_advanced and advanced_features_extractor and keypoints is not None and i < len(keypoints):
            try:
                kp = keypoints[i]
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
            except Exception as e:
                logger.debug(f"Advanced feature extraction failed: {e}")

        # ============ RISK SCORING ============
        risk_score = 0
        risk_level = "normal"

        if aggregator and scorer:
            # Calculate bbox dimensions for context
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height

            frame_features = {
                'bbox_center': center,
                'torso_angle': torso_angle if torso_angle is not None else 90.0,
                'speed': speed,
                'dist_to_edge': dist_edge,
                'bbox_width': bbox_width,  # Width of person bbox
                'bbox_height': bbox_height,  # Height of person bbox
                'bbox_area': bbox_area  # Total area (indicates size/distance)
            }

            if advanced_features:
                frame_features.update(advanced_features)

            aggregator.add_frame_features(track_id, ts, frame_features)
            agg_features = aggregator.get_aggregated_features(track_id)

            if agg_features is not None:
                risk_score, _ = scorer.compute_risk(agg_features)

                # Enhance with emotion risk
                if emotion_risk_assessment:
                    emotion_risk_score = emotion_risk_assessment.get('risk_score', 0)
                    risk_score = int(risk_score * 0.7 + emotion_risk_score * 0.3)

                # Enhance with advanced features
                if advanced_features:
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

                risk_level = scorer.get_risk_level(risk_score)

                # ============ ALERT GENERATION ============
                if risk_score >= MEDIUM_THRESHOLD:
                    last_alert_time = recent_alerts.get(track_id, 0)

                    if (ts - last_alert_time) > ALERT_COOLDOWN:
                        if risk_score >= CRITICAL_THRESHOLD:
                            # Critical alert - fast path
                            logger.warning(f"⚡ CRITICAL: Track {track_id} - Risk {risk_score}")

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
                                    "reasoning": "FAST PATH: Rule-based detection",
                                    "alert_message": alert_message,
                                    "recommended_action": recommended_action,
                                    "llm_used": False
                                }
                            }
                        else:
                            # Medium risk - mark for LLM processing in main process
                            logger.warning(f"⚠️ MEDIUM: Track {track_id} - Risk {risk_score} (LLM analysis pending)")
                            alert = {
                                "track_id": track_id,
                                "camera_id": camera_id,
                                "risk_score": risk_score,
                                "risk_level": risk_level,
                                "timestamp": ts,
                                "features": agg_features,
                                "needs_llm": True,  # Flag for main process to call LLM
                                "llm_analysis": None  # Will be populated by LLM
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
        "objects": objects,
        "alert": alert
    }


class CameraProcess(mp.Process):
    """
    Isolated process for camera detection and processing

    Runs completely independently from other cameras.
    Communicates via shared memory (frames) and queues (tracking data).
    """

    def __init__(
        self,
        camera_id: str,
        video_source: str,
        config: dict,
        shared_frame_buffer: SharedFrameBuffer,
        tracking_queue: Queue,
        control_queue: Queue
    ):
        """
        Initialize camera process

        Args:
            camera_id: Camera identifier
            video_source: Path to video file or camera index
            config: Camera configuration dict
            shared_frame_buffer: Shared memory buffer for frames
            tracking_queue: Queue for sending tracking data to main process
            control_queue: Queue for receiving control commands
        """
        super().__init__(name=f"CameraProcess-{camera_id}", daemon=True)

        self.camera_id = camera_id
        self.video_source = video_source
        self.config = config
        self.shared_frame_buffer = shared_frame_buffer
        self.tracking_queue = tracking_queue
        self.control_queue = control_queue

        self.running = False

    def run(self):
        """Main process entry point"""
        try:
            logger.info(f"🚀 {self.camera_id}: Camera process started (PID: {mp.current_process().pid})")
            self._run_detection_loop()
        except KeyboardInterrupt:
            logger.info(f"⚠️ {self.camera_id}: Received keyboard interrupt")
        except Exception as e:
            logger.error(f"❌ {self.camera_id}: Process crashed: {e}")
            logger.error(traceback.format_exc())
        finally:
            self._cleanup()
            logger.info(f"🛑 {self.camera_id}: Camera process stopped")

    def _run_detection_loop(self):
        """Main detection loop with producer-consumer pattern"""
        import cv2

        # Initialize YOLO model
        if not YOLO_AVAILABLE:
            logger.error(f"❌ {self.camera_id}: YOLO not available")
            return

        model = YOLO('yolo26n-pose.pt')

        # Force GPU usage if available
        import torch
        if torch.cuda.is_available():
            model.to('cuda')
            logger.info(f"✅ {self.camera_id}: YOLO model loaded on GPU (CUDA)")
        else:
            logger.warning(f"⚠️ {self.camera_id}: YOLO model loaded on CPU (CUDA not available)")
            logger.info(f"✅ {self.camera_id}: YOLO model loaded")

        # Initialize brain modules
        aggregator = None
        scorer = None
        if BRAIN_AVAILABLE:
            aggregator = FeatureAggregator(window_seconds=4.0)
            scorer = RuleBasedScorer()
            logger.info(f"✅ {self.camera_id}: Brain modules initialized")

        # Initialize face/emotion services (if enabled)
        face_service = None
        if self.config.get('face_recognition_enabled', False) and FACE_RECOGNITION_AVAILABLE:
            try:
                face_service = FaceRecognitionService()
                logger.info(f"✅ {self.camera_id}: Face recognition enabled")
            except Exception as e:
                logger.warning(f"⚠️ {self.camera_id}: Face recognition failed: {e}")

        emotion_service = None
        if self.config.get('emotion_detection_enabled', False) and EMOTION_RECOGNITION_AVAILABLE:
            try:
                emotion_service = EmotionRecognitionService()
                logger.info(f"✅ {self.camera_id}: Emotion detection enabled")
            except Exception as e:
                logger.warning(f"⚠️ {self.camera_id}: Emotion detection failed: {e}")

        advanced_features_extractor = None
        if self.config.get('advanced_features_enabled', False) and ADVANCED_FEATURES_AVAILABLE:
            try:
                advanced_features_extractor = AdvancedFeatureExtractor()
                logger.info(f"✅ {self.camera_id}: Advanced features enabled")
            except Exception as e:
                logger.warning(f"⚠️ {self.camera_id}: Advanced features failed: {e}")

        # Load calibration
        calibration_manager = CalibrationManager()
        platform_poly = calibration_manager.load_calibration(self.camera_id)
        if platform_poly:
            logger.info(f"✅ {self.camera_id}: Platform calibrated")

        # Setup producer-consumer frame reading (larger queue for smoother playback)
        frame_queue = queue.Queue(maxsize=10)
        frame_reader = FrameReader(self.video_source, frame_queue, self.camera_id, max_queue_size=10)
        frame_reader.start()

        # Tracking state
        track_history = defaultdict(lambda: deque())
        last_positions = {}
        first_seen = {}
        recent_alerts = {}

        # Performance tracking
        processing_times = deque(maxlen=100)
        last_perf_log = time.time()
        frame_count = 0

        self.running = True
        logger.info(f"✅ {self.camera_id}: Detection loop started")

        # Frame skipping for performance
        process_every_n_frames = 2  # Process every 2nd frame (skip 1)
        last_annotated_frame = None

        # Main processing loop
        while self.running:
            loop_start = time.time()

            # Check for control commands (non-blocking)
            try:
                cmd = self.control_queue.get_nowait()
                if cmd == "stop":
                    logger.info(f"🛑 {self.camera_id}: Received stop command")
                    break
            except queue.Empty:
                pass

            # Get frame from reader (with timeout)
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_count += 1
            ts = time.time()

            # Skip frames for performance (process every Nth frame)
            if frame_count % process_every_n_frames != 0:
                # Skip YOLO processing, but still write last frame to shared memory
                if last_annotated_frame is not None:
                    self.shared_frame_buffer.write_frame(last_annotated_frame)
                continue

            # Run YOLO detection (only on processed frames)
            results = model.track(frame, persist=True, conf=0.3, verbose=False)

            # Extract tracking data with features and alerts
            try:
                tracking_data = extract_tracking_data_optimized(
                    result=results[0],
                    camera_id=self.camera_id,
                    track_history=track_history,
                    last_positions=last_positions,
                    first_seen=first_seen,
                    platform_poly=platform_poly,
                    aggregator=aggregator,
                    scorer=scorer,
                    recent_alerts=recent_alerts,
                    ts=ts,
                    frame=frame,
                    face_service=face_service,
                    emotion_service=emotion_service,
                    advanced_features_extractor=advanced_features_extractor,
                    frame_count=frame_count,
                    config=self.config
                )

                # Send tracking data via queue (non-blocking)
                try:
                    self.tracking_queue.put_nowait(tracking_data)
                except queue.Full:
                    logger.debug(f"{self.camera_id}: Tracking queue full, dropping data")

            except Exception as e:
                logger.debug(f"Tracking data extraction error: {e}")

            # Annotate frame (lightweight)
            try:
                annotated_frame = results[0].plot()

                # Draw platform polygon
                if platform_poly:
                    cv2.polylines(annotated_frame, [np.array(platform_poly)], True, (0, 255, 0), 2)

                # Write to shared memory (non-blocking)
                self.shared_frame_buffer.write_frame(annotated_frame)
                last_annotated_frame = annotated_frame

            except Exception as e:
                logger.debug(f"Frame annotation error: {e}")

            # Track performance
            processing_time = time.time() - loop_start
            processing_times.append(processing_time)

            # Log performance every 5 seconds
            if time.time() - last_perf_log > 5.0 and processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                fps = 1 / avg_time if avg_time > 0 else 0
                logger.info(f"📊 {self.camera_id}: FPS={fps:.1f}, Avg={avg_time*1000:.1f}ms")
                last_perf_log = time.time()

        # Stop frame reader
        frame_reader.stop()

    def _cleanup(self):
        """Cleanup resources"""
        self.running = False
        logger.info(f"🧹 {self.camera_id}: Cleaning up resources")


def start_camera_process(camera_id: str, video_source: str, config: dict, shared_frame_buffer: SharedFrameBuffer) -> tuple:
    """
    Start a camera process

    Returns:
        (process, tracking_queue, control_queue)
    """
    tracking_queue = Queue(maxsize=100)
    control_queue = Queue(maxsize=10)

    process = CameraProcess(
        camera_id=camera_id,
        video_source=video_source,
        config=config,
        shared_frame_buffer=shared_frame_buffer,
        tracking_queue=tracking_queue,
        control_queue=control_queue
    )

    process.start()
    logger.info(f"✅ Started camera process: {camera_id} (PID: {process.pid})")

    return process, tracking_queue, control_queue
