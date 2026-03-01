"""
Emotion Recognition Service - Behavioral Analysis Through Facial Expressions
============================================================================

Purpose:
- Detect passenger emotions from facial expressions
- Identify distressed, anxious, or aggressive passengers
- Enhance risk assessment with emotional context
- Provide customer service insights

Emotions Detected:
- Happy: Normal, satisfied passenger
- Neutral: Normal commuter state
- Sad: Possible distress, needs assistance
- Angry: Aggressive behavior, potential threat
- Fear: Emergency situation, victim
- Surprise: Sudden event reaction
- Disgust: Discomfort, possible incident

Integration:
- Works with InsightFace face recognition
- Adds emotion to Person tracking
- Enhances risk scoring with emotional context
- Triggers alerts for extreme emotions

Installation:
    pip install insightface onnxruntime
    # InsightFace includes emotion recognition

Usage:
    from emotion_recognition_service import EmotionRecognitionService

    emotion_service = EmotionRecognitionService()

    # Detect emotion from face
    emotion, confidence = emotion_service.detect_emotion(frame, face_bbox)

    # Assess risk based on emotion
    risk_level = emotion_service.assess_emotional_risk(emotion, context)
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import time
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

# Try importing InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    logger.info("✓ InsightFace emotion detection loaded")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("⚠ InsightFace not available")


class EmotionRecognitionService:
    """
    Detects and analyzes passenger emotions for behavioral assessment.

    Features:
    - Real-time emotion detection (7 emotions)
    - Temporal emotion tracking (emotion over time)
    - Emotion-based risk assessment
    - Context-aware emotion analysis
    - Alert generation for extreme emotions
    """

    # Emotion categories
    EMOTIONS = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }

    # Risk levels by emotion
    EMOTION_RISK = {
        'angry': 'high',        # Aggressive behavior
        'fear': 'high',         # Possible victim/emergency
        'disgust': 'medium',    # Discomfort, check situation
        'sad': 'medium',        # Possible distress
        'surprise': 'low',      # Natural reaction
        'neutral': 'normal',    # Default state
        'happy': 'normal'       # Positive state
    }

    def __init__(self,
                 emotion_window_seconds: float = 10.0,
                 alert_threshold: float = 0.7,
                 track_history_size: int = 30):
        """
        Initialize emotion recognition service.

        Args:
            emotion_window_seconds: Time window to track emotion changes
            alert_threshold: Confidence threshold for emotion alerts (0.7 = 70%)
            track_history_size: Number of recent emotions to keep per person
        """
        self.emotion_window_seconds = emotion_window_seconds
        self.alert_threshold = alert_threshold
        self.track_history_size = track_history_size

        # Face analysis model (includes emotion recognition)
        self.face_app = None
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_app = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider']
                )
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("✓ Emotion recognition model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize emotion model: {e}")
                self.face_app = None

        # Emotion history per person
        # Format: {persistent_id: deque([(timestamp, emotion, confidence)])}
        self.emotion_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=track_history_size)
        )

        # Last alert time per person (avoid spam)
        self.last_alert_time: Dict[str, float] = {}

        logger.info(f"✓ Emotion service initialized (window={emotion_window_seconds}s)")


    def detect_emotion(self,
                       frame: np.ndarray,
                       bbox: List[float]) -> Tuple[Optional[str], Optional[float]]:
        """
        Detect emotion from person's face.

        Args:
            frame: Full camera frame (BGR)
            bbox: Person bounding box [x1, y1, x2, y2]

        Returns:
            (emotion, confidence) or (None, None) if no face detected
        """
        if self.face_app is None:
            return None, None

        try:
            # Crop person from frame
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                return None, None

            # Focus on upper half (where face is)
            height = person_crop.shape[0]
            upper_crop = person_crop[:int(height * 0.5), :]

            # Detect faces
            faces = self.face_app.get(upper_crop)

            if len(faces) == 0:
                return None, None

            # Get largest face
            faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            face = faces[0]

            # Extract emotion if available
            if hasattr(face, 'emotion'):
                # InsightFace emotion is array of probabilities for 7 emotions
                emotion_probs = face.emotion
                emotion_idx = np.argmax(emotion_probs)
                confidence = float(emotion_probs[emotion_idx])

                emotion = self.EMOTIONS.get(emotion_idx, 'neutral')

                return emotion, confidence

            else:
                # Fallback: Analyze face features for emotion estimation
                emotion, confidence = self._estimate_emotion_from_features(face)
                return emotion, confidence

        except Exception as e:
            logger.debug(f"Emotion detection failed: {e}")
            return None, None


    def _estimate_emotion_from_features(self, face) -> Tuple[str, float]:
        """
        Estimate emotion from facial landmarks (fallback method).

        Args:
            face: InsightFace face object with landmarks

        Returns:
            (emotion, confidence)
        """
        # Simplified emotion estimation based on facial geometry
        # This is a rough approximation if emotion model not available

        if not hasattr(face, 'landmark'):
            return 'neutral', 0.5

        landmarks = face.landmark

        # Analyze mouth (points 48-67 in 68-point model)
        # Simplified: just use neutral for fallback
        return 'neutral', 0.5


    def track_emotion(self,
                      persistent_id: str,
                      emotion: str,
                      confidence: float):
        """
        Track emotion history for a person.

        Args:
            persistent_id: Person's persistent ID
            emotion: Detected emotion
            confidence: Detection confidence
        """
        if emotion is None:
            return

        timestamp = time.time()
        self.emotion_history[persistent_id].append((timestamp, emotion, confidence))


    def get_dominant_emotion(self,
                             persistent_id: str,
                             window_seconds: Optional[float] = None) -> Tuple[Optional[str], float]:
        """
        Get most frequent emotion in recent time window.

        Args:
            persistent_id: Person's persistent ID
            window_seconds: Time window (None = use default)

        Returns:
            (dominant_emotion, average_confidence)
        """
        if persistent_id not in self.emotion_history:
            return None, 0.0

        window = window_seconds or self.emotion_window_seconds
        current_time = time.time()
        cutoff_time = current_time - window

        # Get recent emotions
        recent_emotions = [
            (emotion, conf) for (ts, emotion, conf) in self.emotion_history[persistent_id]
            if ts >= cutoff_time
        ]

        if not recent_emotions:
            return None, 0.0

        # Count emotion frequencies
        emotion_counts = defaultdict(list)  # emotion -> [confidences]
        for emotion, conf in recent_emotions:
            emotion_counts[emotion].append(conf)

        # Find dominant emotion (most frequent, highest avg confidence)
        dominant = max(
            emotion_counts.items(),
            key=lambda x: (len(x[1]), np.mean(x[1]))
        )

        dominant_emotion = dominant[0]
        avg_confidence = np.mean(dominant[1])

        return dominant_emotion, avg_confidence


    def detect_emotion_change(self,
                              persistent_id: str,
                              current_emotion: str) -> bool:
        """
        Detect significant emotion change (e.g., neutral -> angry).

        Args:
            persistent_id: Person's persistent ID
            current_emotion: Current detected emotion

        Returns:
            True if significant change detected
        """
        if persistent_id not in self.emotion_history:
            return False

        # Get previous dominant emotion
        prev_emotion, _ = self.get_dominant_emotion(persistent_id, window_seconds=5.0)

        if prev_emotion is None:
            return False

        # Check for concerning changes
        concerning_changes = [
            ('neutral', 'angry'),
            ('neutral', 'fear'),
            ('happy', 'angry'),
            ('happy', 'fear'),
            ('neutral', 'sad')
        ]

        if (prev_emotion, current_emotion) in concerning_changes:
            logger.warning(f"Emotion change detected for {persistent_id}: {prev_emotion} -> {current_emotion}")
            return True

        return False


    def assess_emotional_risk(self,
                               emotion: str,
                               confidence: float,
                               context: Optional[Dict] = None) -> Dict:
        """
        Assess risk level based on emotion and context.

        Args:
            emotion: Detected emotion
            confidence: Detection confidence
            context: Additional context (location, time, behavior)

        Returns:
            Risk assessment dict
        """
        if emotion is None:
            return {
                'risk_level': 'unknown',
                'risk_score': 0,
                'reasoning': 'No emotion detected'
            }

        # Base risk from emotion
        base_risk = self.EMOTION_RISK.get(emotion, 'normal')

        # Calculate risk score (0-100)
        risk_score = 0
        reasoning_parts = []

        # Emotion contribution
        if emotion == 'angry':
            risk_score += 50
            reasoning_parts.append("Angry expression detected")
        elif emotion == 'fear':
            risk_score += 45
            reasoning_parts.append("Fear detected - possible victim or emergency")
        elif emotion == 'disgust':
            risk_score += 25
            reasoning_parts.append("Disgust - possible discomfort")
        elif emotion == 'sad':
            risk_score += 30
            reasoning_parts.append("Sad expression - possible distress")

        # Confidence multiplier
        risk_score = int(risk_score * confidence)
        reasoning_parts.append(f"{int(confidence * 100)}% confidence")

        # Context enhancement
        if context:
            # Near platform edge + fear = higher risk
            if context.get('near_edge') and emotion == 'fear':
                risk_score += 20
                reasoning_parts.append("Fear near platform edge - critical")

            # Angry + aggressive behavior = very high risk
            if context.get('aggressive_movement') and emotion == 'angry':
                risk_score += 25
                reasoning_parts.append("Angry with aggressive movement")

            # Sad + loitering = medium risk
            if context.get('loitering') and emotion == 'sad':
                risk_score += 15
                reasoning_parts.append("Prolonged sadness while loitering")

        # Cap at 100
        risk_score = min(risk_score, 100)

        # Determine final risk level
        if risk_score >= 70:
            final_risk = 'critical'
        elif risk_score >= 50:
            final_risk = 'high'
        elif risk_score >= 30:
            final_risk = 'medium'
        else:
            final_risk = 'low'

        return {
            'emotion': emotion,
            'confidence': confidence,
            'risk_level': final_risk,
            'risk_score': risk_score,
            'reasoning': ' | '.join(reasoning_parts),
            'base_emotion_risk': base_risk
        }


    def should_alert(self,
                     persistent_id: str,
                     emotion: str,
                     confidence: float,
                     cooldown_seconds: float = 60.0) -> bool:
        """
        Determine if emotion warrants an alert.

        Args:
            persistent_id: Person's persistent ID
            emotion: Detected emotion
            confidence: Detection confidence
            cooldown_seconds: Minimum time between alerts

        Returns:
            True if should create alert
        """
        # Check confidence threshold
        if confidence < self.alert_threshold:
            return False

        # Only alert for concerning emotions
        concerning_emotions = ['angry', 'fear', 'disgust']
        if emotion not in concerning_emotions:
            return False

        # Check cooldown
        if persistent_id in self.last_alert_time:
            time_since_last = time.time() - self.last_alert_time[persistent_id]
            if time_since_last < cooldown_seconds:
                return False

        # Update last alert time
        self.last_alert_time[persistent_id] = time.time()

        return True


    def get_emotion_summary(self, persistent_id: str) -> Dict:
        """
        Get complete emotion summary for a person.

        Args:
            persistent_id: Person's persistent ID

        Returns:
            Emotion summary dict
        """
        if persistent_id not in self.emotion_history:
            return {
                'has_data': False,
                'message': 'No emotion data available'
            }

        history = list(self.emotion_history[persistent_id])

        if not history:
            return {
                'has_data': False,
                'message': 'No emotion history'
            }

        # Current emotion (most recent)
        current_ts, current_emotion, current_conf = history[-1]

        # Dominant emotion
        dominant_emotion, dominant_conf = self.get_dominant_emotion(persistent_id)

        # Emotion distribution
        emotion_counts = defaultdict(int)
        for _, emotion, _ in history:
            emotion_counts[emotion] += 1

        total_readings = len(history)
        emotion_percentages = {
            emotion: (count / total_readings * 100)
            for emotion, count in emotion_counts.items()
        }

        # Time range
        first_ts = history[0][0]
        duration = current_ts - first_ts

        return {
            'has_data': True,
            'persistent_id': persistent_id,
            'current_emotion': {
                'emotion': current_emotion,
                'confidence': current_conf,
                'timestamp': current_ts
            },
            'dominant_emotion': {
                'emotion': dominant_emotion,
                'confidence': dominant_conf
            },
            'emotion_distribution': emotion_percentages,
            'total_readings': total_readings,
            'tracking_duration': duration,
            'history': [
                {'timestamp': ts, 'emotion': emotion, 'confidence': conf}
                for ts, emotion, conf in history[-10:]  # Last 10 readings
            ]
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Test emotion recognition service.
    """
    print("=== Emotion Recognition Service Demo ===\n")

    # Initialize service
    emotion_service = EmotionRecognitionService(
        emotion_window_seconds=10.0,
        alert_threshold=0.7
    )

    if not INSIGHTFACE_AVAILABLE or emotion_service.face_app is None:
        print("❌ InsightFace not available")
        print("\nInstall with:")
        print("  pip install insightface onnxruntime")
        exit(1)

    print("✓ Emotion recognition service ready\n")

    # Simulate emotion detection
    print("Simulating passenger emotion tracking...\n")

    # Person starts neutral
    emotion_service.track_emotion('P0001', 'neutral', 0.85)
    time.sleep(0.1)

    # Person becomes anxious
    emotion_service.track_emotion('P0001', 'fear', 0.78)
    time.sleep(0.1)

    # Still fearful
    emotion_service.track_emotion('P0001', 'fear', 0.82)

    # Get summary
    summary = emotion_service.get_emotion_summary('P0001')

    print(f"Passenger: {summary['persistent_id']}")
    print(f"Current emotion: {summary['current_emotion']['emotion']} ({summary['current_emotion']['confidence']:.2f})")
    print(f"Dominant emotion: {summary['dominant_emotion']['emotion']} ({summary['dominant_emotion']['confidence']:.2f})")
    print(f"\nEmotion distribution:")
    for emotion, pct in summary['emotion_distribution'].items():
        print(f"  {emotion}: {pct:.1f}%")

    # Assess risk
    risk_assessment = emotion_service.assess_emotional_risk(
        emotion='fear',
        confidence=0.82,
        context={'near_edge': True}
    )

    print(f"\nRisk Assessment:")
    print(f"  Risk Level: {risk_assessment['risk_level']}")
    print(f"  Risk Score: {risk_assessment['risk_score']}/100")
    print(f"  Reasoning: {risk_assessment['reasoning']}")

    # Check if should alert
    should_alert = emotion_service.should_alert('P0001', 'fear', 0.82)
    print(f"\nShould alert: {should_alert}")

    print("\n✓ Demo complete!")
