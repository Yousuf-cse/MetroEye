"""
Feature Aggregator - Sliding Window Logic
==========================================

PURPOSE:
This module aggregates per-frame features into time windows (3-5 seconds).

WHY?
- Individual frames are noisy (person might move randomly)
- Suspicious behavior happens over TIME (pacing, dwelling, approaching)
- We need to look at PATTERNS, not single moments

WHAT IT DOES:
1. Maintains a sliding window of features for each tracked person
2. Computes aggregate statistics (mean, max, std) over the window
3. Detects temporal patterns (direction changes, dwell time)

LEARNING CONCEPTS:
- Sliding Window: Like a moving average, keeps last N seconds of data
- Temporal Aggregation: Summarizing time-series data
- Deque: Efficient data structure for sliding windows (fast add/remove from both ends)
"""

from collections import defaultdict, deque
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

# Import advanced feature extractor
try:
    from brain.advanced_features import AdvancedFeatureExtractor
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False


class FeatureAggregator:
    """
    Aggregates per-frame features into temporal windows.

    Example usage:
        aggregator = FeatureAggregator(window_seconds=4.0)

        # For each frame:
        aggregator.add_frame_features(
            track_id=3,
            timestamp=current_time,
            features={'torso_angle': 82, 'speed': 145, ...}
        )

        # Get aggregated features:
        agg_features = aggregator.get_aggregated_features(track_id=3)
        # Returns: {'mean_torso_angle': 82.5, 'max_speed': 350, ...}
    """

    def __init__(self, window_seconds: float = 4.0):
        """
        Initialize the feature aggregator.

        Args:
            window_seconds: Length of sliding window in seconds
                          - Too short: Noisy, miss long-term patterns
                          - Too long: Delayed detection, miss quick events
                          - 3-5s is good balance for human behavior
        """
        self.window_seconds = window_seconds

        # Store sliding windows for each tracked person
        # defaultdict: Automatically creates empty deque for new track_ids
        # deque: Double-ended queue, fast O(1) append/pop from both ends
        self.track_windows = defaultdict(lambda: deque())

        # Track when we first saw each person (for dwell time calculation)
        self.first_seen = {}

        # Initialize advanced feature extractor
        if ADVANCED_FEATURES_AVAILABLE:
            self.advanced_extractor = AdvancedFeatureExtractor()
            print(f"✓ FeatureAggregator initialized (window={window_seconds}s, advanced features enabled)")
        else:
            self.advanced_extractor = None
            print(f"✓ FeatureAggregator initialized (window={window_seconds}s)")


    def add_frame_features(self, track_id: int, timestamp: float, features: Dict):
        """
        Add features from a single frame to the sliding window.

        Args:
            track_id: Unique ID for this person (from tracker)
            timestamp: Current time (seconds, e.g., time.time())
            features: Dictionary with per-frame features:
                - 'torso_angle': Angle from vertical (degrees)
                - 'speed': Movement speed (pixels/second)
                - 'dist_to_edge': Distance to platform edge (pixels)
                - 'center': (x, y) bbox center coordinates
                - 'keypoints': YOLO pose keypoints array (17, 2) [OPTIONAL, for advanced features]
                - 'bbox': Bounding box (x1, y1, x2, y2) [OPTIONAL, for isolation metric]

        LEARNING: This is called EVERY frame (30 times per second).
        We store all recent frames in a window, then aggregate them.
        """
        # Add this frame's data to the window
        self.track_windows[track_id].append((timestamp, features))

        # Record first appearance (for dwell time)
        if track_id not in self.first_seen:
            self.first_seen[track_id] = timestamp

        # Remove old frames outside the window
        # Example: If window=4s and current_time=10s, remove frames before 6s
        while self.track_windows[track_id]:
            oldest_ts = self.track_windows[track_id][0][0]  # Timestamp of first frame

            if timestamp - oldest_ts > self.window_seconds:
                # Frame is too old, remove it
                self.track_windows[track_id].popleft()  # O(1) operation with deque
            else:
                break  # Rest of frames are recent enough


    def get_aggregated_features(self, track_id: int) -> Optional[Dict]:
        """
        Compute aggregated features over the sliding window.

        Returns:
            Dictionary with aggregated features, or None if not enough data

        LEARNING: This is where we convert 100+ frames into a single feature vector.
        Instead of analyzing each frame, we look at PATTERNS over time.
        """
        if track_id not in self.track_windows:
            return None

        window = self.track_windows[track_id]

        # Need at least 3 frames to compute meaningful statistics
        if len(window) < 3:
            return None

        # Extract lists of values for each feature
        # LEARNING: We're "unpacking" the window into separate lists per feature
        torso_angles = []
        speeds = []
        dist_to_edges = []
        centers = []
        timestamps = []  # NEW: for acceleration computation
        head_pitches = []  # NEW: for phone detection
        keypoints_list = []  # NEW: for advanced features
        bboxes = []  # NEW: for isolation metric

        for ts, feat in window:
            timestamps.append(ts)  # Track timestamps for acceleration

            # Only add if feature exists (some frames might be missing data)
            if feat.get('torso_angle') is not None:
                torso_angles.append(feat['torso_angle'])
            if feat.get('speed') is not None:
                speeds.append(feat['speed'])
            if feat.get('dist_to_edge') is not None:
                dist_to_edges.append(feat['dist_to_edge'])
            if feat.get('center') is not None:
                centers.append(feat['center'])
            if feat.get('head_pitch') is not None:  # NEW
                head_pitches.append(feat['head_pitch'])
            if feat.get('keypoints') is not None:  # NEW: for advanced features
                keypoints_list.append(feat['keypoints'])
            if feat.get('bbox') is not None:  # NEW: for isolation
                bboxes.append(feat['bbox'])

        # Compute aggregate statistics
        # LEARNING: We use mean, max, std to capture different aspects:
        # - mean: Overall behavior (is person usually close to edge?)
        # - max: Peak behavior (did person get very close at any point?)
        # - std: Variability (is person stable or erratic?)

        agg = {
            'track_id': track_id,
            'window_start': window[0][0],
            'window_duration': window[-1][0] - window[0][0],
            'num_frames': len(window),

            # Torso angle statistics
            'mean_torso_angle': float(np.mean(torso_angles)) if torso_angles else None,
            'max_torso_angle': float(np.max(torso_angles)) if torso_angles else None,
            'std_torso_angle': float(np.std(torso_angles)) if torso_angles else None,

            # Speed statistics
            'mean_speed': float(np.mean(speeds)) if speeds else None,
            'max_speed': float(np.max(speeds)) if speeds else None,

            # Distance to edge
            'min_dist_to_edge': float(np.min(dist_to_edges)) if dist_to_edges else None,
            'mean_dist_to_edge': float(np.mean(dist_to_edges)) if dist_to_edges else None,

            # Dwell time near edge
            'dwell_time_near_edge': self._compute_dwell_time(dist_to_edges, threshold=150),

            # Direction changes (temporal pattern detection)
            'direction_changes': self._compute_direction_changes(centers),

            # NEW: Head pitch features (phone usage detection)
            'mean_head_pitch': float(np.mean(head_pitches)) if head_pitches else None,
            'max_head_pitch': float(np.max(head_pitches)) if head_pitches else None,
            'time_looking_down': self._compute_time_looking_down(head_pitches, threshold=30),

            # NEW: Acceleration features (rushing behavior detection)
            'max_acceleration': self._compute_max_acceleration(speeds, timestamps),
            'acceleration_spikes': self._compute_acceleration_spikes(speeds, timestamps, threshold=200),
        }

        # ===== ADVANCED FEATURES (Critical Distress Indicators) =====
        if self.advanced_extractor and len(keypoints_list) > 0:
            # Get most recent keypoints for micro-postural analysis
            latest_keypoints = keypoints_list[-1] if keypoints_list else None

            if latest_keypoints is not None and len(latest_keypoints) >= 17:
                # 1. Shoulder Hunch Index (tension marker)
                agg['shoulder_hunch_index'] = self.advanced_extractor.compute_shoulder_hunch(latest_keypoints)

                # 2. Closed Body Posture (psychological withdrawal)
                agg['closed_body_posture'] = self.advanced_extractor.detect_closed_body_posture(latest_keypoints)

                # 3. Hand-to-Face Proximity (distress/anxiety)
                agg['hand_to_face_distance'] = self.advanced_extractor.compute_hand_to_face_proximity(latest_keypoints)

                # 4. Head Yaw (track fixation)
                agg['head_yaw_angle'] = self.advanced_extractor.compute_head_yaw(latest_keypoints)

            # 5. Edge Transgression Count (hesitation loop - CRITICAL!)
            agg['edge_transgression_count'] = self.advanced_extractor.compute_edge_transgression_count(
                dist_to_edges, threshold=50.0
            )

            # 6. Weight Shifting Variance (nervous fidgeting)
            if len(keypoints_list) >= 5:
                # Extract hip positions from keypoints
                hip_positions = []
                for kp in keypoints_list:
                    if kp is not None and len(kp) >= 17:
                        left_hip_y = kp[11][1] if len(kp[11]) >= 2 else 0
                        right_hip_y = kp[12][1] if len(kp[12]) >= 2 else 0
                        hip_positions.append((left_hip_y, right_hip_y))

                agg['weight_shifting_variance'] = self.advanced_extractor.compute_weight_shifting_variance(
                    hip_positions, centers
                )
        else:
            # Advanced features not available or no keypoints
            agg['shoulder_hunch_index'] = None
            agg['closed_body_posture'] = False
            agg['hand_to_face_distance'] = None
            agg['head_yaw_angle'] = None
            agg['edge_transgression_count'] = 0
            agg['weight_shifting_variance'] = None

        return agg


    def _compute_dwell_time(self, distances: List[float], threshold: float) -> float:
        """
        Compute how long person stayed within threshold distance.

        LEARNING: Dwell time is a KEY indicator of suspicious behavior.
        Normal: Person walks past the edge (low dwell time)
        Suspicious: Person stands near edge for long time (high dwell time)

        Args:
            distances: List of distances to edge (pixels)
            threshold: Maximum distance to count as "near edge" (pixels)

        Returns:
            Time in seconds spent near edge
        """
        if not distances:
            return 0.0

        # Count frames where distance < threshold
        near_edge_frames = sum(1 for d in distances if d < threshold)

        # Convert frames to seconds (assuming 30 fps)
        # LEARNING: This is an approximation. In production, you'd track actual frame timestamps.
        fps = 30.0
        dwell_time_seconds = near_edge_frames / fps

        return float(dwell_time_seconds)


    def _compute_direction_changes(self, centers: List[Tuple[float, float]]) -> int:
        """
        Count number of times person changed direction significantly.

        LEARNING: Direction changes detect PACING behavior.
        Normal: Person walks in straight line (0-2 direction changes)
        Suspicious: Person paces back and forth (5+ direction changes)

        How it works:
        1. Compute velocity vectors between consecutive centers
        2. Compute angle between consecutive velocity vectors
        3. Count angles > 90 degrees (sharp turns)

        Args:
            centers: List of (x, y) bbox centers

        Returns:
            Number of significant direction changes
        """
        if len(centers) < 3:
            return 0

        direction_changes = 0
        prev_velocity = None

        # Compute velocity vectors
        for i in range(len(centers) - 1):
            cx1, cy1 = centers[i]
            cx2, cy2 = centers[i + 1]

            # Velocity = displacement vector
            velocity = (cx2 - cx1, cy2 - cy1)

            if prev_velocity is not None:
                # Compute angle between this velocity and previous velocity
                # Using dot product: v1 · v2 = |v1||v2|cos(θ)

                dot = prev_velocity[0] * velocity[0] + prev_velocity[1] * velocity[1]
                mag1 = np.sqrt(prev_velocity[0]**2 + prev_velocity[1]**2)
                mag2 = np.sqrt(velocity[0]**2 + velocity[1]**2)

                if mag1 > 0 and mag2 > 0:
                    # Compute angle in degrees
                    cos_angle = dot / (mag1 * mag2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
                    angle_deg = np.degrees(np.arccos(cos_angle))

                    # Count as direction change if angle > 90 degrees
                    if angle_deg > 90:
                        direction_changes += 1

            prev_velocity = velocity

        return direction_changes


    def _compute_time_looking_down(self, head_pitches: List[float], threshold: float = 30) -> float:
        """
        Compute how long person was looking down (possible phone usage).

        NEW FEATURE: Phone usage detection via head pose.

        LEARNING: People looking at phones typically have head tilted down 25-35°.
        This is a PROXY for phone usage - not perfect, but useful!

        Why this matters:
        - Distracted person near edge = HIGH RISK
        - LLM can reason: "Person looking at phone + near edge + train arriving = CRITICAL"

        Args:
            head_pitches: List of head pitch angles (degrees)
                         +30° = looking down, 0° = straight, -30° = looking up
            threshold: Angle considered "looking down" (default: 30°)

        Returns:
            Time in seconds spent looking down
        """
        if not head_pitches:
            return 0.0

        # Count frames where head pitch > threshold (looking down)
        looking_down_frames = sum(1 for pitch in head_pitches if pitch > threshold)

        # Convert to seconds (assuming 30 fps)
        fps = 30.0
        time_looking_down = looking_down_frames / fps

        return float(time_looking_down)


    def _compute_max_acceleration(self, speeds: List[float], timestamps: List[float]) -> float:
        """
        Compute maximum acceleration (sudden speed changes).

        NEW FEATURE: Rushing behavior detection.

        LEARNING: Acceleration = rate of change of speed.
        - Slow acceleration: Person walking normally
        - High acceleration: Person suddenly rushing toward edge

        Formula: acceleration = (speed[i+1] - speed[i]) / (time[i+1] - time[i])
        Units: pixels per second² (px/s²)

        Args:
            speeds: List of speeds (px/s)
            timestamps: List of corresponding timestamps (seconds)

        Returns:
            Maximum acceleration observed (px/s²)
        """
        if len(speeds) < 2 or len(timestamps) < 2:
            return 0.0

        accelerations = []

        for i in range(len(speeds) - 1):
            dt = timestamps[i + 1] - timestamps[i]

            if dt > 0:  # Avoid division by zero
                # Compute acceleration
                accel = (speeds[i + 1] - speeds[i]) / dt
                accelerations.append(accel)

        if not accelerations:
            return 0.0

        # Return maximum absolute acceleration
        # LEARNING: We use abs() because deceleration is also interesting
        # (person suddenly stops near edge = suspicious)
        max_accel = max(abs(a) for a in accelerations)

        return float(max_accel)


    def _compute_acceleration_spikes(self, speeds: List[float], timestamps: List[float], threshold: float = 200) -> int:
        """
        Count number of sudden acceleration spikes.

        NEW FEATURE: Erratic movement detection.

        LEARNING: Multiple acceleration spikes = erratic, unpredictable behavior.
        - Normal: 0-1 spikes (smooth movement)
        - Suspicious: 3+ spikes (agitated, erratic)

        Args:
            speeds: List of speeds (px/s)
            timestamps: List of corresponding timestamps
            threshold: Acceleration considered "spike" (default: 200 px/s²)

        Returns:
            Number of acceleration spikes above threshold
        """
        if len(speeds) < 2 or len(timestamps) < 2:
            return 0

        spikes = 0

        for i in range(len(speeds) - 1):
            dt = timestamps[i + 1] - timestamps[i]

            if dt > 0:
                accel = abs((speeds[i + 1] - speeds[i]) / dt)

                if accel > threshold:
                    spikes += 1

        return spikes


    def cleanup_old_tracks(self, current_timestamp: float, max_age: float = 30.0):
        """
        Remove tracks not seen in last N seconds to prevent memory leak.

        LEARNING: In long-running systems, you must clean up old data.
        Without this, memory usage grows forever as more people are tracked.

        Args:
            current_timestamp: Current time (seconds)
            max_age: Remove tracks not seen for this many seconds
        """
        to_remove = []

        for track_id, window in self.track_windows.items():
            if window:
                last_seen = window[-1][0]  # Timestamp of most recent frame

                if current_timestamp - last_seen > max_age:
                    to_remove.append(track_id)

        # Remove old tracks
        for track_id in to_remove:
            del self.track_windows[track_id]
            if track_id in self.first_seen:
                del self.first_seen[track_id]

        if to_remove:
            print(f"✓ Cleaned up {len(to_remove)} old tracks")


# EDUCATIONAL EXAMPLE
if __name__ == "__main__":
    """
    Run this file directly to see a demo of feature aggregation.

    Command: python feature_aggregator.py
    """
    print("=== Feature Aggregator Demo ===\n")

    # Create aggregator
    aggregator = FeatureAggregator(window_seconds=4.0)

    # Simulate 15 frames (0.5 seconds at 30fps)
    print("Simulating 15 frames of movement...\n")

    base_time = time.time()
    for i in range(15):
        # Simulate person moving toward edge
        aggregator.add_frame_features(
            track_id=3,
            timestamp=base_time + i * 0.033,  # 30 fps = 0.033s per frame
            features={
                'torso_angle': 85.0 - i * 0.5,  # Leaning more over time
                'speed': 100.0 + i * 15,         # Accelerating
                'dist_to_edge': 150.0 - i * 8,  # Getting closer to edge
                'center': (100 + i * 5, 300)     # Moving right
            }
        )

    # Get aggregated features
    result = aggregator.get_aggregated_features(track_id=3)

    if result:
        print("✓ Aggregated Features:")
        print(f"  - Window duration: {result['window_duration']:.2f}s")
        print(f"  - Frames: {result['num_frames']}")
        print(f"  - Mean torso angle: {result['mean_torso_angle']:.1f}°")
        print(f"  - Max speed: {result['max_speed']:.1f} px/s")
        print(f"  - Min distance to edge: {result['min_dist_to_edge']:.1f} px")
        print(f"  - Dwell time near edge: {result['dwell_time_near_edge']:.2f}s")
        print(f"  - Direction changes: {result['direction_changes']}")

        print("\n✓ Feature aggregation works!")
    else:
        print("✗ Not enough data to aggregate")
