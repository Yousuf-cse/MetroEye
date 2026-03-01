"""
Advanced Behavioral Feature Extraction
=======================================

PURPOSE:
Implements micro-postural, spatiotemporal, and contextual features
for critical distress detection in metro platform surveillance.

RESEARCH BASIS:
These features are derived from psychological research on behavioral
indicators of severe distress and suicidal behavior in transit systems.

CRITICAL USE CASE:
This module is designed for suicide prevention in metro platforms.
Features detect:
- Tension & distress markers (micro-postures)
- Hesitation & fixation patterns
- Social isolation indicators
- Environmental context

KEYPOINT REFERENCE (YOLO v11 Pose - 17 keypoints):
0: Nose
1: Left Eye
2: Right Eye
3: Left Ear
4: Right Ear
5: Left Shoulder
6: Right Shoulder
7: Left Elbow
8: Right Elbow
9: Left Wrist
10: Right Wrist
11: Left Hip
12: Right Hip
13: Left Knee
14: Right Knee
15: Left Ankle
16: Right Ankle
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class AdvancedFeatureExtractor:
    """
    Extracts advanced psychological and behavioral features from pose keypoints.

    These features go beyond basic movement and posture to detect
    subtle psychological distress indicators.
    """

    def __init__(self):
        """Initialize the advanced feature extractor."""
        # Keypoint indices (YOLO v11 format)
        self.NOSE = 0
        self.LEFT_EYE = 1
        self.RIGHT_EYE = 2
        self.LEFT_EAR = 3
        self.RIGHT_EAR = 4
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16

        print("✓ AdvancedFeatureExtractor initialized")


    # ========================================================================
    # 1. MICRO-POSTURAL FEATURES (Tension & Distress Markers)
    # ========================================================================

    def compute_shoulder_hunch(self, keypoints: np.ndarray) -> Optional[float]:
        """
        Detect shoulder hunching - a key indicator of stress/distress.

        PSYCHOLOGY:
        High stress, fear, or "giving up" state causes trapezius muscles
        to contract, pulling shoulders up toward ears, or deep slouching.

        FORMULA:
        Hunch_Index = (Y_shoulder - Y_ear) / Torso_Length

        Negative value = shoulders higher than ears (hunched up)
        Positive value = shoulders much lower than ears (slouched)

        Args:
            keypoints: YOLO pose keypoints (17, 2) array [x, y]

        Returns:
            Normalized hunch index (-1.0 to 1.0), or None if keypoints missing
        """
        try:
            # Get keypoints with confidence check
            left_ear = keypoints[self.LEFT_EAR]
            right_ear = keypoints[self.RIGHT_EAR]
            left_shoulder = keypoints[self.LEFT_SHOULDER]
            right_shoulder = keypoints[self.RIGHT_SHOULDER]
            left_hip = keypoints[self.LEFT_HIP]
            right_hip = keypoints[self.RIGHT_HIP]

            # Check if keypoints are valid (not [0, 0])
            if not all(self._is_valid_keypoint(kp) for kp in
                      [left_ear, right_ear, left_shoulder, right_shoulder, left_hip, right_hip]):
                return None

            # Calculate average ear Y and shoulder Y
            avg_ear_y = (left_ear[1] + right_ear[1]) / 2.0
            avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0

            # Calculate torso length (shoulder to hip distance)
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2.0
            torso_length = abs(avg_hip_y - avg_shoulder_y)

            if torso_length < 10:  # Avoid division by very small numbers
                return None

            # Compute normalized hunch index
            # Note: In image coordinates, Y increases downward
            # So shoulder_y > ear_y means slouched, shoulder_y < ear_y means hunched
            hunch_index = (avg_shoulder_y - avg_ear_y) / torso_length

            # Clip to reasonable range
            hunch_index = np.clip(hunch_index, -1.0, 1.0)

            return float(hunch_index)

        except Exception as e:
            return None


    def detect_closed_body_posture(self, keypoints: np.ndarray) -> bool:
        """
        Detect closed body posture (arms crossing, self-hugging).

        PSYCHOLOGY:
        Subconscious attempt to protect vital organs and self-soothe.
        Signifies psychological withdrawal or isolation.

        LOGIC:
        Detect when wrists move inward and cross over the torso midline.

        Args:
            keypoints: YOLO pose keypoints (17, 2) array

        Returns:
            True if closed posture detected, False otherwise
        """
        try:
            left_wrist = keypoints[self.LEFT_WRIST]
            right_wrist = keypoints[self.RIGHT_WRIST]
            left_shoulder = keypoints[self.LEFT_SHOULDER]
            right_shoulder = keypoints[self.RIGHT_SHOULDER]
            left_elbow = keypoints[self.LEFT_ELBOW]
            right_elbow = keypoints[self.RIGHT_ELBOW]

            if not all(self._is_valid_keypoint(kp) for kp in
                      [left_wrist, right_wrist, left_shoulder, right_shoulder]):
                return False

            # Calculate torso midline X coordinate
            torso_midline_x = (left_shoulder[0] + right_shoulder[0]) / 2.0

            # Calculate shoulder width for reference
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

            if shoulder_width < 10:
                return False

            # Check if left wrist has crossed to the right side of midline
            left_wrist_crossed = left_wrist[0] > torso_midline_x

            # Check if right wrist has crossed to the left side of midline
            right_wrist_crossed = right_wrist[0] < torso_midline_x

            # Additional check: wrists should be relatively close to torso
            # (not reaching out far)
            left_wrist_close = abs(left_wrist[0] - torso_midline_x) < shoulder_width
            right_wrist_close = abs(right_wrist[0] - torso_midline_x) < shoulder_width

            # Closed posture if both wrists cross midline OR
            # both wrists are very close to torso center
            closed_posture = (left_wrist_crossed and right_wrist_crossed) or \
                           (left_wrist_close and right_wrist_close and
                            abs(left_wrist[0] - right_wrist[0]) < shoulder_width * 0.5)

            return closed_posture

        except Exception as e:
            return False


    def compute_hand_to_face_proximity(self, keypoints: np.ndarray) -> float:
        """
        Measure proximity of hands to face/head.

        PSYCHOLOGY:
        Hands running through hair, covering face, or rubbing neck are
        universal signs of severe distress, anxiety, or erratic pacing.

        LOGIC:
        Calculate minimum Euclidean distance between wrists and face keypoints.

        Args:
            keypoints: YOLO pose keypoints (17, 2) array

        Returns:
            Minimum distance from hands to face (pixels)
        """
        try:
            # Face keypoints
            nose = keypoints[self.NOSE]
            left_eye = keypoints[self.LEFT_EYE]
            right_eye = keypoints[self.RIGHT_EYE]
            left_ear = keypoints[self.LEFT_EAR]
            right_ear = keypoints[self.RIGHT_EAR]

            # Hand keypoints
            left_wrist = keypoints[self.LEFT_WRIST]
            right_wrist = keypoints[self.RIGHT_WRIST]

            # Collect valid face keypoints
            face_points = [nose, left_eye, right_eye, left_ear, right_ear]
            face_points = [fp for fp in face_points if self._is_valid_keypoint(fp)]

            if not face_points:
                return float('inf')

            # Calculate distances
            min_distance = float('inf')

            for face_point in face_points:
                if self._is_valid_keypoint(left_wrist):
                    dist_left = np.linalg.norm(left_wrist - face_point)
                    min_distance = min(min_distance, dist_left)

                if self._is_valid_keypoint(right_wrist):
                    dist_right = np.linalg.norm(right_wrist - face_point)
                    min_distance = min(min_distance, dist_right)

            return float(min_distance)

        except Exception as e:
            return float('inf')


    # ========================================================================
    # 2. ADVANCED SPATIOTEMPORAL FEATURES (Hesitation & Fixation)
    # ========================================================================

    def compute_edge_transgression_count(self, dist_to_edge_history: List[float],
                                         threshold: float = 50.0) -> int:
        """
        Count platform edge transgression cycles (approach and retreat).

        PSYCHOLOGY:
        The "hesitation loop" - ambivalence of survival instinct.
        Person approaches absolute edge, looks, steps back, repeats.
        This is one of the STRONGEST indicators of suicidal intent.

        LOGIC:
        Count number of times dist_to_edge drops below threshold and
        then increases again (complete cycle).

        Args:
            dist_to_edge_history: List of distance measurements over time
            threshold: Critical distance threshold (pixels)

        Returns:
            Number of complete transgression cycles
        """
        if len(dist_to_edge_history) < 3:
            return 0

        transgressions = 0
        in_critical_zone = False

        for dist in dist_to_edge_history:
            if dist < threshold and not in_critical_zone:
                # Entering critical zone
                in_critical_zone = True
            elif dist >= threshold and in_critical_zone:
                # Exiting critical zone - complete cycle
                transgressions += 1
                in_critical_zone = False

        return transgressions


    def compute_head_yaw(self, keypoints: np.ndarray) -> Optional[float]:
        """
        Estimate head yaw angle (horizontal head rotation).

        PSYCHOLOGY:
        Fixating on tunnel entrance where train will emerge.
        Leaning over to look is a massive red flag.

        LOGIC:
        Use horizontal distance ratio between nose and ears.

        Args:
            keypoints: YOLO pose keypoints (17, 2) array

        Returns:
            Estimated yaw angle in degrees (-90 to +90)
            0 = facing forward, positive = turned right, negative = turned left
        """
        try:
            nose = keypoints[self.NOSE]
            left_ear = keypoints[self.LEFT_EAR]
            right_ear = keypoints[self.RIGHT_EAR]

            if not all(self._is_valid_keypoint(kp) for kp in [nose, left_ear, right_ear]):
                return None

            # Calculate horizontal distances from nose to each ear
            dist_to_left_ear = abs(nose[0] - left_ear[0])
            dist_to_right_ear = abs(nose[0] - right_ear[0])

            # Calculate face width
            face_width = abs(left_ear[0] - right_ear[0])

            if face_width < 5:  # Too close to calculate reliable angle
                return None

            # Estimate yaw using ratio
            # When looking straight: both distances equal
            # When turning right: right ear closer
            # When turning left: left ear closer

            ratio = (dist_to_left_ear - dist_to_right_ear) / face_width

            # Convert ratio to approximate angle (-90 to +90 degrees)
            # This is an approximation; exact mapping requires calibration
            yaw_angle = ratio * 60.0  # Scale factor
            yaw_angle = np.clip(yaw_angle, -90.0, 90.0)

            return float(yaw_angle)

        except Exception as e:
            return None


    def compute_weight_shifting_variance(self, hip_positions: List[Tuple[float, float]],
                                         bbox_centers: List[Tuple[float, float]]) -> Optional[float]:
        """
        Detect nervous weight shifting (rocking/fidgeting).

        PSYCHOLOGY:
        Nervous energy manifests as rocking back and forth or shifting
        weight from foot to foot without actually walking.

        LOGIC:
        Calculate variance of hip positions while bbox_center remains
        relatively static.

        Args:
            hip_positions: List of (left_hip_y, right_hip_y) tuples
            bbox_centers: List of (center_x, center_y) tuples

        Returns:
            Variance metric (high = nervous shifting)
        """
        if len(hip_positions) < 5 or len(bbox_centers) < 5:
            return None

        try:
            # Calculate variance of bbox center (should be low if person is stationary)
            bbox_x_coords = [center[0] for center in bbox_centers]
            bbox_x_variance = np.var(bbox_x_coords)

            # Calculate variance of hip lateral movement
            left_hip_y_coords = [pos[0] for pos in hip_positions]
            right_hip_y_coords = [pos[1] for pos in hip_positions]

            hip_variance = (np.var(left_hip_y_coords) + np.var(right_hip_y_coords)) / 2.0

            # Weight shifting: high hip variance while bbox variance is low
            # Normalize by bbox variance to detect "stationary fidgeting"
            if bbox_x_variance < 100:  # Person is relatively stationary
                # High hip variance indicates weight shifting
                return float(hip_variance)
            else:
                # Person is moving around - hip variance is expected
                return 0.0

        except Exception as e:
            return None


    # ========================================================================
    # 3. CONTEXTUAL & ENVIRONMENTAL FEATURES (Isolation)
    # ========================================================================

    def compute_social_isolation_metric(self, person_bbox_center: Tuple[float, float],
                                         all_bbox_centers: List[Tuple[float, float]]) -> float:
        """
        Calculate distance to nearest other person.

        PSYCHOLOGY:
        Individuals intending self-harm tend to isolate at extreme
        ends of platform, staying far from crowds to avoid intervention.

        LOGIC:
        Calculate distance from subject to nearest tracked person.

        Args:
            person_bbox_center: (x, y) center of subject
            all_bbox_centers: List of (x, y) centers of all tracked people

        Returns:
            Distance to nearest person in pixels
        """
        if len(all_bbox_centers) <= 1:
            # Person is alone on platform
            return float('inf')

        min_distance = float('inf')

        for other_center in all_bbox_centers:
            # Skip self
            if np.allclose(person_bbox_center, other_center, atol=5.0):
                continue

            distance = np.linalg.norm(
                np.array(person_bbox_center) - np.array(other_center)
            )

            min_distance = min(min_distance, distance)

        return float(min_distance)


    def detect_object_separation(self, person_keypoints: np.ndarray,
                                  person_bbox: Tuple[float, float, float, float],
                                  other_bboxes: List[Tuple[float, float, float, float]],
                                  prev_associations: Dict) -> bool:
        """
        Detect bag drop or asset abandonment.

        PSYCHOLOGY:
        Discarding luggage, backpacks, or shoes near platform edge.

        LOGIC:
        Detect when a previously associated object (bag) separates
        from person and remains static near edge.

        NOTE: Requires object detection in addition to pose detection.
        This is a placeholder - full implementation needs object tracking.

        Args:
            person_keypoints: Person's pose keypoints
            person_bbox: Person's bounding box (x1, y1, x2, y2)
            other_bboxes: List of non-person object bboxes
            prev_associations: Dictionary tracking object-person associations

        Returns:
            True if object separation detected
        """
        # PLACEHOLDER: Full implementation requires:
        # 1. Object detection model (detect bags, luggage)
        # 2. Object tracking (maintain object IDs)
        # 3. Association tracking (which objects belong to which person)
        # 4. Temporal analysis (detect when object stops moving with person)

        # For now, return False (feature not implemented)
        # This can be added in future versions with full object detection
        return False


    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    def _is_valid_keypoint(self, keypoint: np.ndarray, min_confidence: float = 0.3) -> bool:
        """
        Check if keypoint is valid (detected with sufficient confidence).

        Args:
            keypoint: Keypoint array [x, y] or [x, y, confidence]
            min_confidence: Minimum confidence threshold

        Returns:
            True if keypoint is valid
        """
        if keypoint is None or len(keypoint) < 2:
            return False

        # Check if keypoint is not [0, 0] (missing detection)
        if keypoint[0] == 0 and keypoint[1] == 0:
            return False

        # If confidence is available, check it
        if len(keypoint) >= 3:
            return keypoint[2] >= min_confidence

        return True


# TESTING EXAMPLE
if __name__ == "__main__":
    """
    Test the advanced feature extractor.
    """
    print("=== Advanced Feature Extractor Demo ===\n")

    extractor = AdvancedFeatureExtractor()

    # Simulate YOLO keypoints (17, 2) - normal posture
    normal_keypoints = np.array([
        [320, 100],  # Nose
        [310, 95],   # Left Eye
        [330, 95],   # Right Eye
        [300, 100],  # Left Ear
        [340, 100],  # Right Ear
        [290, 150],  # Left Shoulder
        [350, 150],  # Right Shoulder
        [270, 200],  # Left Elbow
        [370, 200],  # Right Elbow
        [260, 250],  # Left Wrist
        [380, 250],  # Right Wrist
        [300, 300],  # Left Hip
        [340, 300],  # Right Hip
        [295, 400],  # Left Knee
        [345, 400],  # Right Knee
        [290, 500],  # Left Ankle
        [350, 500],  # Right Ankle
    ])

    print("Testing normal posture:")
    print(f"  Shoulder Hunch: {extractor.compute_shoulder_hunch(normal_keypoints):.3f}")
    print(f"  Closed Posture: {extractor.detect_closed_body_posture(normal_keypoints)}")
    print(f"  Hand-Face Distance: {extractor.compute_hand_to_face_proximity(normal_keypoints):.1f}px")
    print(f"  Head Yaw: {extractor.compute_head_yaw(normal_keypoints):.1f}°")

    print("\n✓ Advanced features extraction working!")
