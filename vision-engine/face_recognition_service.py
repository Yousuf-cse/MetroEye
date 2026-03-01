"""
Face Recognition Service - Persistent Tracking & Re-Identification
===================================================================

Purpose:
- Extract face embeddings from detected persons
- Assign persistent track IDs based on face similarity
- Re-identify people across cameras and over time
- Track passenger journeys from entry to exit

Integration with InsightFace:
- Uses InsightFace for face detection and embedding extraction
- Generates 512-dimensional embeddings per face
- Matches faces using cosine similarity

Installation:
    pip install insightface onnxruntime

Usage:
    from face_recognition_service import FaceRecognitionService

    face_service = FaceRecognitionService()

    # Extract face embedding from person bounding box
    embedding = face_service.extract_face_embedding(frame, bbox)

    # Get or create persistent track ID
    persistent_id = face_service.get_or_create_track_id(embedding)

    # Track journey
    face_service.update_journey(persistent_id, camera_id, location)
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import time
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Try importing InsightFace
try:
    from insightface.app import FaceAnalysis
    from insightface.data import get_image
    INSIGHTFACE_AVAILABLE = True
    logger.info("✓ InsightFace loaded successfully")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("⚠ InsightFace not available. Install: pip install insightface onnxruntime")


class FaceRecognitionService:
    """
    Manages face recognition for persistent tracking and re-identification.

    Features:
    - Face embedding extraction (512-dim vectors)
    - Similarity matching for re-identification
    - Persistent track ID assignment
    - Cross-camera tracking
    - Journey mapping (entry gate → platforms → exit)
    """

    def __init__(self,
                 similarity_threshold: float = 0.6,
                 min_face_size: int = 50,
                 embedding_dim: int = 512,
                 max_gallery_size: int = 1000):
        """
        Initialize face recognition service.

        Args:
            similarity_threshold: Minimum cosine similarity to match faces (0.6 = 60%)
            min_face_size: Minimum face size in pixels (reject small/blurry faces)
            embedding_dim: Dimension of face embeddings (512 for InsightFace)
            max_gallery_size: Maximum known faces to keep in memory
        """
        self.similarity_threshold = similarity_threshold
        self.min_face_size = min_face_size
        self.embedding_dim = embedding_dim
        self.max_gallery_size = max_gallery_size

        # Face analysis model
        self.face_app = None
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_app = FaceAnalysis(
                    name='buffalo_l',  # Medium accuracy, good speed
                    providers=['CPUExecutionProvider']  # Use GPU if available
                )
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("✓ InsightFace model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize InsightFace: {e}")
                self.face_app = None

        # Gallery of known faces
        # Format: {persistent_id: {'embedding': np.array, 'last_seen': timestamp, 'metadata': dict}}
        self.face_gallery: Dict[str, Dict] = {}

        # Track ID counter (incremental)
        self.next_track_id = 1

        # Journey tracking
        # Format: {persistent_id: [{'camera_id': str, 'location': str, 'timestamp': float}]}
        self.journeys: Dict[str, List[Dict]] = defaultdict(list)

        # Mapping: YOLO track_id -> persistent_id (per camera)
        # Format: {camera_id: {yolo_track_id: persistent_id}}
        self.track_id_mapping: Dict[str, Dict[int, str]] = defaultdict(dict)

        logger.info(f"✓ Face recognition service initialized (threshold={similarity_threshold})")


    def extract_face_embedding(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Extract face embedding from person bounding box.

        Args:
            frame: Full camera frame (BGR format)
            bbox: Person bounding box [x1, y1, x2, y2]

        Returns:
            512-dim face embedding or None if no face detected
        """
        if self.face_app is None:
            return None

        try:
            # Crop person from frame
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                return None

            # Focus on upper half (where face usually is)
            # This improves accuracy and speed
            height = person_crop.shape[0]
            upper_crop = person_crop[:int(height * 0.5), :]

            # Detect faces in crop
            faces = self.face_app.get(upper_crop)

            if len(faces) == 0:
                return None

            # Take largest face (closest to camera)
            faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            largest_face = faces[0]

            # Check face size
            face_w = largest_face.bbox[2] - largest_face.bbox[0]
            face_h = largest_face.bbox[3] - largest_face.bbox[1]
            if face_w < self.min_face_size or face_h < self.min_face_size:
                logger.debug(f"Face too small: {face_w}x{face_h}px")
                return None

            # Extract embedding (512-dim)
            embedding = largest_face.normed_embedding

            return embedding

        except Exception as e:
            logger.debug(f"Face extraction failed: {e}")
            return None


    def get_or_create_track_id(self,
                                embedding: np.ndarray,
                                camera_id: str,
                                location: str = "unknown") -> str:
        """
        Get existing persistent track ID or create new one based on face similarity.

        Args:
            embedding: Face embedding (512-dim)
            camera_id: Camera where person was detected
            location: Location description (e.g., "Entry Gate A", "Platform 2")

        Returns:
            Persistent track ID (format: "P001", "P002", etc.)
        """
        if embedding is None:
            # No face detected - assign temporary ID
            temp_id = f"TEMP-{int(time.time() * 1000)}"
            return temp_id

        # Search gallery for matching face
        best_match_id = None
        best_similarity = 0.0

        for persistent_id, face_data in self.face_gallery.items():
            stored_embedding = face_data['embedding']

            # Compute cosine similarity
            similarity = self._cosine_similarity(embedding, stored_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = persistent_id

        # Check if match is good enough
        if best_match_id and best_similarity >= self.similarity_threshold:
            # Update existing person
            self.face_gallery[best_match_id]['last_seen'] = time.time()
            self.face_gallery[best_match_id]['embedding'] = embedding  # Update with latest
            logger.info(f"✓ Re-identified person: {best_match_id} (similarity: {best_similarity:.2f})")

            # Update journey
            self.update_journey(best_match_id, camera_id, location)

            return best_match_id

        else:
            # New person - create persistent ID
            persistent_id = f"P{self.next_track_id:04d}"  # P0001, P0002, etc.
            self.next_track_id += 1

            # Add to gallery
            self.face_gallery[persistent_id] = {
                'embedding': embedding,
                'first_seen': time.time(),
                'last_seen': time.time(),
                'metadata': {
                    'entry_camera': camera_id,
                    'entry_location': location
                }
            }

            logger.info(f"✓ New person registered: {persistent_id} at {camera_id}")

            # Initialize journey
            self.update_journey(persistent_id, camera_id, location)

            # Clean up gallery if too large
            self._cleanup_gallery()

            return persistent_id


    def map_yolo_to_persistent_id(self,
                                   camera_id: str,
                                   yolo_track_id: int,
                                   embedding: Optional[np.ndarray],
                                   location: str = "unknown") -> str:
        """
        Map YOLO track ID to persistent face-based track ID.

        This allows YOLO tracks to be associated with persistent IDs.

        Args:
            camera_id: Camera ID
            yolo_track_id: Track ID from YOLO tracker
            embedding: Face embedding (if available)
            location: Location description

        Returns:
            Persistent track ID
        """
        # Check if we already have mapping for this YOLO track
        if yolo_track_id in self.track_id_mapping[camera_id]:
            return self.track_id_mapping[camera_id][yolo_track_id]

        # Get or create persistent ID based on face
        if embedding is not None:
            persistent_id = self.get_or_create_track_id(embedding, camera_id, location)
        else:
            # No face - use YOLO track ID as temporary ID
            persistent_id = f"{camera_id}_T{yolo_track_id}"

        # Store mapping
        self.track_id_mapping[camera_id][yolo_track_id] = persistent_id

        return persistent_id


    def update_journey(self, persistent_id: str, camera_id: str, location: str):
        """
        Update journey log for a person.

        Args:
            persistent_id: Persistent track ID
            camera_id: Current camera
            location: Current location
        """
        journey_entry = {
            'camera_id': camera_id,
            'location': location,
            'timestamp': time.time()
        }

        # Avoid duplicate consecutive entries
        if len(self.journeys[persistent_id]) > 0:
            last_entry = self.journeys[persistent_id][-1]
            if (last_entry['camera_id'] == camera_id and
                last_entry['location'] == location):
                # Just update timestamp
                last_entry['timestamp'] = time.time()
                return

        self.journeys[persistent_id].append(journey_entry)


    def get_journey(self, persistent_id: str) -> List[Dict]:
        """
        Get journey history for a person.

        Args:
            persistent_id: Persistent track ID

        Returns:
            List of journey entries
        """
        return self.journeys.get(persistent_id, [])


    def get_person_info(self, persistent_id: str) -> Optional[Dict]:
        """
        Get all information about a tracked person.

        Args:
            persistent_id: Persistent track ID

        Returns:
            Dictionary with person info or None
        """
        if persistent_id not in self.face_gallery:
            return None

        face_data = self.face_gallery[persistent_id]
        journey = self.get_journey(persistent_id)

        return {
            'persistent_id': persistent_id,
            'first_seen': face_data['first_seen'],
            'last_seen': face_data['last_seen'],
            'duration': face_data['last_seen'] - face_data['first_seen'],
            'entry_camera': face_data['metadata'].get('entry_camera'),
            'entry_location': face_data['metadata'].get('entry_location'),
            'journey': journey,
            'current_location': journey[-1] if journey else None
        }


    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        return float(similarity)


    def _cleanup_gallery(self):
        """
        Remove old entries if gallery exceeds max size.
        Keeps most recently seen faces.
        """
        if len(self.face_gallery) <= self.max_gallery_size:
            return

        # Sort by last_seen
        sorted_entries = sorted(
            self.face_gallery.items(),
            key=lambda x: x[1]['last_seen'],
            reverse=True
        )

        # Keep only max_gallery_size most recent
        self.face_gallery = dict(sorted_entries[:self.max_gallery_size])

        logger.info(f"Cleaned gallery: {len(self.face_gallery)} faces remaining")


    def cleanup_old_mappings(self, max_age_seconds: float = 300):
        """
        Clean up old track ID mappings.

        Args:
            max_age_seconds: Remove mappings older than this
        """
        current_time = time.time()

        for camera_id in list(self.track_id_mapping.keys()):
            for yolo_id in list(self.track_id_mapping[camera_id].keys()):
                persistent_id = self.track_id_mapping[camera_id][yolo_id]

                if persistent_id in self.face_gallery:
                    last_seen = self.face_gallery[persistent_id]['last_seen']
                    if current_time - last_seen > max_age_seconds:
                        del self.track_id_mapping[camera_id][yolo_id]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Test face recognition service.
    """
    print("=== Face Recognition Service Demo ===\n")

    # Initialize service
    face_service = FaceRecognitionService(
        similarity_threshold=0.6,
        min_face_size=50
    )

    if not INSIGHTFACE_AVAILABLE or face_service.face_app is None:
        print("❌ InsightFace not available")
        print("\nInstall with:")
        print("  pip install insightface onnxruntime")
        exit(1)

    print("✓ Face recognition service ready\n")

    # Simulate person detection
    print("Simulating person entering at Entry Gate A...")

    # In real usage, you'd get this from camera frame + YOLO bbox
    # For demo, we'll use a test image
    test_image_path = "test_face.jpg"

    if not os.path.exists(test_image_path):
        print(f"⚠ No test image found at {test_image_path}")
        print("\nIn real usage:")
        print("  frame = cv2.imread('camera_frame.jpg')")
        print("  bbox = [100, 50, 300, 400]  # From YOLO")
        print("  embedding = face_service.extract_face_embedding(frame, bbox)")
        print("  persistent_id = face_service.get_or_create_track_id(embedding, 'camera_1', 'Entry Gate A')")
    else:
        import os
        frame = cv2.imread(test_image_path)
        bbox = [0, 0, frame.shape[1], frame.shape[0]]

        embedding = face_service.extract_face_embedding(frame, bbox)

        if embedding is not None:
            persistent_id = face_service.get_or_create_track_id(
                embedding,
                camera_id='gate_cam_a',
                location='Entry Gate A'
            )

            print(f"✓ Person registered: {persistent_id}")

            # Simulate movement to platform
            print(f"\nPerson {persistent_id} moves to Platform 1...")
            face_service.update_journey(persistent_id, 'platform_cam_1', 'Platform 1 - North')

            # Get journey
            journey = face_service.get_journey(persistent_id)
            print(f"\nJourney for {persistent_id}:")
            for entry in journey:
                print(f"  - {entry['location']} (camera: {entry['camera_id']}) at {entry['timestamp']}")

        else:
            print("❌ No face detected in test image")

    print("\n✓ Demo complete!")
