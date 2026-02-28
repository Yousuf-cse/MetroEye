"""
Person Appearance Extraction Module
====================================

Extracts person appearance features for identification in alerts.
Uses YOLO detection results + color analysis (no extra models needed).

Fast enough for real-time processing (~2-5ms per person).
"""

import cv2
import numpy as np
from collections import Counter


class PersonAppearanceExtractor:
    """Extract appearance features from person detections"""

    def __init__(self):
        """Initialize appearance extractor"""
        self.color_map = {
            'red': (0, 15, 165, 180),      # Hue ranges in HSV
            'orange': (15, 30),
            'yellow': (30, 45),
            'green': (45, 75),
            'cyan': (75, 95),
            'blue': (95, 135),
            'purple': (135, 165),
        }

    def extract_appearance(self, frame, bbox, keypoints=None):
        """
        Extract person appearance from frame

        Args:
            frame: Full video frame
            bbox: Person bounding box (x1, y1, x2, y2)
            keypoints: Optional YOLO pose keypoints

        Returns:
            dict: Appearance features
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Crop person region
        person_crop = frame[y1:y2, x1:x2]
        h, w = person_crop.shape[:2]

        if h == 0 or w == 0:
            return self._default_appearance()

        # Split into body regions
        # Upper body: 20-60% of height (shoulders to waist)
        # Lower body: 60-90% of height (waist to knees)
        upper_body = person_crop[int(h*0.2):int(h*0.6), :]
        lower_body = person_crop[int(h*0.6):int(h*0.9), :]

        # Extract colors
        upper_color = self._get_dominant_color(upper_body)
        lower_color = self._get_dominant_color(lower_body)

        # Height estimation (relative to frame)
        height_px = y2 - y1
        frame_height = frame.shape[0]
        height_ratio = height_px / frame_height

        if height_ratio > 0.4:
            height_category = "tall"
        elif height_ratio > 0.25:
            height_category = "average"
        else:
            height_category = "short or distant"

        # Detect accessories (simple heuristics)
        has_bag = self._detect_bag(person_crop)

        # Generate description
        description = self._generate_description(
            upper_color, lower_color, height_category, has_bag
        )

        return {
            "upper_clothing": upper_color,
            "lower_clothing": lower_color,
            "height": height_category,
            "has_bag": has_bag,
            "description": description,
            "height_pixels": height_px
        }

    def _get_dominant_color(self, image_region):
        """
        Get dominant color from image region

        Args:
            image_region: BGR image region

        Returns:
            str: Color name
        """
        if image_region.size == 0:
            return "unknown"

        # Convert to HSV
        hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)

        # Get statistics
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])

        # Check for achromatic colors first (low saturation)
        if avg_saturation < 30:
            if avg_value < 50:
                return "black"
            elif avg_value > 200:
                return "white"
            else:
                return "gray"

        # Determine color from hue
        if avg_hue < 10 or avg_hue > 170:
            return "red"
        elif 10 <= avg_hue < 25:
            return "orange"
        elif 25 <= avg_hue < 35:
            return "yellow"
        elif 35 <= avg_hue < 85:
            return "green"
        elif 85 <= avg_hue < 130:
            return "blue"
        elif 130 <= avg_hue < 150:
            return "purple"
        elif 150 <= avg_hue < 170:
            return "pink"

        return "unknown"

    def _detect_bag(self, person_crop):
        """
        Simple bag detection heuristic

        Looks for rectangular regions on shoulder/back area

        Args:
            person_crop: Person image

        Returns:
            bool: Whether bag likely present
        """
        h, w = person_crop.shape[:2]

        # Focus on shoulder/back area (top 50% of person)
        shoulder_region = person_crop[0:int(h*0.5), :]

        # Convert to grayscale
        gray = cv2.cvtColor(shoulder_region, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look for rectangular contours (bags often have straight edges)
        for contour in contours:
            area = cv2.contourArea(contour)

            # Bag should be 10-40% of person area
            if 0.1 * gray.size < area < 0.4 * gray.size:
                # Approximate polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

                # Bags tend to be rectangular (4-6 vertices)
                if 4 <= len(approx) <= 6:
                    return True

        return False

    def _generate_description(self, upper_color, lower_color, height, has_bag):
        """
        Generate natural language description

        Args:
            upper_color: Upper clothing color
            lower_color: Lower clothing color
            height: Height category
            has_bag: Whether person has bag

        Returns:
            str: Description
        """
        desc_parts = []

        # Height
        if height != "short or distant":
            desc_parts.append(height)

        desc_parts.append("person")

        # Clothing
        if upper_color != "unknown":
            upper_garment = self._guess_upper_garment(upper_color)
            desc_parts.append(f"wearing {upper_color} {upper_garment}")

        if lower_color != "unknown" and lower_color != upper_color:
            lower_garment = self._guess_lower_garment(lower_color)
            desc_parts.append(f"and {lower_color} {lower_garment}")

        # Accessories
        if has_bag:
            desc_parts.append("carrying a bag")

        return " ".join(desc_parts).capitalize()

    def _guess_upper_garment(self, color):
        """Guess upper garment type (generic)"""
        # Simple heuristic - could be enhanced
        return "shirt/jacket"

    def _guess_lower_garment(self, color):
        """Guess lower garment type (generic)"""
        return "pants"

    def _default_appearance(self):
        """Return default appearance when extraction fails"""
        return {
            "upper_clothing": "unknown",
            "lower_clothing": "unknown",
            "height": "unknown",
            "has_bag": False,
            "description": "Person (appearance unclear)",
            "height_pixels": 0
        }


def test_appearance_extractor():
    """Test the appearance extractor"""
    import cv2

    # Load test image
    frame = cv2.imread("test_frame.jpg")

    if frame is None:
        print("Please provide a test image 'test_frame.jpg'")
        return

    extractor = PersonAppearanceExtractor()

    # Example bbox (you'd get this from YOLO)
    bbox = (100, 200, 300, 500)  # x1, y1, x2, y2

    appearance = extractor.extract_appearance(frame, bbox)

    print("Appearance Features:")
    print(f"  Description: {appearance['description']}")
    print(f"  Upper clothing: {appearance['upper_clothing']}")
    print(f"  Lower clothing: {appearance['lower_clothing']}")
    print(f"  Height: {appearance['height']}")
    print(f"  Has bag: {appearance['has_bag']}")


if __name__ == "__main__":
    test_appearance_extractor()
