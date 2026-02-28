/**
 * Coordinate Transformation Utilities
 * ====================================
 *
 * Handles conversion between normalized (0-1) and pixel coordinates
 * Ensures calibration points work uniformly across different display sizes
 *
 * Key Concept:
 * - Store coordinates as normalized (0.0 to 1.0)
 * - Convert to pixels based on target resolution
 * - Works for any display size (floating window, fullscreen, mobile, etc.)
 */

/**
 * Normalize pixel coordinates to 0-1 range
 *
 * @param {Array<Array<number>>} points - Array of [x, y] pixel coordinates
 * @param {number} width - Original image/video width in pixels
 * @param {number} height - Original image/video height in pixels
 * @returns {Array<Array<number>>} - Array of [normX, normY] in 0-1 range
 *
 * @example
 * // Video is 1920x1080, point clicked at pixel (960, 540)
 * normalizeCoordinates([[960, 540]], 1920, 1080)
 * // Returns: [[0.5, 0.5]] (center of image)
 */
function normalizeCoordinates(points, width, height) {
  return points.map(([x, y]) => [
    parseFloat((x / width).toFixed(4)),
    parseFloat((y / height).toFixed(4))
  ]);
}

/**
 * Denormalize coordinates from 0-1 range to pixel coordinates
 *
 * @param {Array<Array<number>>} normalizedPoints - Array of [normX, normY] in 0-1 range
 * @param {number} width - Target width in pixels
 * @param {number} height - Target height in pixels
 * @returns {Array<Array<number>>} - Array of [x, y] pixel coordinates
 *
 * @example
 * // Want to display on 640x360 screen
 * denormalizeCoordinates([[0.5, 0.5]], 640, 360)
 * // Returns: [[320, 180]] (center of 640x360 display)
 */
function denormalizeCoordinates(normalizedPoints, width, height) {
  return normalizedPoints.map(([normX, normY]) => [
    Math.round(normX * width),
    Math.round(normY * height)
  ]);
}

/**
 * Get calibration points for specific display resolution
 *
 * @param {Object} calibrationData - Calibration data from config file
 * @param {number} displayWidth - Target display width
 * @param {number} displayHeight - Target display height
 * @returns {Array<Array<number>>} - Pixel coordinates for display
 *
 * @example
 * const calibration = loadCalibration('camera_1');
 * const pointsForMobile = getPointsForDisplay(calibration, 375, 667);
 * const pointsForDesktop = getPointsForDisplay(calibration, 1920, 1080);
 */
function getPointsForDisplay(calibrationData, displayWidth, displayHeight) {
  const normalizedPoints = calibrationData.platform_edge.normalized;
  return denormalizeCoordinates(normalizedPoints, displayWidth, displayHeight);
}

/**
 * Scale bounding box from one resolution to another
 *
 * @param {Object} bbox - Bounding box {x, y, width, height}
 * @param {Object} sourceResolution - {width, height} of source
 * @param {Object} targetResolution - {width, height} of target
 * @returns {Object} - Scaled bounding box
 *
 * @example
 * const bbox = { x: 100, y: 50, width: 200, height: 300 };
 * scaleBoundingBox(bbox, {width: 1920, height: 1080}, {width: 640, height: 360})
 */
function scaleBoundingBox(bbox, sourceResolution, targetResolution) {
  const scaleX = targetResolution.width / sourceResolution.width;
  const scaleY = targetResolution.height / sourceResolution.height;

  return {
    x: Math.round(bbox.x * scaleX),
    y: Math.round(bbox.y * scaleY),
    width: Math.round(bbox.width * scaleX),
    height: Math.round(bbox.height * scaleY)
  };
}

/**
 * Check if a point is inside a polygon (platform edge detection)
 * Uses ray casting algorithm
 *
 * @param {Array<number>} point - [x, y] coordinates
 * @param {Array<Array<number>>} polygon - Array of [x, y] coordinates
 * @returns {boolean} - True if point is inside polygon
 *
 * @example
 * const platformEdge = [[0, 0], [100, 0], [100, 100], [0, 100]];
 * isPointInPolygon([50, 50], platformEdge) // true
 * isPointInPolygon([150, 150], platformEdge) // false
 */
function isPointInPolygon(point, polygon) {
  const [x, y] = point;
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];

    const intersect = ((yi > y) !== (yj > y)) &&
      (x < (xj - xi) * (y - yi) / (yj - yi) + xi);

    if (intersect) inside = !inside;
  }

  return inside;
}

/**
 * Check if person bounding box is on platform
 *
 * @param {Object} bbox - Person bounding box {x, y, width, height}
 * @param {Array<Array<number>>} platformPolygon - Platform edge points
 * @param {number} threshold - Percentage of bbox that must be on platform (0-1)
 * @returns {boolean} - True if person is on platform
 *
 * @example
 * const bbox = { x: 100, y: 100, width: 50, height: 150 };
 * const platform = [[0, 80], [200, 80], [200, 200], [0, 200]];
 * isPersonOnPlatform(bbox, platform, 0.5) // true if >50% of bbox is on platform
 */
function isPersonOnPlatform(bbox, platformPolygon, threshold = 0.5) {
  // Check bottom-center point of bbox (person's feet)
  const feetPoint = [
    bbox.x + bbox.width / 2,
    bbox.y + bbox.height
  ];

  return isPointInPolygon(feetPoint, platformPolygon);
}

/**
 * Calculate distance from point to line segment
 * Used to check how close person is to platform edge
 *
 * @param {Array<number>} point - [x, y]
 * @param {Array<number>} lineStart - [x, y]
 * @param {Array<number>} lineEnd - [x, y]
 * @returns {number} - Distance in pixels
 */
function pointToLineDistance(point, lineStart, lineEnd) {
  const [x, y] = point;
  const [x1, y1] = lineStart;
  const [x2, y2] = lineEnd;

  const A = x - x1;
  const B = y - y1;
  const C = x2 - x1;
  const D = y2 - y1;

  const dot = A * C + B * D;
  const lenSq = C * C + D * D;
  let param = -1;

  if (lenSq !== 0) {
    param = dot / lenSq;
  }

  let xx, yy;

  if (param < 0) {
    xx = x1;
    yy = y1;
  } else if (param > 1) {
    xx = x2;
    yy = y2;
  } else {
    xx = x1 + param * C;
    yy = y1 + param * D;
  }

  const dx = x - xx;
  const dy = y - yy;

  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Get minimum distance from point to platform edge
 *
 * @param {Array<number>} point - [x, y]
 * @param {Array<Array<number>>} platformEdge - Array of edge points
 * @returns {number} - Minimum distance in pixels
 */
function getDistanceToEdge(point, platformEdge) {
  let minDistance = Infinity;

  for (let i = 0; i < platformEdge.length - 1; i++) {
    const distance = pointToLineDistance(
      point,
      platformEdge[i],
      platformEdge[i + 1]
    );
    minDistance = Math.min(minDistance, distance);
  }

  return minDistance;
}

module.exports = {
  normalizeCoordinates,
  denormalizeCoordinates,
  getPointsForDisplay,
  scaleBoundingBox,
  isPointInPolygon,
  isPersonOnPlatform,
  pointToLineDistance,
  getDistanceToEdge
};
