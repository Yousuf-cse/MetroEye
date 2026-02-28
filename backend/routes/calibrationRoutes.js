/**
 * Calibration API Routes for MetroEye
 * ===================================
 *
 * Handles platform edge calibration for cameras
 * Python does the heavy lifting, Node.js serves the API
 *
 * Endpoints:
 * - GET    /api/calibration/:cameraId          - Get calibration status
 * - GET    /api/calibration/:cameraId/preview  - Get preview image with overlay
 * - POST   /api/calibration/:cameraId/manual   - Save manual calibration points
 * - POST   /api/calibration/:cameraId/auto     - Trigger auto calibration
 * - DELETE /api/calibration/:cameraId          - Delete calibration
 */

const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs').promises;
const fsSync = require('fs');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

// Path to calibration configs (saved by Python)
const CONFIGS_DIR = path.join(__dirname, '../../vision-engine/configs');
const PYTHON_SCRIPT = path.join(__dirname, '../../vision-engine/edge_calibrator.py');

/**
 * Helper: Get config file path for a camera
 */
function getConfigPath(cameraId) {
  return path.join(CONFIGS_DIR, `${cameraId}_calibration.json`);
}

/**
 * Helper: Denormalize coordinates to specific resolution
 * Converts normalized (0-1) coords to pixel coords for given resolution
 */
function denormalizeCoordinates(normalizedPoints, width, height) {
  return normalizedPoints.map(([normX, normY]) => [
    Math.round(normX * width),
    Math.round(normY * height)
  ]);
}

/**
 * Helper: Normalize coordinates from pixel to 0-1 range
 */
function normalizeCoordinates(pixelPoints, width, height) {
  return pixelPoints.map(([x, y]) => [
    parseFloat((x / width).toFixed(4)),
    parseFloat((y / height).toFixed(4))
  ]);
}

/**
 * GET /api/calibration/:cameraId
 * Get calibration status and data for a camera
 */
router.get('/:cameraId', async (req, res) => {
  const { cameraId } = req.params;
  const configPath = getConfigPath(cameraId);

  try {
    // Check if calibration exists
    const exists = fsSync.existsSync(configPath);

    if (!exists) {
      return res.json({
        camera_id: cameraId,
        calibrated: false,
        data: null
      });
    }

    // Load calibration data
    const configData = await fs.readFile(configPath, 'utf-8');
    const config = JSON.parse(configData);

    res.json({
      camera_id: cameraId,
      calibrated: true,
      data: {
        detection_method: config.detection_method,
        video_dimensions: config.video_dimensions,
        platform_edge: config.platform_edge,
        num_points: config.num_points,
        video_source: config.video_source
      }
    });

  } catch (error) {
    console.error('Error loading calibration:', error);
    res.status(500).json({
      error: 'Failed to load calibration',
      details: error.message
    });
  }
});

/**
 * GET /api/calibration/:cameraId/for-display
 * Get calibration points scaled for specific display resolution
 *
 * Query params:
 *   - width: Display width (e.g., 640)
 *   - height: Display height (e.g., 360)
 *
 * Returns pixel coordinates for the specified display size
 */
router.get('/:cameraId/for-display', async (req, res) => {
  const { cameraId } = req.params;
  const { width, height } = req.query;

  if (!width || !height) {
    return res.status(400).json({
      error: 'width and height query parameters required'
    });
  }

  const displayWidth = parseInt(width);
  const displayHeight = parseInt(height);

  try {
    const configPath = getConfigPath(cameraId);
    const configData = await fs.readFile(configPath, 'utf-8');
    const config = JSON.parse(configData);

    // Get normalized coordinates
    const normalizedPoints = config.platform_edge.normalized;

    // Convert to display resolution
    const displayPoints = denormalizeCoordinates(
      normalizedPoints,
      displayWidth,
      displayHeight
    );

    res.json({
      camera_id: cameraId,
      display_resolution: { width: displayWidth, height: displayHeight },
      original_resolution: config.video_dimensions,
      points: displayPoints,
      normalized_points: normalizedPoints
    });

  } catch (error) {
    console.error('Error converting calibration:', error);
    res.status(500).json({
      error: 'Failed to convert calibration',
      details: error.message
    });
  }
});

/**
 * POST /api/calibration/:cameraId/manual
 * Save manual calibration points from frontend
 *
 * Request body:
 * {
 *   "points": [[x1, y1], [x2, y2], ...],  // Pixel coordinates
 *   "video_dimensions": {"width": 1920, "height": 1080},
 *   "video_source": "camera_feed_url or file path"
 * }
 */
router.post('/:cameraId/manual', async (req, res) => {
  const { cameraId } = req.params;
  const { points, video_dimensions, video_source } = req.body;

  // Validation
  if (!points || !Array.isArray(points) || points.length < 3) {
    return res.status(400).json({
      error: 'At least 3 points required for calibration'
    });
  }

  if (!video_dimensions || !video_dimensions.width || !video_dimensions.height) {
    return res.status(400).json({
      error: 'video_dimensions (width, height) required'
    });
  }

  try {
    const { width, height } = video_dimensions;

    // Normalize the points
    const normalizedPoints = normalizeCoordinates(points, width, height);

    // Create calibration config
    const config = {
      camera_id: cameraId,
      video_dimensions: { width, height },
      platform_edge: {
        normalized: normalizedPoints,  // 0.0-1.0 range (resolution-independent)
        absolute: points                // Original pixel coordinates
      },
      video_source: video_source || 'unknown',
      detection_method: 'manual',
      num_points: points.length,
      created_at: new Date().toISOString()
    };

    // Ensure configs directory exists
    await fs.mkdir(CONFIGS_DIR, { recursive: true });

    // Save configuration
    const configPath = getConfigPath(cameraId);
    await fs.writeFile(configPath, JSON.stringify(config, null, 2));

    console.log(`✓ Saved manual calibration for ${cameraId}`);
    console.log(`  Points: ${points.length}`);
    console.log(`  Resolution: ${width}x${height}`);

    // Broadcast to WebSocket clients
    const io = req.app.get('io');
    if (io) {
      io.to(`calibration_${cameraId}`).emit('calibration_updated', {
        camera_id: cameraId,
        method: 'manual',
        num_points: points.length
      });
    }

    res.status(201).json({
      success: true,
      camera_id: cameraId,
      detection_method: 'manual',
      num_points: points.length,
      data: config
    });

  } catch (error) {
    console.error('Error saving manual calibration:', error);
    res.status(500).json({
      error: 'Failed to save calibration',
      details: error.message
    });
  }
});

/**
 * POST /api/calibration/:cameraId/auto
 * Trigger automatic calibration using Python script
 *
 * Request body:
 * {
 *   "video_path": "/path/to/video.mp4",
 *   "method": "auto" | "yolo" | "hough",  // optional, default: "auto"
 *   "force": true | false                  // optional, recalibrate even if exists
 * }
 */
router.post('/:cameraId/auto', async (req, res) => {
  const { cameraId } = req.params;
  const { video_path, method = 'auto', force = false } = req.body;

  if (!video_path) {
    return res.status(400).json({
      error: 'video_path required'
    });
  }

  try {
    // Build Python command
    const forceFlag = force ? '--force' : '';
    const command = `python "${PYTHON_SCRIPT}" --video "${video_path}" --camera "${cameraId}" --method ${method} ${forceFlag}`;

    console.log(`Running auto calibration: ${command}`);

    // Execute Python script (this may take 10-30 seconds)
    const { stdout, stderr } = await execPromise(command, {
      timeout: 60000, // 60 second timeout
      maxBuffer: 1024 * 1024 * 10 // 10MB buffer
    });

    console.log('Python output:', stdout);
    if (stderr) console.error('Python errors:', stderr);

    // Check if calibration was created
    const configPath = getConfigPath(cameraId);
    const exists = fsSync.existsSync(configPath);

    if (!exists) {
      return res.status(400).json({
        success: false,
        error: 'Calibration failed - no config file created',
        python_output: stdout
      });
    }

    // Load the created calibration
    const configData = await fs.readFile(configPath, 'utf-8');
    const config = JSON.parse(configData);

    // Broadcast to WebSocket clients
    const io = req.app.get('io');
    if (io) {
      io.to(`calibration_${cameraId}`).emit('calibration_updated', {
        camera_id: cameraId,
        method: config.detection_method,
        num_points: config.num_points
      });
    }

    res.status(201).json({
      success: true,
      camera_id: cameraId,
      detection_method: config.detection_method,
      num_points: config.num_points,
      data: config,
      python_output: stdout
    });

  } catch (error) {
    console.error('Error running auto calibration:', error);

    // Check if it's a timeout
    if (error.killed) {
      return res.status(504).json({
        error: 'Calibration timed out (>60s)',
        details: error.message
      });
    }

    res.status(500).json({
      error: 'Auto calibration failed',
      details: error.message,
      stderr: error.stderr
    });
  }
});

/**
 * DELETE /api/calibration/:cameraId
 * Delete calibration for a camera
 */
router.delete('/:cameraId', async (req, res) => {
  const { cameraId } = req.params;
  const configPath = getConfigPath(cameraId);

  try {
    const exists = fsSync.existsSync(configPath);

    if (!exists) {
      return res.status(404).json({
        error: 'No calibration found for this camera'
      });
    }

    await fs.unlink(configPath);

    console.log(`✓ Deleted calibration for ${cameraId}`);

    // Broadcast to WebSocket clients
    const io = req.app.get('io');
    if (io) {
      io.to(`calibration_${cameraId}`).emit('calibration_deleted', {
        camera_id: cameraId
      });
    }

    res.json({
      success: true,
      message: `Calibration deleted for ${cameraId}`
    });

  } catch (error) {
    console.error('Error deleting calibration:', error);
    res.status(500).json({
      error: 'Failed to delete calibration',
      details: error.message
    });
  }
});

module.exports = router;
