/**
 * MongoDB Schema for Platform Edge Calibration
 * ============================================
 *
 * Stores calibration data with normalized coordinates (0.0-1.0) for
 * resolution independence.
 *
 * Features:
 * - Normalized points work at any resolution
 * - Automatic validation
 * - Helper method to scale points
 * - Timestamp tracking
 */

const mongoose = require('mongoose');

const calibrationSchema = new mongoose.Schema({
  camera_id: {
    type: String,
    required: true,
    unique: true,
    index: true,
    description: 'Unique identifier for camera'
  },

  // Normalized coordinates (0.0 - 1.0) - RESOLUTION INDEPENDENT!
  // Example: [[0.0, 0.15], [0.95, 0.35]] means start at top-left, end at bottom-right
  normalized_points: {
    type: [[Number]],  // Array of [x, y] pairs
    required: true,
    validate: {
      validator: function(points) {
        // Check each point is [x, y] with values between 0-1
        return points.every(point =>
          Array.isArray(point) &&
          point.length === 2 &&
          point[0] >= 0 && point[0] <= 1 &&
          point[1] >= 0 && point[1] <= 1
        );
      },
      message: 'Points must be [x, y] pairs normalized between 0.0 and 1.0'
    }
  },

  // Original absolute coordinates (for reference)
  absolute_points: {
    type: [[Number]],
    description: 'Original pixel coordinates when calibrated'
  },

  // Video dimensions when calibrated
  video_dimensions: {
    width: {
      type: Number,
      required: true,
      min: 1
    },
    height: {
      type: Number,
      required: true,
      min: 1
    }
  },

  // Detection method used
  detection_method: {
    type: String,
    enum: ['manual', 'yolo', 'yolo-seg', 'hough', 'hough-transform', 'auto'],
    required: true,
    description: 'Method used to detect platform edge'
  },

  // Metadata
  calibrated_at: {
    type: Date,
    default: Date.now
  },

  calibrated_by: {
    type: String,
    default: 'system',
    description: 'User or system that performed calibration'
  },

  // Optional notes
  notes: {
    type: String,
    maxlength: 500
  }

}, {
  timestamps: true,  // Adds createdAt and updatedAt automatically
  collection: 'calibrations'
});

/**
 * Method to get scaled points for specific resolution
 *
 * Example:
 *   const calibration = await Calibration.findOne({camera_id: 'cam1'});
 *   const scaled = calibration.getScaledPoints(1920, 1080);
 *   // Returns: [[0, 162], [1824, 378]]
 *
 * @param {number} targetWidth - Target frame width
 * @param {number} targetHeight - Target frame height
 * @returns {Array} Array of [x, y] pairs scaled to target resolution
 */
calibrationSchema.methods.getScaledPoints = function(targetWidth, targetHeight) {
  return this.normalized_points.map(([x, y]) => [
    Math.round(x * targetWidth),
    Math.round(y * targetHeight)
  ]);
};

/**
 * Static method to create or update calibration
 *
 * @param {string} cameraId - Camera identifier
 * @param {Array} points - Absolute pixel coordinates
 * @param {Object} videoDimensions - {width, height}
 * @param {string} detectionMethod - How edge was detected
 * @param {string} calibratedBy - Who/what calibrated
 * @returns {Promise} Calibration document
 */
calibrationSchema.statics.createOrUpdate = async function(
  cameraId,
  points,
  videoDimensions,
  detectionMethod,
  calibratedBy = 'system'
) {
  // Normalize points
  const normalized = points.map(([x, y]) => [
    x / videoDimensions.width,
    y / videoDimensions.height
  ]);

  // Update or create
  return await this.findOneAndUpdate(
    { camera_id: cameraId },
    {
      camera_id: cameraId,
      normalized_points: normalized,
      absolute_points: points,
      video_dimensions: videoDimensions,
      detection_method: detectionMethod,
      calibrated_by: calibratedBy,
      calibrated_at: new Date()
    },
    { upsert: true, new: true }
  );
};

/**
 * Index for fast lookups
 */
calibrationSchema.index({ camera_id: 1 });

module.exports = mongoose.model('Calibration', calibrationSchema);
