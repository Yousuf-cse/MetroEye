/**
 * Driver Alert Routes - Simple POST API
 *
 * POST /api/driver-alert - Send alert when risk >= 0.85
 */

const express = require('express');
const router = express.Router();

// Store latest alert (in-memory - use DB in production)
let latestAlert = null;
let recentAlerts = [];
const MAX_ALERTS = 100;

/**
 * POST /api/driver-alert
 *
 * Send driver alert when high risk detected (>= 0.85)
 *
 * Body:
 * {
 *   "track_id": 42,
 *   "risk_score": 0.92,
 *   "camera_id": "platform_3_camA",
 *   "distance_from_edge": 25,
 *   "timestamp": 1234567890
 * }
 */
router.post('/driver-alert', (req, res) => {
  try {
    const { track_id, risk_score, camera_id, distance_from_edge, timestamp } = req.body;

    // Validate
    if (risk_score === undefined || track_id === undefined) {
      return res.status(400).json({
        error: 'Missing required fields: track_id, risk_score'
      });
    }

    // Only accept high-risk alerts (>= 0.85)
    if (risk_score < 0.85) {
      return res.status(200).json({
        message: 'Risk below threshold (0.85), alert not created'
      });
    }

    // Create alert
    const alert = {
      track_id,
      risk_score,
      camera_id: camera_id || 'unknown',
      distance_from_edge: distance_from_edge || null,
      timestamp: timestamp || Date.now(),
      alert_type: risk_score >= 0.95 ? 'EMERGENCY' : 'CRITICAL',
      created_at: new Date().toISOString()
    };

    // Store alert
    latestAlert = alert;
    recentAlerts.push(alert);

    // Keep only recent alerts
    if (recentAlerts.length > MAX_ALERTS) {
      recentAlerts.shift();
    }

    console.log(`🚨 DRIVER ALERT: Track ${track_id}, Risk ${risk_score.toFixed(2)}, Camera ${camera_id}`);

    // Emit to connected clients via Socket.IO
    if (req.io) {
      req.io.emit('driver_alert', alert);
      console.log('  └─ Alert broadcasted to dashboard');
    }

    res.status(200).json({
      success: true,
      message: 'Driver alert created',
      alert: alert
    });

  } catch (error) {
    console.error('Error creating driver alert:', error);
    res.status(500).json({
      error: 'Failed to create driver alert',
      details: error.message
    });
  }
});

/**
 * GET /api/driver-alert/latest
 *
 * Get the most recent driver alert
 */
router.get('/driver-alert/latest', (req, res) => {
  if (latestAlert) {
    res.json(latestAlert);
  } else {
    res.json({ message: 'No alerts' });
  }
});

/**
 * GET /api/driver-alert/recent
 *
 * Get recent driver alerts
 */
router.get('/driver-alert/recent', (req, res) => {
  const limit = parseInt(req.query.limit) || 10;
  const recent = recentAlerts.slice(-limit);

  res.json({
    alerts: recent,
    count: recent.length
  });
});

module.exports = router;
