const express = require('express');
const router = express.Router();

// Send test tracking data
router.post('/send-test-tracking', (req, res) => {
  const testData = {
    camera_id: 'camera_1',
    timestamp: Date.now(),
    objects: [
      {
        track_id: 1,
        bbox: [100, 150, 200, 350],
        risk_score: 45,
        speed: 120.5,
        confidence: 0.92,
        keypoints: []
      },
      {
        track_id: 2,
        bbox: [300, 200, 400, 400],
        risk_score: 75,
        speed: 250.3,
        confidence: 0.88,
        keypoints: []
      }
    ]
  };

  req.io.to('tracking_camera_1').emit('tracking_update', testData);
  
  res.json({ success: true, message: 'Test tracking data sent' });
});

// Send test alert
router.post('/send-test-alert', async (req, res) => {
  const testAlert = {
    camera_id: 'camera_1',
    track_id: 5,
    risk_score: 85,
    risk_level: 'high',
    alert_message: 'Test alert: Person pacing near edge',
    confidence: 0.92,
    timestamp: new Date(),
    features: {
      speed: 250.5,
      distance_to_edge: 50
    },
    llm_reasoning: 'This is a test alert for demonstration purposes',
    status: 'pending'
  };

  // Save to database
  const Alert = require('../models/Alert');
  const alert = new Alert(testAlert);
  await alert.save();

  // Broadcast via WebSocket
  req.io.emit('new_alert', {
    type: 'new_alert',
    data: alert
  });

  res.json({ success: true, data: alert });
});

module.exports = router;