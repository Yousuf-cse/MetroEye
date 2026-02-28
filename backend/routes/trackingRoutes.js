const express = require('express');
const router = express.Router();

// Store latest tracking data for each camera (in-memory cache)
const trackingCache = new Map();

// Receive tracking data from Python
router.post('/:cameraId', (req, res) => {
  const { cameraId } = req.params;
  const trackingData = req.body;
  
  // Validate data
  if (!trackingData.objects || !Array.isArray(trackingData.objects)) {
    return res.status(400).json({
      success: false,
      error: 'Invalid tracking data format. Expected: { timestamp, objects: [...] }'
    });
  }
  
  console.log(`📊 Received tracking data for ${cameraId}: ${trackingData.objects.length} objects`);
  
  // Update cache
  trackingCache.set(cameraId, {
    ...trackingData,
    received_at: Date.now()
  });
  
  // Broadcast to all subscribed WebSocket clients
  req.io.to(`tracking_${cameraId}`).emit('tracking_update', {
    camera_id: cameraId,
    timestamp: trackingData.timestamp || Date.now(),
    objects: trackingData.objects
  });
  
  res.json({
    success: true,
    message: 'Tracking data received',
    objects_count: trackingData.objects.length
  });
});

// Get latest tracking data (for polling fallback)
router.get('/:cameraId', (req, res) => {
  const { cameraId } = req.params;
  const data = trackingCache.get(cameraId);
  
  if (!data) {
    return res.status(404).json({
      success: false,
      error: 'No tracking data available for this camera'
    });
  }
  
  // Check if data is stale (older than 5 seconds)
  const ageMs = Date.now() - data.received_at;
  if (ageMs > 5000) {
    return res.status(503).json({
      success: false,
      error: 'Tracking data is stale',
      age_seconds: ageMs / 1000
    });
  }
  
  res.json({
    success: true,
    data: data
  });
});

// Get tracking statistics
router.get('/:cameraId/stats', (req, res) => {
  const { cameraId } = req.params;
  const data = trackingCache.get(cameraId);
  
  if (!data) {
    return res.json({
      active: false,
      total_objects: 0
    });
  }
  
  const ageMs = Date.now() - data.received_at;
  
  res.json({
    active: ageMs < 5000,
    total_objects: data.objects.length,
    last_update: new Date(data.received_at).toISOString(),
    age_seconds: ageMs / 1000,
    high_risk_count: data.objects.filter(obj => obj.risk_score > 70).length
  });
});

module.exports = router;