const express = require('express');
const router = express.Router();
const axios = require('axios');

const PYTHON_FASTAPI_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:5000';

// Proxy the MJPEG stream from Python FastAPI
router.get('/:cameraId/stream', async (req, res) => {
  const { cameraId } = req.params;
  
  try {
    // Make request to Python FastAPI
    const response = await axios({
      method: 'GET',
      url: `${PYTHON_FASTAPI_URL}/stream/${cameraId}`,
      responseType: 'stream',
      timeout: 30000  // 30 seconds timeout
    });
    
    // Set headers for MJPEG stream
    res.setHeader('Content-Type', 'multipart/x-mixed-replace; boundary=frame');
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    
    // Pipe the stream from Python to client
    response.data.pipe(res);
    
    // Handle errors
    response.data.on('error', (error) => {
      console.error(`Stream error for ${cameraId}:`, error.message);
      res.end();
    });
    
    // Clean up on client disconnect
    req.on('close', () => {
      response.data.destroy();
    });
    
  } catch (error) {
    console.error(`Failed to connect to camera ${cameraId}:`, error.message);
    res.status(503).json({
      success: false,
      error: 'Camera stream unavailable'
    });
  }
});

// Get single frame (for testing)
router.get('/:cameraId/frame', async (req, res) => {
  const { cameraId } = req.params;
  
  try {
    const response = await axios({
      method: 'GET',
      url: `${PYTHON_FASTAPI_URL}/frame/${cameraId}`,
      responseType: 'arraybuffer',
      timeout: 5000
    });
    
    res.setHeader('Content-Type', 'image/jpeg');
    res.send(response.data);
    
  } catch (error) {
    console.error(`Failed to get frame for ${cameraId}:`, error.message);
    res.status(503).json({
      success: false,
      error: 'Frame unavailable'
    });
  }
});

// Get camera status
router.get('/:cameraId/status', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_FASTAPI_URL}/health`);
    const { active_cameras } = response.data;
    
    res.json({
      online: active_cameras.includes(req.params.cameraId),
      active_cameras: active_cameras
    });
    
  } catch (error) {
    res.status(503).json({
      success: false,
      error: 'Python service unavailable'
    });
  }
});

module.exports = router;