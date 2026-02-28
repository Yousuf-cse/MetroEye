const express = require('express');
const router = express.Router();
const alertController = require('../controllers/alertController');

/**
 * ALERT CREATION ROUTES
 * Three pathways for creating alerts based on processing requirements
 */

// MAIN ROUTE: Create alert from detection (auto-routing based on threshold)
// Vision engine sends detection data, backend decides fast-path vs LLM-path
router.post('/from-detection', alertController.createAlertFromDetection);

// LLM ROUTE: Create alert from pre-analyzed LLM data
// Vision engine has already processed through LLM, just save to DB
router.post('/from-llm', alertController.createAlertFromLLM);

// FAST-PATH ROUTE: Emergency bypass for critical alerts
// Skips all processing, direct to database and consumers
router.post('/fast-path', alertController.createFastPathAlert);

// LEGACY: Create new alert (AI system calls this) - backwards compatibility
router.post('/', alertController.createAlert);

/**
 * ALERT RETRIEVAL ROUTES
 */

// Get all alerts (with filters and pagination)
// Supports: status, risk_level, camera_id, start_date, end_date
router.get('/', alertController.getAlerts);

// Get statistics (dashboard metrics)
router.get('/stats', alertController.getStats);

// Get single alert by ID
router.get('/:id', alertController.getAlertById);

/**
 * ALERT UPDATE ROUTES
 */

// Update alert status (operator actions)
router.patch('/:id/status', alertController.updateAlertStatus);

// Delete alert
router.delete('/:id', alertController.deleteAlert);

module.exports = router;
