const express = require('express');
const router = express.Router();
const alertController = require('../controllers/alertController');

// Create new alert (AI system calls this)
router.post('/', alertController.createAlert);

// Get all alerts (with filters)
router.get('/', alertController.getAlerts);

// Get statistics
router.get('/stats', alertController.getStats);

// Get single alert
router.get('/:id', alertController.getAlertById);

// Update alert status
router.patch('/:id/status', alertController.updateAlertStatus);

// Delete alert
router.delete('/:id', alertController.deleteAlert);

module.exports = router;