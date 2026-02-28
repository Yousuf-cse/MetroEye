const Alert = require('../models/Alert');

// CREATE NEW ALERT (from AI system)
exports.createAlert = async (req, res) => {
  try {
    const alertData = req.body;
    
    // Create new alert document
    const alert = new Alert(alertData);
    await alert.save();
    
    console.log(`✅ New alert saved: ${alert.alert_id}`);
    
    // Emit to all connected WebSocket clients
    req.io.emit('new_alert', {
      type: 'new_alert',
      data: {
        ...alert.toObject(),
        video_clip_url: alert.video_clip_path 
          ? `/api/videos/${alert.video_clip_path}` 
          : null
      }
    });
    
    res.status(201).json({
      success: true,
      data: alert
    });
    
  } catch (error) {
    console.error('Error creating alert:', error);
    res.status(400).json({
      success: false,
      error: error.message
    });
  }
};

// GET ALL ALERTS (with filters and pagination)
exports.getAlerts = async (req, res) => {
  try {
    const {
      limit = 50,
      offset = 0,
      status,
      risk_level,
      camera_id,
      start_date,
      end_date
    } = req.query;
    
    // Build filter object
    const filter = {};
    if (status) filter.status = status;
    if (risk_level) filter.risk_level = risk_level;
    if (camera_id) filter.camera_id = camera_id;
    
    // Date range filter
    if (start_date || end_date) {
      filter.timestamp = {};
      if (start_date) filter.timestamp.$gte = new Date(start_date);
      if (end_date) filter.timestamp.$lte = new Date(end_date);
    }
    
    // Execute query with pagination
    const alerts = await Alert.find(filter)
      .sort({ timestamp: -1 }) // Most recent first
      .skip(parseInt(offset))
      .limit(parseInt(limit));
    
    // Get total count for pagination
    const total = await Alert.countDocuments(filter);
    
    res.json({
      success: true,
      data: alerts.map(alert => ({
        ...alert.toObject(),
        video_clip_url: alert.video_clip_path 
          ? `/api/videos/${alert.video_clip_path}` 
          : null
      })),
      pagination: {
        total,
        limit: parseInt(limit),
        offset: parseInt(offset),
        hasMore: total > parseInt(offset) + parseInt(limit)
      }
    });
    
  } catch (error) {
    console.error('Error fetching alerts:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// GET SINGLE ALERT BY ID
exports.getAlertById = async (req, res) => {
  try {
    const alert = await Alert.findById(req.params.id);
    
    if (!alert) {
      return res.status(404).json({
        success: false,
        error: 'Alert not found'
      });
    }
    
    res.json({
      success: true,
      data: {
        ...alert.toObject(),
        video_clip_url: alert.video_clip_path 
          ? `/api/videos/${alert.video_clip_path}` 
          : null
      }
    });
    
  } catch (error) {
    console.error('Error fetching alert:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// UPDATE ALERT STATUS
exports.updateAlertStatus = async (req, res) => {
  try {
    const { status, operator_action } = req.body;
    
    const updateData = { status };
    if (operator_action) updateData.operator_action = operator_action;
    if (status === 'handled' || status === 'false_alarm') {
      updateData.resolved_at = new Date();
    }
    
    const alert = await Alert.findByIdAndUpdate(
      req.params.id,
      updateData,
      { new: true } // Return updated document
    );
    
    if (!alert) {
      return res.status(404).json({
        success: false,
        error: 'Alert not found'
      });
    }
    
    console.log(`✅ Alert ${alert._id} status updated to: ${status}`);
    
    // Broadcast update to WebSocket clients
    req.io.emit('alert_updated', {
      type: 'alert_updated',
      data: {
        id: alert._id,
        status: alert.status,
        operator_action: alert.operator_action,
        resolved_at: alert.resolved_at
      }
    });
    
    res.json({
      success: true,
      data: alert
    });
    
  } catch (error) {
    console.error('Error updating alert:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// DELETE ALERT
exports.deleteAlert = async (req, res) => {
  try {
    const alert = await Alert.findByIdAndDelete(req.params.id);
    
    if (!alert) {
      return res.status(404).json({
        success: false,
        error: 'Alert not found'
      });
    }
    
    res.json({
      success: true,
      message: 'Alert deleted successfully'
    });
    
  } catch (error) {
    console.error('Error deleting alert:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// GET STATISTICS
exports.getStats = async (req, res) => {
  try {
    // Total alerts
    const total = await Alert.countDocuments();
    
    // Group by status
    const byStatus = await Alert.aggregate([
      {
        $group: {
          _id: '$status',
          count: { $sum: 1 }
        }
      }
    ]);
    
    // Group by risk level
    const byRiskLevel = await Alert.aggregate([
      {
        $group: {
          _id: '$risk_level',
          count: { $sum: 1 }
        }
      }
    ]);
    
    // Alerts in last 24 hours
    const last24h = await Alert.countDocuments({
      timestamp: { $gte: new Date(Date.now() - 24 * 60 * 60 * 1000) }
    });
    
    // False alarm rate
    const falseAlarms = await Alert.countDocuments({ status: 'false_alarm' });
    const handled = await Alert.countDocuments({ status: 'handled' });
    const falseAlarmRate = total > 0 ? (falseAlarms / total * 100).toFixed(2) : 0;
    
    res.json({
      success: true,
      data: {
        total_alerts: total,
        last_24h: last24h,
        false_alarm_rate: falseAlarmRate,
        by_status: byStatus.reduce((acc, item) => {
          acc[item._id] = item.count;
          return acc;
        }, {}),
        by_risk_level: byRiskLevel.reduce((acc, item) => {
          acc[item._id] = item.count;
          return acc;
        }, {})
      }
    });
    
  } catch (error) {
    console.error('Error fetching stats:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};