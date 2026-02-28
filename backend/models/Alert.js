const mongoose = require('mongoose');

const alertSchema = new mongoose.Schema({
  // Basic identification
  alert_id: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  timestamp: {
    type: Date,
    required: true,
    index: true
  },
  camera_id: {
    type: String,
    required: true,
    index: true
  },
  track_id: {
    type: Number,
    required: true
  },
  
  // Risk assessment
  risk_level: {
    type: String,
    enum: ['low', 'medium', 'high', 'critical'],
    required: true
  },
  confidence: {
    type: Number,
    min: 0,
    max: 1,
    required: true
  },
  risk_score: {
    type: Number,
    min: 0,
    max: 100,
    required: true
  },
  
  // Detailed information
  features: {
    type: Map,
    of: mongoose.Schema.Types.Mixed // Flexible - accepts any JSON
  },
  llm_reasoning: {
    type: String,
    required: true
  },
  alert_message: {
    type: String,
    required: true
  },
  recommended_action: {
    type: String,
    required: true
  },
  
  // Operator actions
  operator_action: {
    type: String,
    default: null
  },
  resolved_at: {
    type: Date,
    default: null
  },
  video_clip_path: {
    type: String,
    default: null
  }, 
  
  // Status tracking
  status: {
    type: String,
    enum: ['pending', 'handled', 'false_alarm'],
    default: 'pending',
    index: true
  }
}, {
  timestamps: true // Automatically adds createdAt and updatedAt
});

// Add indexes for faster queries
alertSchema.index({ timestamp: -1 });
alertSchema.index({ status: 1, timestamp: -1 });
alertSchema.index({ camera_id: 1, timestamp: -1 });

module.exports = mongoose.model('Alert', alertSchema);