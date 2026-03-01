const mongoose = require('mongoose');

/**
 * Person Model - Tracks passengers across the metro system
 *
 * Uses face recognition embeddings for re-identification
 * Maintains journey history from entry to exit
 */
const personSchema = new mongoose.Schema({
  // Persistent identifier
  persistent_id: {
    type: String,
    required: true,
    unique: true,
    index: true,
    // Format: "P0001", "P0002", etc.
  },

  // Face recognition data
  face_embedding: {
    type: [Number],  // 512-dimensional array
    required: false   // May not have face detected
  },

  // Entry information
  entry_info: {
    timestamp: {
      type: Date,
      required: true,
      index: true
    },
    camera_id: {
      type: String,
      required: true
    },
    location: {
      type: String,
      default: 'Unknown'
    },
    gate_id: String  // e.g., "Gate A", "Gate 3"
  },

  // Current status
  current_status: {
    camera_id: String,
    location: String,
    last_seen: {
      type: Date,
      default: Date.now,
      index: true
    },
    is_active: {
      type: Boolean,
      default: true,
      index: true
    }
  },

  // Journey history
  // Array of locations visited with timestamps
  journey: [{
    camera_id: {
      type: String,
      required: true
    },
    location: {
      type: String,
      required: true
    },
    timestamp: {
      type: Date,
      required: true
    },
    activity: {
      type: String,
      enum: ['entered', 'moved', 'waiting', 'exited'],
      default: 'moved'
    }
  }],

  // Exit information (if exited)
  exit_info: {
    timestamp: Date,
    camera_id: String,
    location: String,
    gate_id: String
  },

  // Emotion tracking (NEW)
  emotions: {
    current: {
      emotion: {
        type: String,
        enum: ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'unknown'],
        default: 'unknown'
      },
      confidence: {
        type: Number,
        min: 0,
        max: 1,
        default: 0
      },
      timestamp: Date
    },
    dominant: {
      emotion: String,
      confidence: Number
    },
    history: [{
      emotion: String,
      confidence: Number,
      timestamp: {
        type: Date,
        required: true
      }
    }],
    emotion_alerts: {
      type: Number,
      default: 0  // Count of emotion-based alerts
    }
  },

  // Statistics
  stats: {
    total_duration_seconds: {
      type: Number,
      default: 0
    },
    platforms_visited: {
      type: [String],
      default: []
    },
    cameras_seen: {
      type: [String],
      default: []
    },
    alert_count: {
      type: Number,
      default: 0
    }
  },

  // Associated alerts (references)
  alerts: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Alert'
  }],

  // YOLO track ID mappings (per camera)
  // Format: { camera_id: yolo_track_id }
  yolo_mappings: {
    type: Map,
    of: Number,
    default: {}
  },

  // Metadata
  metadata: {
    ticket_type: String,  // If available from gate
    passenger_category: String,  // e.g., "regular", "suspicious", "VIP"
    notes: String
  }

}, {
  timestamps: true  // Adds createdAt and updatedAt
});

// Indexes for efficient queries
personSchema.index({ 'current_status.last_seen': -1 });
personSchema.index({ 'current_status.is_active': 1 });
personSchema.index({ 'entry_info.timestamp': -1 });
personSchema.index({ 'stats.alert_count': -1 });

// Virtual for journey duration
personSchema.virtual('duration').get(function() {
  if (!this.entry_info || !this.entry_info.timestamp) return 0;

  const endTime = this.exit_info && this.exit_info.timestamp
    ? this.exit_info.timestamp
    : this.current_status.last_seen || new Date();

  return (endTime - this.entry_info.timestamp) / 1000; // seconds
});

// Method: Add journey entry
personSchema.methods.addJourneyEntry = function(camera_id, location, activity = 'moved') {
  this.journey.push({
    camera_id,
    location,
    timestamp: new Date(),
    activity
  });

  // Update current status
  this.current_status.camera_id = camera_id;
  this.current_status.location = location;
  this.current_status.last_seen = new Date();

  // Update stats
  if (!this.stats.cameras_seen.includes(camera_id)) {
    this.stats.cameras_seen.push(camera_id);
  }

  if (!this.stats.platforms_visited.includes(location)) {
    this.stats.platforms_visited.push(location);
  }

  return this.save();
};

// Method: Mark as exited
personSchema.methods.markExited = function(camera_id, location, gate_id = null) {
  this.current_status.is_active = false;
  this.exit_info = {
    timestamp: new Date(),
    camera_id,
    location,
    gate_id
  };

  // Add exit to journey
  this.journey.push({
    camera_id,
    location,
    timestamp: new Date(),
    activity: 'exited'
  });

  // Calculate total duration
  if (this.entry_info && this.entry_info.timestamp) {
    this.stats.total_duration_seconds = (this.exit_info.timestamp - this.entry_info.timestamp) / 1000;
  }

  return this.save();
};

// Method: Update emotion
personSchema.methods.updateEmotion = function(emotion, confidence) {
  // Update current emotion
  this.emotions.current = {
    emotion,
    confidence,
    timestamp: new Date()
  };

  // Add to history (keep last 100)
  this.emotions.history.push({
    emotion,
    confidence,
    timestamp: new Date()
  });

  // Trim history if too long
  if (this.emotions.history.length > 100) {
    this.emotions.history = this.emotions.history.slice(-100);
  }

  // Calculate dominant emotion
  const emotionCounts = {};
  const recentHistory = this.emotions.history.slice(-30);  // Last 30 readings

  for (const entry of recentHistory) {
    if (!emotionCounts[entry.emotion]) {
      emotionCounts[entry.emotion] = { count: 0, totalConf: 0 };
    }
    emotionCounts[entry.emotion].count++;
    emotionCounts[entry.emotion].totalConf += entry.confidence;
  }

  // Find dominant
  let maxCount = 0;
  let dominantEmotion = 'unknown';
  let avgConfidence = 0;

  for (const [emotion, data] of Object.entries(emotionCounts)) {
    if (data.count > maxCount) {
      maxCount = data.count;
      dominantEmotion = emotion;
      avgConfidence = data.totalConf / data.count;
    }
  }

  this.emotions.dominant = {
    emotion: dominantEmotion,
    confidence: avgConfidence
  };

  return this.save();
};

// Method: Link alert
personSchema.methods.linkAlert = function(alertId, isEmotionBased = false) {
  if (!this.alerts.includes(alertId)) {
    this.alerts.push(alertId);
    this.stats.alert_count += 1;

    if (isEmotionBased) {
      this.emotions.emotion_alerts += 1;
    }

    // Update passenger category if multiple alerts
    if (this.stats.alert_count >= 3) {
      this.metadata.passenger_category = 'high_risk';
    } else if (this.stats.alert_count >= 1) {
      this.metadata.passenger_category = 'suspicious';
    }
  }

  return this.save();
};

// Static method: Find active passengers
personSchema.statics.findActive = function(max_age_minutes = 30) {
  const cutoff = new Date(Date.now() - max_age_minutes * 60 * 1000);

  return this.find({
    'current_status.is_active': true,
    'current_status.last_seen': { $gte: cutoff }
  });
};

// Static method: Find passengers with alerts
personSchema.statics.findWithAlerts = function() {
  return this.find({
    'stats.alert_count': { $gt: 0 }
  }).populate('alerts');
};

module.exports = mongoose.model('Person', personSchema);
