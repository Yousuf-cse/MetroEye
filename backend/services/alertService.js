const axios = require('axios');
const Alert = require('../models/Alert');
const { v4: uuidv4 } = require('uuid');

// Configuration
const PYTHON_VISION_ENGINE_URL = process.env.PYTHON_URL || 'http://localhost:5000';
const HIGH_RISK_THRESHOLD = parseInt(process.env.HIGH_RISK_THRESHOLD) || 85; // Skip LLM if score >= 85
const MEDIUM_RISK_THRESHOLD = parseInt(process.env.MEDIUM_RISK_THRESHOLD) || 50;

/**
 * Alert Processing Service
 *
 * This service implements a two-path alert system:
 * 1. FAST PATH: High-risk detections (>= 85) bypass LLM and go directly to consumers
 * 2. LLM PATH: Medium-risk detections (50-84) go through LLM for detailed analysis
 */
class AlertService {
  /**
   * Process incoming alert from vision engine
   * Routes to fast-path or LLM-path based on risk score
   *
   * @param {Object} detectionData - Raw detection data from vision engine
   * @param {Object} io - Socket.io instance for real-time broadcast
   * @returns {Object} Created alert
   */
  async processDetection(detectionData, io) {
    const { track_id, camera_id, risk_score, risk_level, features, timestamp } = detectionData;

    console.log(`📊 Processing detection - Track: ${track_id}, Risk: ${risk_score}, Level: ${risk_level}`);

    // FAST PATH: Critical/high risk - bypass LLM for immediate alert
    if (risk_score >= HIGH_RISK_THRESHOLD) {
      console.log(`⚡ FAST PATH: Risk score ${risk_score} >= ${HIGH_RISK_THRESHOLD}, bypassing LLM`);
      return await this.createFastPathAlert(detectionData, io);
    }

    // MEDIUM PATH: Send to LLM for detailed analysis
    if (risk_score >= MEDIUM_RISK_THRESHOLD) {
      console.log(`🧠 LLM PATH: Risk score ${risk_score} in range [${MEDIUM_RISK_THRESHOLD}, ${HIGH_RISK_THRESHOLD}), using LLM`);
      return await this.createLLMPathAlert(detectionData, io);
    }

    // Below threshold - no alert created
    console.log(`✓ Risk score ${risk_score} below threshold ${MEDIUM_RISK_THRESHOLD}, no alert created`);
    return null;
  }

  /**
   * FAST PATH: Create immediate alert without LLM processing
   * Used for critical situations requiring instant response
   *
   * @param {Object} detectionData - Detection data
   * @param {Object} io - Socket.io instance
   * @returns {Object} Created alert
   */
  async createFastPathAlert(detectionData, io) {
    const { track_id, camera_id, risk_score, risk_level, features, timestamp } = detectionData;

    // Generate simple rule-based message
    const alertMessage = this._generateFastPathMessage(track_id, camera_id, risk_score, features);
    const recommendedAction = this._determineAction(risk_score);

    // Create alert document
    const alertData = {
      alert_id: `ALERT-${uuidv4()}`,
      timestamp: timestamp ? new Date(timestamp * 1000) : new Date(),
      camera_id,
      track_id,
      risk_level: risk_level || this._getRiskLevel(risk_score),
      confidence: 0.95, // High confidence for rule-based fast path
      risk_score,
      features: features || {},
      llm_reasoning: 'FAST PATH: Immediate alert generated without LLM analysis due to critical risk score',
      alert_message: alertMessage,
      recommended_action: recommendedAction,
      status: 'pending'
    };

    const alert = new Alert(alertData);
    await alert.save();

    console.log(`✅ FAST PATH alert created: ${alert.alert_id}`);

    // Broadcast immediately via WebSocket
    if (io) {
      io.emit('new_alert', {
        type: 'new_alert',
        priority: 'critical',
        data: alert.toObject()
      });
    }

    return alert;
  }

  /**
   * LLM PATH: Process alert through LLM for detailed reasoning
   * Used for medium-risk detections that need contextual analysis
   *
   * @param {Object} detectionData - Detection data
   * @param {Object} io - Socket.io instance
   * @returns {Object} Created alert with LLM reasoning
   */
  async createLLMPathAlert(detectionData, io) {
    const { track_id, camera_id, risk_score, risk_level, features, timestamp } = detectionData;

    try {
      // Call Python vision engine's LLM analyzer
      console.log(`📡 Requesting LLM analysis from ${PYTHON_VISION_ENGINE_URL}/api/analyze-alert`);

      const llmResponse = await axios.post(
        `${PYTHON_VISION_ENGINE_URL}/api/analyze-alert`,
        {
          track_id,
          camera_id,
          risk_score,
          features: features || {}
        },
        { timeout: 8000 } // 8 second timeout for LLM
      );

      const llmResult = llmResponse.data;

      // Create alert with LLM reasoning
      const alertData = {
        alert_id: `ALERT-${uuidv4()}`,
        timestamp: timestamp ? new Date(timestamp * 1000) : new Date(),
        camera_id,
        track_id,
        risk_level: llmResult.risk_level || risk_level || this._getRiskLevel(risk_score),
        confidence: llmResult.confidence || 0.75,
        risk_score,
        features: features || {},
        llm_reasoning: llmResult.reasoning || 'LLM analysis completed',
        alert_message: llmResult.alert_message || this._generateFastPathMessage(track_id, camera_id, risk_score, features),
        recommended_action: llmResult.recommended_action || this._determineAction(risk_score),
        status: 'pending'
      };

      const alert = new Alert(alertData);
      await alert.save();

      console.log(`✅ LLM PATH alert created: ${alert.alert_id}`);

      // Broadcast via WebSocket
      if (io) {
        io.emit('new_alert', {
          type: 'new_alert',
          priority: 'normal',
          data: alert.toObject()
        });
      }

      return alert;

    } catch (error) {
      console.error(`⚠️ LLM processing failed: ${error.message}`);
      console.log(`🔄 Falling back to fast path for track ${track_id}`);

      // Fallback to fast path if LLM fails
      return await this.createFastPathAlert(detectionData, io);
    }
  }

  /**
   * Create alert directly from LLM-analyzed data
   * Used when vision engine has already processed through LLM
   *
   * @param {Object} llmAlertData - Alert data with LLM reasoning
   * @param {Object} io - Socket.io instance
   * @returns {Object} Created alert
   */
  async createFromLLM(llmAlertData, io) {
    const {
      track_id,
      camera_id,
      risk_score,
      risk_level,
      confidence,
      features,
      reasoning,
      alert_message,
      recommended_action,
      timestamp
    } = llmAlertData;

    const alertData = {
      alert_id: `ALERT-${uuidv4()}`,
      timestamp: timestamp ? new Date(timestamp * 1000) : new Date(),
      camera_id,
      track_id,
      risk_level,
      confidence: confidence || 0.75,
      risk_score,
      features: features || {},
      llm_reasoning: reasoning || 'LLM analysis completed',
      alert_message: alert_message || `Alert for track #${track_id}`,
      recommended_action: recommended_action || this._determineAction(risk_score),
      status: 'pending'
    };

    const alert = new Alert(alertData);
    await alert.save();

    console.log(`✅ LLM alert created: ${alert.alert_id}`);

    // Broadcast via WebSocket
    if (io) {
      io.emit('new_alert', {
        type: 'new_alert',
        priority: risk_level === 'critical' || risk_level === 'high' ? 'high' : 'normal',
        data: alert.toObject()
      });
    }

    return alert;
  }

  /**
   * Generate fast-path alert message (rule-based)
   * @private
   */
  _generateFastPathMessage(track_id, camera_id, risk_score, features) {
    const parts = [`CRITICAL: Person #${track_id} at ${camera_id}`];

    if (features) {
      if (features.min_dist_to_edge !== undefined && features.min_dist_to_edge < 100) {
        parts.push(`very close to edge (${features.min_dist_to_edge.toFixed(0)}px)`);
      }
      if (features.dwell_time_near_edge && features.dwell_time_near_edge > 5) {
        parts.push(`dwelling for ${features.dwell_time_near_edge.toFixed(1)}s`);
      }
      if (features.max_speed && features.max_speed > 250) {
        parts.push(`high speed movement (${features.max_speed.toFixed(0)}px/s)`);
      }
      if (features.mean_torso_angle && (features.mean_torso_angle < 75 || features.mean_torso_angle > 105)) {
        parts.push(`unusual posture (${features.mean_torso_angle.toFixed(0)}°)`);
      }
    }

    parts.push(`Risk: ${risk_score}/100. IMMEDIATE INTERVENTION REQUIRED.`);

    return parts.join(' - ');
  }

  /**
   * Determine recommended action based on risk score
   * @private
   */
  _determineAction(risk_score) {
    if (risk_score >= 90) return 'driver_alert';
    if (risk_score >= 75) return 'control_room';
    if (risk_score >= 60) return 'mic_warning';
    return 'monitor';
  }

  /**
   * Get risk level from score
   * @private
   */
  _getRiskLevel(risk_score) {
    if (risk_score >= 90) return 'critical';
    if (risk_score >= 70) return 'high';
    if (risk_score >= 40) return 'medium';
    return 'low';
  }
}

module.exports = new AlertService();
