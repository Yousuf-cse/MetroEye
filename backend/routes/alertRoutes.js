const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
// const fetch = require('node-fetch'); // node 18+: global fetch exists; use node-fetch if older
const { v4: uuidv4 } = require('uuid');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

app.use(express.json({ limit: '10mb' }));

// In-memory store (replace with DB in production)
const alerts = new Map();

// Thresholds (tune per deployment)
const QUICK_ALERT_THRESHOLD = 0.95; // bypass LLM/human review -> immediate escalation
const CONTROL_ROOM_THRESHOLD = 0.80; // go to control room for review

// Utility: broadcast new alert
function broadcastNewAlert(alert) {
  io.emit('new_alert', alert);
}

function broadcastUpdate(alert) {
  io.emit('update_alert', alert);
}

// Placeholder: notify control room (email/SMS/secure socket) - implement for real
async function notifyControlRoom(alert) {
  console.log('Notify control room:', alert.id, alert.risk_score);
  // e.g., POST to control-room API or send SMS
}

// Placeholder: notify driver or trigger driver alarm via rail API
async function notifyDriver(alert) {
  console.log('Notify driver (EMERGENCY):', alert.id, alert.camera_id);
  // integrate with driver system: secure API endpoint / radio / dedicated signaling channel
}

// Optional enrichment call to ai-service (synchronous). The ai-service might be your LLM that adds more context.
async function enrichAlertViaAI(alert) {
  try {
    // ai-service expected to run in ai-service:8000 or change to your host
    const resp = await fetch('http://localhost:8000/enrich_alert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(alert)
    });
    if (!resp.ok) return alert;
    const enriched = await resp.json();
    return { ...alert, enriched };
  } catch (e) {
    console.warn('AI enrichment failed', e.message);
    return alert;
  }
}

// POST /api/alerts  <-- AI/LLM or ai-service will call this
app.post('/api/alerts', async (req, res) => {
  try {
    const payload = req.body;
    // basic validation
    if (typeof payload.risk_score !== 'number' || !payload.camera_id) {
      return res.status(400).json({ error: 'invalid payload' });
    }

    const id = uuidv4();
    const now = Date.now() / 1000.0;

    const alert = {
      id,
      camera_id: payload.camera_id,
      track_id: payload.track_id ?? null,
      timestamp: payload.timestamp ?? now,
      received_at: now,
      risk_score: Number(payload.risk_score),
      features: payload.features ?? {},
      video_clip_url: payload.video_clip_url ?? null,
      source: payload.meta?.source ?? 'ai-service',
      status: 'new', // new | pending_review | escalated | acknowledged | closed
      enrichment: null,
    };

    // Quick escalation path (bypass human/LLM review)
    if (alert.risk_score >= QUICK_ALERT_THRESHOLD) {
      alert.status = 'escalated';
      alert.escalation = 'immediate';
      // store and broadcast immediately
      alerts.set(id, alert);
      broadcastNewAlert(alert);

      // trigger emergency notifications (driver + control room) in parallel
      // we await these so response indicates result - for health & audit
      try {
        await Promise.all([notifyDriver(alert), notifyControlRoom(alert)]);
      } catch (e) {
        console.error('Quick alert notifications failed:', e.message);
      }

      return res.status(201).json({ id, status: alert.status, note: 'immediate escalation sent' });
    }

    // If not immediate, but above control room threshold, send for review/enrichment
    if (alert.risk_score >= CONTROL_ROOM_THRESHOLD) {
      alert.status = 'pending_review';
      alerts.set(id, alert);
      broadcastNewAlert(alert);

      // Enrich with LLM/AI service synchronously (so UI gets enriched data quickly)
      const enriched = await enrichAlertViaAI(alert);
      alert.enrichment = enriched.enriched ?? enriched;
      alerts.set(id, alert);
      broadcastUpdate(alert);

      return res.status(201).json({ id, status: alert.status });
    }

    // Low-risk: log only, optionally return for offline review
    alert.status = 'logged';
    alerts.set(id, alert);
    broadcastNewAlert(alert);

    return res.status(201).json({ id, status: alert.status });
  } catch (err) {
    console.error('POST /api/alerts error', err);
    return res.status(500).json({ error: 'server error' });
  }
});

// GET /api/alerts - list (simple, returns all)
app.get('/api/alerts', (req, res) => {
  const all = Array.from(alerts.values()).sort((a, b) => b.received_at - a.received_at);
  res.json(all);
});

// Acknowledge an alert
app.post('/api/alerts/:id/ack', (req, res) => {
  const id = req.params.id;
  const item = alerts.get(id);
  if (!item) return res.status(404).json({ error: 'not found' });
  item.status = 'acknowledged';
  item.ack_by = req.body.user ?? 'operator';
  item.ack_at = Date.now() / 1000.0;
  alerts.set(id, item);
  broadcastUpdate(item);
  res.json({ ok: true });
});


