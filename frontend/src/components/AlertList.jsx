// frontend/src/components/AlertList.jsx
import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:3001';
const socket = io(API_BASE);

export default function AlertList() {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    fetch(`${API_BASE}/api/alerts`).then(r => r.json()).then(setAlerts);

    socket.on('new_alert', (alert) => {
      setAlerts((prev) => [alert, ...prev]);
    });

    socket.on('update_alert', (alert) => {
      setAlerts((prev) => prev.map(a => a.id === alert.id ? alert : a));
    });

    return () => {
      socket.off('new_alert');
      socket.off('update_alert');
    };
  }, []);

  const ack = async (id) => {
    await fetch(`${API_BASE}/api/alerts/${id}/ack`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user: 'operator_1' }) });
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Live Alerts</h2>
      <div style={{ display: 'grid', gap: 12 }}>
        {alerts.map(alert => (
          <div key={alert.id} style={{ border: '1px solid #ddd', padding: 12, borderRadius: 8, background: '#fff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>{alert.camera_id} · {alert.track_id ?? '—'}</strong>
              <div>
                <span style={{ marginRight: 8 }}>{(alert.risk_score*100).toFixed(0)}%</span>
                <span style={{ marginRight: 8 }}>{alert.status}</span>
                <button onClick={() => ack(alert.id)}>Acknowledge</button>
              </div>
            </div>
            <div style={{ marginTop: 8, fontSize: 13 }}>
              <div>Received: {new Date(alert.received_at * 1000).toLocaleString()}</div>
              <div>Features: {alert.features ? JSON.stringify(alert.features) : '—'}</div>
              {alert.video_clip_url && <div><a href={alert.video_clip_url} target="_blank" rel="noreferrer">Play clip</a></div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}