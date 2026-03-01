// frontend/src/components/AlertList.jsx
import React, { useEffect, useState, useRef, useCallback } from 'react';
import io from 'socket.io-client';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';
const socket = io(API_BASE);

export default function AlertList() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [offset, setOffset] = useState(0);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [filterStatus, setFilterStatus] = useState('');
  const [filterRiskLevel, setFilterRiskLevel] = useState('');
  const observerRef = useRef();
  const lastAlertRef = useRef();

  const LIMIT = 20;

  const fetchAlerts = useCallback(async (resetOffset = false) => {
    if (loading) return;
    setLoading(true);

    try {
      const currentOffset = resetOffset ? 0 : offset;
      const params = new URLSearchParams({
        limit: LIMIT,
        offset: currentOffset,
        ...(filterStatus && { status: filterStatus }),
        ...(filterRiskLevel && { risk_level: filterRiskLevel }),
        ...(startDate && { start_date: new Date(startDate).toISOString() }),
        ...(endDate && { end_date: new Date(endDate).toISOString() })
      });

      const response = await fetch(`${API_BASE}/api/alerts?${params}`);
      const data = await response.json();

      if (data.success) {
        if (resetOffset) {
          setAlerts(data.data || []);
          setOffset(LIMIT);
        } else {
          setAlerts(prev => [...prev, ...(data.data || [])]);
          setOffset(prev => prev + LIMIT);
        }
        setHasMore(data.pagination?.hasMore || false);
      }
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
    } finally {
      setLoading(false);
    }
  }, [loading, offset, filterStatus, filterRiskLevel, startDate, endDate]);

  // Initial load
  useEffect(() => {
    fetchAlerts(true);
  }, [filterStatus, filterRiskLevel, startDate, endDate]);

  // Intersection Observer for infinite scroll
  useEffect(() => {
    if (loading) return;

    const observer = new IntersectionObserver(entries => {
      if (entries[0].isIntersecting && hasMore && !loading) {
        fetchAlerts();
      }
    }, { threshold: 1.0 });

    if (lastAlertRef.current) {
      observer.observe(lastAlertRef.current);
    }

    observerRef.current = observer;

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [loading, hasMore, fetchAlerts]);

  // WebSocket for real-time updates
  useEffect(() => {
    socket.on('new_alert', (data) => {
      const newAlert = data.data || data;
      setAlerts(prev => [newAlert, ...prev]);
    });

    socket.on('update_alert', (data) => {
      setAlerts(prev => prev.map(a => a._id === data.id ? { ...a, ...data } : a));
    });

    return () => {
      socket.off('new_alert');
      socket.off('update_alert');
    };
  }, []);

  const ack = async (id) => {
    try {
      await fetch(`${API_BASE}/api/alerts/${id}/status`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'handled', operator_action: 'acknowledged' })
      });
      setAlerts(prev => prev.map(a => a._id === id ? { ...a, status: 'handled' } : a));
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'critical': return '#d32f2f';
      case 'high': return '#f57c00';
      case 'medium': return '#ffa726';
      case 'low': return '#fdd835';
      default: return '#9e9e9e';
    }
  };

  const clearFilters = () => {
    setStartDate('');
    setEndDate('');
    setFilterStatus('');
    setFilterRiskLevel('');
  };

  return (
    <div style={{ padding: 20, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16 }}>
        <h2 style={{ margin: 0 }}>Live Alerts</h2>

        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <div>
            <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>Start Date:</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={{ padding: 6, borderRadius: 4, border: '1px solid #ccc' }}
            />
          </div>

          <div>
            <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>End Date:</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              style={{ padding: 6, borderRadius: 4, border: '1px solid #ccc' }}
            />
          </div>

          <div>
            <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>Status:</label>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              style={{ padding: 6, borderRadius: 4, border: '1px solid #ccc' }}
            >
              <option value="">All</option>
              <option value="pending">Pending</option>
              <option value="handled">Handled</option>
              <option value="false_alarm">False Alarm</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>Risk Level:</label>
            <select
              value={filterRiskLevel}
              onChange={(e) => setFilterRiskLevel(e.target.value)}
              style={{ padding: 6, borderRadius: 4, border: '1px solid #ccc' }}
            >
              <option value="">All</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>

          <button
            onClick={clearFilters}
            style={{ padding: '6px 12px', borderRadius: 4, border: '1px solid #ccc', background: '#fff', cursor: 'pointer', alignSelf: 'flex-end' }}
          >
            Clear Filters
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gap: 16 }}>
        {alerts.map((alert, index) => (
          <div
            key={alert._id}
            ref={index === alerts.length - 1 ? lastAlertRef : null}
            style={{
              border: `2px solid ${getRiskColor(alert.risk_level)}`,
              padding: 16,
              borderRadius: 8,
              background: '#fff',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
                  <strong style={{ fontSize: 16 }}>{alert.camera_id}</strong>
                  <span style={{ fontSize: 14, color: '#666' }}>Track #{alert.track_id}</span>
                  <span style={{
                    padding: '2px 8px',
                    borderRadius: 4,
                    background: getRiskColor(alert.risk_level),
                    color: '#fff',
                    fontSize: 12,
                    fontWeight: 'bold'
                  }}>
                    {alert.risk_level?.toUpperCase() || 'UNKNOWN'}
                  </span>
                  <span style={{ fontSize: 14, fontWeight: 'bold', color: getRiskColor(alert.risk_level) }}>
                    {alert.risk_score}/100
                  </span>
                </div>
                <div style={{ fontSize: 14, color: '#666' }}>
                  {new Date(alert.timestamp).toLocaleString()}
                </div>
              </div>

              <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                <span style={{
                  padding: '4px 8px',
                  borderRadius: 4,
                  background: alert.status === 'pending' ? '#fff3cd' : alert.status === 'handled' ? '#d4edda' : '#f8d7da',
                  fontSize: 12,
                  border: '1px solid #ddd'
                }}>
                  {alert.status}
                </span>
                {alert.status === 'pending' && (
                  <button
                    onClick={() => ack(alert._id)}
                    style={{
                      padding: '4px 12px',
                      borderRadius: 4,
                      border: 'none',
                      background: '#28a745',
                      color: '#fff',
                      cursor: 'pointer',
                      fontSize: 12
                    }}
                  >
                    Acknowledge
                  </button>
                )}
              </div>
            </div>

            <div style={{ marginBottom: 12 }}>
              <div style={{ fontWeight: 'bold', marginBottom: 4 }}>Alert Message:</div>
              <div style={{ fontSize: 14 }}>{alert.alert_message || 'No message'}</div>
            </div>

            {alert.llm_reasoning && alert.llm_reasoning !== 'Alert processed by vision engine' && (
              <div style={{ marginBottom: 12, padding: 12, background: '#f8f9fa', borderRadius: 4 }}>
                <div style={{ fontWeight: 'bold', marginBottom: 4 }}>LLM Analysis:</div>
                <div style={{ fontSize: 13 }}>{alert.llm_reasoning}</div>
              </div>
            )}

            <div style={{ marginBottom: 12 }}>
              <div style={{ fontWeight: 'bold', marginBottom: 4 }}>Recommended Action:</div>
              <div style={{ fontSize: 14 }}>{alert.recommended_action || 'monitor'}</div>
            </div>

            {alert.features && Object.keys(alert.features).length > 0 && (
              <div style={{ marginTop: 12, padding: 12, background: '#f8f9fa', borderRadius: 4 }}>
                <div style={{ fontWeight: 'bold', marginBottom: 8 }}>Behavioral Features:</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 8, fontSize: 13 }}>
                  {Object.entries(alert.features).map(([key, value]) => (
                    <div key={key}>
                      <span style={{ color: '#666' }}>{key.replace(/_/g, ' ')}:</span>{' '}
                      <strong>{typeof value === 'number' ? value.toFixed(2) : value}</strong>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {alert.video_clip_url && (
              <div style={{ marginTop: 12 }}>
                <a
                  href={alert.video_clip_url}
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: '#007bff', textDecoration: 'none' }}
                >
                  View Video Clip →
                </a>
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div style={{ textAlign: 'center', padding: 20, color: '#666' }}>
            Loading more alerts...
          </div>
        )}

        {!loading && !hasMore && alerts.length > 0 && (
          <div style={{ textAlign: 'center', padding: 20, color: '#666' }}>
            No more alerts to load
          </div>
        )}

        {!loading && alerts.length === 0 && (
          <div style={{ textAlign: 'center', padding: 40, color: '#666' }}>
            No alerts found
          </div>
        )}
      </div>
    </div>
  );
}