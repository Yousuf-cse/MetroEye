// frontend/src/components/AlertList.jsx
import React, { useEffect, useState, useRef, useCallback } from 'react';
import io from 'socket.io-client';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';
const socket = io(API_BASE);

// Retro/Brutalist theme matching the calibration UI
const palette = {
  bg: '#101014',
  surface: '#18181c',
  surfaceHover: '#1f1f25',
  border: '#2a2a32',
  borderActive: '#e04040',
  text: '#e8e8ec',
  textMuted: '#6b6b78',
  accent: '#e04040',
  safe: '#34d399',
  warn: '#fbbf24',
  critical: '#ea580c',
  info: '#3b82f6',
};

const font = {
  mono: "'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
  display: "'Inter', 'Helvetica Neue', Arial, sans-serif",
};

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
      case 'critical': return palette.accent;
      case 'high': return palette.critical;
      case 'medium': return palette.warn;
      case 'low': return palette.safe;
      default: return palette.textMuted;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'pending': return { bg: `${palette.warn}20`, border: palette.warn, text: palette.warn };
      case 'handled': return { bg: `${palette.safe}20`, border: palette.safe, text: palette.safe };
      case 'false_alarm': return { bg: `${palette.textMuted}20`, border: palette.textMuted, text: palette.textMuted };
      default: return { bg: `${palette.border}`, border: palette.border, text: palette.textMuted };
    }
  };

  const clearFilters = () => {
    setStartDate('');
    setEndDate('');
    setFilterStatus('');
    setFilterRiskLevel('');
  };

  return (
    <div style={{
      padding: 20,
      maxWidth: 1400,
      margin: '0 auto',
      background: palette.bg,
      minHeight: '100vh',
    }}>
      <div style={{
        marginBottom: 20,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: 16,
        padding: '16px 0',
        borderBottom: `2px solid ${palette.border}`,
      }}>
        <h2 style={{
          margin: 0,
          fontFamily: font.mono,
          fontSize: '1.5rem',
          color: palette.text,
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
        }}>🚨 Live Alerts</h2>

        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div>
            <label style={{
              fontSize: '0.65rem',
              display: 'block',
              marginBottom: 6,
              fontFamily: font.mono,
              color: palette.textMuted,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
            }}>Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: 0,
                border: `1px solid ${palette.border}`,
                background: palette.surface,
                color: palette.text,
                fontFamily: font.mono,
                fontSize: '0.75rem',
              }}
            />
          </div>

          <div>
            <label style={{
              fontSize: '0.65rem',
              display: 'block',
              marginBottom: 6,
              fontFamily: font.mono,
              color: palette.textMuted,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
            }}>End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: 0,
                border: `1px solid ${palette.border}`,
                background: palette.surface,
                color: palette.text,
                fontFamily: font.mono,
                fontSize: '0.75rem',
              }}
            />
          </div>

          <div>
            <label style={{
              fontSize: '0.65rem',
              display: 'block',
              marginBottom: 6,
              fontFamily: font.mono,
              color: palette.textMuted,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
            }}>Status</label>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: 0,
                border: `1px solid ${palette.border}`,
                background: palette.surface,
                color: palette.text,
                fontFamily: font.mono,
                fontSize: '0.75rem',
                cursor: 'pointer',
              }}
            >
              <option value="">All</option>
              <option value="pending">Pending</option>
              <option value="handled">Handled</option>
              <option value="false_alarm">False Alarm</option>
            </select>
          </div>

          <div>
            <label style={{
              fontSize: '0.65rem',
              display: 'block',
              marginBottom: 6,
              fontFamily: font.mono,
              color: palette.textMuted,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
            }}>Risk Level</label>
            <select
              value={filterRiskLevel}
              onChange={(e) => setFilterRiskLevel(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: 0,
                border: `1px solid ${palette.border}`,
                background: palette.surface,
                color: palette.text,
                fontFamily: font.mono,
                fontSize: '0.75rem',
                cursor: 'pointer',
              }}
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
            style={{
              padding: '8px 16px',
              borderRadius: 0,
              border: `1px solid ${palette.border}`,
              background: palette.surface,
              color: palette.text,
              cursor: 'pointer',
              fontFamily: font.mono,
              fontSize: '0.7rem',
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.target.style.background = palette.surfaceHover;
              e.target.style.borderColor = palette.textMuted;
            }}
            onMouseLeave={(e) => {
              e.target.style.background = palette.surface;
              e.target.style.borderColor = palette.border;
            }}
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
              padding: 20,
              borderRadius: 0,
              background: palette.surface,
              boxShadow: `0 0 20px ${getRiskColor(alert.risk_level)}20, 0 4px 12px rgba(0,0,0,0.6)`,
              transition: 'all 0.2s ease',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
                  <strong style={{
                    fontSize: '1rem',
                    fontFamily: font.mono,
                    color: palette.text,
                    letterSpacing: '0.05em',
                  }}>📹 {alert.camera_id}</strong>
                  <span style={{
                    fontSize: '0.75rem',
                    fontFamily: font.mono,
                    color: palette.textMuted,
                  }}>Track #{alert.track_id}</span>
                  <span style={{
                    padding: '4px 10px',
                    borderRadius: 0,
                    background: getRiskColor(alert.risk_level),
                    color: palette.text,
                    fontSize: '0.65rem',
                    fontWeight: 700,
                    fontFamily: font.mono,
                    letterSpacing: '0.1em',
                  }}>
                    {alert.risk_level?.toUpperCase() || 'UNKNOWN'}
                  </span>
                  <span style={{
                    fontSize: '0.8rem',
                    fontWeight: 700,
                    fontFamily: font.mono,
                    color: getRiskColor(alert.risk_level),
                  }}>
                    {alert.risk_score}/100
                  </span>
                </div>
                <div style={{
                  fontSize: '0.7rem',
                  fontFamily: font.mono,
                  color: palette.textMuted,
                  letterSpacing: '0.02em',
                }}>
                  {new Date(alert.timestamp).toLocaleString()}
                </div>
              </div>

              <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                {(() => {
                  const statusColors = getStatusColor(alert.status);
                  return (
                    <span style={{
                      padding: '6px 12px',
                      borderRadius: 0,
                      background: statusColors.bg,
                      fontSize: '0.65rem',
                      border: `1px solid ${statusColors.border}`,
                      color: statusColors.text,
                      fontFamily: font.mono,
                      letterSpacing: '0.08em',
                      textTransform: 'uppercase',
                      fontWeight: 600,
                    }}>
                      {alert.status}
                    </span>
                  );
                })()}
                {alert.status === 'pending' && (
                  <button
                    onClick={() => ack(alert._id)}
                    style={{
                      padding: '6px 16px',
                      borderRadius: 0,
                      border: `1px solid ${palette.safe}`,
                      background: palette.safe,
                      color: palette.bg,
                      cursor: 'pointer',
                      fontSize: '0.65rem',
                      fontFamily: font.mono,
                      letterSpacing: '0.1em',
                      textTransform: 'uppercase',
                      fontWeight: 700,
                      transition: 'all 0.15s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.target.style.background = 'transparent';
                      e.target.style.color = palette.safe;
                    }}
                    onMouseLeave={(e) => {
                      e.target.style.background = palette.safe;
                      e.target.style.color = palette.bg;
                    }}
                  >
                    Acknowledge
                  </button>
                )}
              </div>
            </div>

            <div style={{ marginBottom: 16 }}>
              <div style={{
                fontWeight: 700,
                marginBottom: 6,
                fontFamily: font.mono,
                fontSize: '0.65rem',
                color: palette.textMuted,
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
              }}>Alert Message</div>
              <div style={{
                fontSize: '0.85rem',
                fontFamily: font.mono,
                color: palette.text,
                lineHeight: '1.5',
              }}>{alert.alert_message || 'No message'}</div>
            </div>

            {alert.llm_reasoning && alert.llm_reasoning !== 'Alert processed by vision engine' && (
              <div style={{
                marginBottom: 16,
                padding: 14,
                background: palette.bg,
                borderRadius: 0,
                border: `1px solid ${palette.border}`,
              }}>
                <div style={{
                  fontWeight: 700,
                  marginBottom: 6,
                  fontFamily: font.mono,
                  fontSize: '0.65rem',
                  color: palette.info,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                }}>🤖 LLM Analysis</div>
                <div style={{
                  fontSize: '0.75rem',
                  fontFamily: font.mono,
                  color: palette.text,
                  lineHeight: '1.6',
                }}>{alert.llm_reasoning}</div>
              </div>
            )}

            <div style={{ marginBottom: 16 }}>
              <div style={{
                fontWeight: 700,
                marginBottom: 6,
                fontFamily: font.mono,
                fontSize: '0.65rem',
                color: palette.textMuted,
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
              }}>Recommended Action</div>
              <div style={{
                fontSize: '0.8rem',
                fontFamily: font.mono,
                color: palette.warn,
                fontWeight: 600,
              }}>{alert.recommended_action || 'monitor'}</div>
            </div>

            {alert.features && Object.keys(alert.features).length > 0 && (
              <div style={{
                marginTop: 16,
                padding: 14,
                background: palette.bg,
                borderRadius: 0,
                border: `1px solid ${palette.border}`,
              }}>
                <div style={{
                  fontWeight: 700,
                  marginBottom: 10,
                  fontFamily: font.mono,
                  fontSize: '0.65rem',
                  color: palette.textMuted,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                }}>📊 Behavioral Features</div>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: 10,
                  fontSize: '0.75rem',
                }}>
                  {Object.entries(alert.features).map(([key, value]) => (
                    <div key={key} style={{ fontFamily: font.mono }}>
                      <span style={{ color: palette.textMuted }}>
                        {key.replace(/_/g, ' ')}:
                      </span>{' '}
                      <strong style={{ color: palette.text }}>
                        {typeof value === 'number' ? value.toFixed(2) : value}
                      </strong>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {alert.video_clip_url && (
              <div style={{ marginTop: 16 }}>
                <a
                  href={alert.video_clip_url}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    color: palette.info,
                    textDecoration: 'none',
                    fontFamily: font.mono,
                    fontSize: '0.75rem',
                    letterSpacing: '0.05em',
                    fontWeight: 600,
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '8px 12px',
                    border: `1px solid ${palette.info}`,
                    borderRadius: 0,
                    transition: 'all 0.15s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.background = palette.info;
                    e.target.style.color = palette.bg;
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.background = 'transparent';
                    e.target.style.color = palette.info;
                  }}
                >
                  🎥 View Video Clip →
                </a>
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div style={{
            textAlign: 'center',
            padding: 30,
            color: palette.textMuted,
            fontFamily: font.mono,
            fontSize: '0.75rem',
            letterSpacing: '0.08em',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 12,
          }}>
            <div style={{
              width: 20,
              height: 20,
              border: `2px solid ${palette.border}`,
              borderTopColor: palette.info,
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }} />
            Loading more alerts...
          </div>
        )}

        {!loading && !hasMore && alerts.length > 0 && (
          <div style={{
            textAlign: 'center',
            padding: 30,
            color: palette.textMuted,
            fontFamily: font.mono,
            fontSize: '0.7rem',
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            borderTop: `1px solid ${palette.border}`,
          }}>
            ─── End of alerts ───
          </div>
        )}

        {!loading && alerts.length === 0 && (
          <div style={{
            textAlign: 'center',
            padding: 60,
            color: palette.textMuted,
            fontFamily: font.mono,
            fontSize: '0.85rem',
            letterSpacing: '0.05em',
          }}>
            <div style={{ fontSize: '3rem', marginBottom: 16, opacity: 0.3 }}>📭</div>
            <div>No alerts found</div>
          </div>
        )}
      </div>

      {/* CSS Animations */}
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}