/**
 * Driver Alert Beep Component
 *
 * Shows visual alert + plays beep sound when high risk detected
 * For driver dashboard notification
 */

import { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';

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
};

const font = {
  mono: "'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
  display: "'Inter', 'Helvetica Neue', Arial, sans-serif",
};

const DriverAlertBeep = ({ backendUrl = 'http://localhost:8000' }) => {
  const [alert, setAlert] = useState(null);
  const [isVisible, setIsVisible] = useState(false);
  const audioRef = useRef(null);
  const socketRef = useRef(null);

  useEffect(() => {
    // Connect to Socket.IO for real-time alerts
    console.log('Connecting to Socket.IO...');
    const socket = io(backendUrl, {
      transports: ['websocket', 'polling']
    });

    socket.on('connect', () => {
      console.log('✓ Socket.IO connected for driver alerts');
    });

    socket.on('driver_alert', (alertData) => {
      console.log('🚨 Driver alert received:', alertData);
      handleAlert(alertData);
    });

    socket.on('disconnect', () => {
      console.log('✗ Socket.IO disconnected');
    });

    socket.on('error', (error) => {
      console.error('Socket.IO error:', error);
    });

    socketRef.current = socket;

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [backendUrl]);

  const handleAlert = (alertData) => {
    setAlert(alertData);
    setIsVisible(true);

    // Play beep sound
    if (audioRef.current) {
      audioRef.current.play().catch(e => console.error('Audio play error:', e));
    }

    // Auto-dismiss after 10 seconds
    setTimeout(() => {
      setIsVisible(false);
    }, 10000);
  };

  const dismissAlert = () => {
    setIsVisible(false);
  };

  if (!isVisible || !alert) {
    return null;
  }

  const getRiskColor = () => {
    if (alert.risk_score >= 0.95) return palette.accent; // Red - Emergency
    if (alert.risk_score >= 0.80) return palette.critical; // Orange - Critical
    return palette.warn; // Yellow - Warning
  };

  return (
    <>
      {/* Beep Sound */}
      <audio
        ref={audioRef}
        src="/sounds/driver_alert_beep.wav"
        preload="auto"
      />

      {/* Visual Alert Overlay */}
      <div
        style={{
          position: 'fixed',
          top: 20,
          right: 20,
          zIndex: 9999,
          maxWidth: 400,
          backgroundColor: palette.surface,
          border: `2px solid ${getRiskColor()}`,
          borderRadius: 0, // Sharp edges for brutalist look
          padding: 20,
          boxShadow: `0 0 20px ${getRiskColor()}40, 0 8px 16px rgba(0,0,0,0.8)`,
          animation: 'pulse 1s ease-in-out infinite',
          fontFamily: font.mono,
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 8,
                height: 8,
                backgroundColor: getRiskColor(),
                borderRadius: 0, // Square for brutalist look
                animation: 'blink 0.5s ease-in-out infinite',
              }}
            />
            <span
              style={{
                fontFamily: font.mono,
                fontSize: '0.8rem',
                color: getRiskColor(),
                fontWeight: 700,
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
              }}
            >
              {alert.risk_score >= 0.95 ? '🚨 EMERGENCY' : '⚠️ DRIVER ALERT'}
            </span>
          </div>
          <button
            onClick={dismissAlert}
            style={{
              background: 'transparent',
              border: `1px solid ${palette.border}`,
              color: palette.text,
              cursor: 'pointer',
              padding: '4px 12px',
              fontSize: '0.65rem',
              fontFamily: font.mono,
              borderRadius: 0,
              letterSpacing: '0.1em',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.target.style.background = palette.surfaceHover;
              e.target.style.borderColor = palette.textMuted;
            }}
            onMouseLeave={(e) => {
              e.target.style.background = 'transparent';
              e.target.style.borderColor = palette.border;
            }}
          >
            DISMISS
          </button>
        </div>

        {/* Alert Content */}
        <div style={{ marginBottom: 12 }}>
          <div style={{
            fontSize: '1rem',
            fontWeight: 600,
            color: palette.text,
            marginBottom: 8,
            fontFamily: font.mono,
            letterSpacing: '0.02em',
          }}>
            {alert.alert_type || 'Person at Platform Edge'}
          </div>
          <div style={{
            fontSize: '0.75rem',
            color: palette.textMuted,
            marginBottom: 12,
            fontFamily: font.mono,
            lineHeight: '1.4',
          }}>
            {alert.description || 'Person detected dangerously close to platform edge'}
          </div>

          {/* Risk Score Bar */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{
                fontSize: '0.65rem',
                color: palette.textMuted,
                fontFamily: font.mono,
                letterSpacing: '0.1em',
              }}>
                RISK SCORE
              </span>
              <span style={{
                fontSize: '0.75rem',
                color: getRiskColor(),
                fontFamily: font.mono,
                fontWeight: 700,
                letterSpacing: '0.05em',
              }}>
                {(alert.risk_score * 100).toFixed(0)}%
              </span>
            </div>
            <div style={{
              width: '100%',
              height: 6,
              backgroundColor: palette.border,
              borderRadius: 0,
              overflow: 'hidden',
              border: `1px solid ${palette.border}`,
            }}>
              <div
                style={{
                  width: `${alert.risk_score * 100}%`,
                  height: '100%',
                  backgroundColor: getRiskColor(),
                  transition: 'width 0.3s ease',
                }}
              />
            </div>
          </div>

          {/* Location Info */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 10,
              fontSize: '0.65rem',
              fontFamily: font.mono,
            }}
          >
            <div>
              <div style={{
                color: palette.textMuted,
                marginBottom: 3,
                letterSpacing: '0.08em',
                fontSize: '0.6rem',
              }}>CAMERA</div>
              <div style={{
                color: palette.text,
                fontWeight: 600,
                letterSpacing: '0.02em',
              }}>{alert.camera_id || 'N/A'}</div>
            </div>
            <div>
              <div style={{
                color: palette.textMuted,
                marginBottom: 3,
                letterSpacing: '0.08em',
                fontSize: '0.6rem',
              }}>TRACK ID</div>
              <div style={{
                color: palette.text,
                fontWeight: 600,
                letterSpacing: '0.02em',
              }}>#{alert.track_id || 'N/A'}</div>
            </div>
          </div>
        </div>

        {/* Action Button */}
        {alert.risk_score >= 0.95 && (
          <button
            onClick={() => alert('Emergency protocol initiated')}
            style={{
              width: '100%',
              padding: '12px',
              backgroundColor: getRiskColor(),
              color: palette.text,
              border: `2px solid ${getRiskColor()}`,
              borderRadius: 0,
              cursor: 'pointer',
              fontFamily: font.mono,
              fontSize: '0.7rem',
              fontWeight: 700,
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = 'transparent';
              e.target.style.color = getRiskColor();
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = getRiskColor();
              e.target.style.color = palette.text;
            }}
          >
            INITIATE EMERGENCY STOP
          </button>
        )}
      </div>

      {/* CSS Animations + Retro CRT Effect */}
      <style>{`
        @keyframes pulse {
          0%, 100% {
            transform: scale(1) translateY(0);
            box-shadow: 0 0 20px ${getRiskColor()}40, 0 8px 16px rgba(0,0,0,0.8);
          }
          50% {
            transform: scale(1.005) translateY(-1px);
            box-shadow: 0 0 30px ${getRiskColor()}60, 0 10px 20px rgba(0,0,0,0.9);
          }
        }

        @keyframes blink {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.2;
          }
        }

        /* Retro CRT scanline effect - subtle overlay */
        @keyframes scanline {
          0% {
            transform: translateY(-100%);
          }
          100% {
            transform: translateY(100%);
          }
        }
      `}</style>
    </>
  );
};

export default DriverAlertBeep;
