/**
 * Multi-Camera Dashboard - Monitor multiple cameras simultaneously
 *
 * Features:
 * - Grid layout with multiple camera feeds
 * - Real-time tracking data for each camera
 * - Responsive design (adapts to screen size)
 * - Connection status for all cameras
 * - Quick stats summary
 * - Red laser scan animation background
 */
import { useState, useEffect } from 'react';
import axios from 'axios';
import LiveFeed from '../components/LiveFeed';
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const palette = {
  bg: '#101014',
  surface: '#18181c',
  surfaceHover: '#1f1f25',
  border: '#2a2a32',
  borderActive: '#e04040',
  text: '#e8e8ec',
  textMuted: '#6b6b78',
  accent: '#e04040',
  accentGlow: 'rgba(224,64,64,0.25)',
  accentSoft: 'rgba(224,64,64,0.08)',
  safe: '#34d399',
  safeGlow: 'rgba(52,211,153,0.3)',
  warn: '#fbbf24',
};
const font = {
  mono: "'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
  display: "'Inter', 'Helvetica Neue', Arial, sans-serif",
};
const keyframesStyle = `
@keyframes dash-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
@keyframes slide-up {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes status-blink {
  0%, 100% { box-shadow: 0 0 0 0 rgba(52,211,153,0.5); }
  50% { box-shadow: 0 0 8px 2px rgba(52,211,153,0.35); }
}
@keyframes scanline {
  0% { transform: translateY(-100%); }
  100% { transform: translateY(100vh); }
}
@keyframes border-glow {
  0%, 100% { border-color: #2a2a32; }
  50% { border-color: #e04040; }
}
@keyframes laser-sweep {
  0% { top: -2px; opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { top: 100%; opacity: 0; }
}
@keyframes laser-horizontal {
  0% { left: -20%; opacity: 0; }
  10% { opacity: 0.6; }
  50% { opacity: 0.3; }
  90% { opacity: 0.6; }
  100% { left: 120%; opacity: 0; }
}
@keyframes laser-pulse {
  0%, 100% { opacity: 0.15; }
  50% { opacity: 0.35; }
}
`;
export default function MultiCameraDashboard() {
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState('grid');
  const [focusedCamera, setFocusedCamera] = useState(null);
  const defaultCameras = [
    { id: 'camera_1', name: 'Platform 1 — North', location: 'Gate A' },
    { id: 'camera_2', name: 'Platform 2 — South', location: 'Gate B' },
    { id: 'camera_3', name: 'Platform 3 — Central', location: 'Main Hall' },
  ];
  useEffect(() => {
    loadCameras();
  }, []);
  const loadCameras = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/cameras`);
      setCameras(response.data.cameras || defaultCameras);
    } catch (error) {
      console.log('Using default cameras');
      setCameras(defaultCameras);
    } finally {
      setLoading(false);
    }
  };
  const getGridClass = () => {
    const count = cameras.length;
    if (count === 1) return 'grid-cols-1';
    if (count === 2) return 'grid-cols-1 md:grid-cols-2';
    if (count === 3) return 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3';
    return 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4';
  };
  if (loading) {
    return (
      <>
        <style>{keyframesStyle}</style>
        <div
          style={{
            minHeight: '100vh',
            background: palette.bg,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <div style={{ textAlign: 'center' }}>
            <div
              style={{
                width: 48,
                height: 48,
                border: `2px solid ${palette.border}`,
                borderTop: `2px solid ${palette.accent}`,
                margin: '0 auto 24px',
                animation: 'dash-pulse 1.2s linear infinite',
                borderRadius: 0,
              }}
            />
            <div
              style={{
                fontFamily: font.mono,
                fontSize: '0.8rem',
                color: palette.text,
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
              }}
            >
              Initializing Feeds
            </div>
            <div
              style={{
                fontFamily: font.mono,
                fontSize: '0.65rem',
                color: palette.textMuted,
                marginTop: 8,
                letterSpacing: '0.1em',
              }}
            >
              Establishing Secure Connection
            </div>
          </div>
        </div>
      </>
    );
  }
  return (
    <>
      <style>{keyframesStyle}</style>
      <div
        style={{
          minHeight: '100vh',
          background: palette.bg,
          color: palette.text,
          fontFamily: font.display,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* ─── RED LASER BACKGROUND EFFECTS ─── */}
        {/* Vertical laser sweep */}
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: -1,
            pointerEvents: 'none',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              height: 2,
              background: `linear-gradient(90deg, transparent 0%, ${palette.accent} 20%, ${palette.accent} 80%, transparent 100%)`,
              boxShadow: `0 0 20px 4px ${palette.accentGlow}, 0 0 60px 10px rgba(224,64,64,0.1)`,
              animation: 'laser-sweep 6s ease-in-out infinite',
              opacity: 0.6,
            }}
          />
          {/* Horizontal laser sweep */}
          <div
            style={{
              position: 'absolute',
              top: '30%',
              width: 2,
              height: '40%',
              background: `linear-gradient(180deg, transparent 0%, ${palette.accent} 30%, ${palette.accent} 70%, transparent 100%)`,
              boxShadow: `0 0 15px 3px ${palette.accentGlow}, 0 0 40px 8px rgba(224,64,64,0.08)`,
              animation: 'laser-horizontal 8s ease-in-out infinite',
              opacity: 0.4,
            }}
          />
          {/* Pulsing grid lines */}
          <div
            style={{
              position: 'absolute',
              inset: 0,
              backgroundImage: `
                linear-gradient(rgba(224,64,64,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(224,64,64,0.03) 1px, transparent 1px)
              `,
              backgroundSize: '60px 60px',
              animation: 'laser-pulse 4s ease-in-out infinite',
            }}
          />
          {/* Corner accent glows */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: 200,
              height: 200,
              background: `radial-gradient(circle at top left, rgba(224,64,64,0.06) 0%, transparent 70%)`,
            }}
          />
          <div
            style={{
              position: 'absolute',
              bottom: 0,
              right: 0,
              width: 200,
              height: 200,
              background: `radial-gradient(circle at bottom right, rgba(224,64,64,0.06) 0%, transparent 70%)`,
            }}
          />
        </div>
        {/* Subtle scanline overlay */}
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            height: 2,
            background: `linear-gradient(90deg, transparent, ${palette.accent}, transparent)`,
            opacity: 0.3,
            animation: 'scanline 4s linear infinite',
            zIndex: 50,
            pointerEvents: 'none',
          }}
        />
        {/* ─── HEADER ─── */}
        <header
          style={{
            borderBottom: `1px solid ${palette.border}`,
            background: `${palette.bg}ee`,
            backdropFilter: 'blur(12px)',
            position: 'sticky',
            top: 0,
            zIndex: 40,
          }}
        >
          <div
            style={{
              maxWidth: 1600,
              margin: '0 auto',
              padding: '16px 24px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 16,
            }}
          >
            {/* Logo & Title */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              <div
                style={{
                  width: 40,
                  height: 40,
                  border: `2px solid ${palette.accent}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontFamily: font.mono,
                  fontWeight: 700,
                  fontSize: '1.1rem',
                  color: palette.accent,
                  position: 'relative',
                  borderRadius: 0,
                }}
              >
                M
                <div
                  style={{
                    position: 'absolute',
                    top: -1,
                    left: 8,
                    right: 8,
                    height: 2,
                    background: palette.accent,
                  }}
                />
              </div>
              <div>
                <div
                  style={{
                    fontFamily: font.display,
                    fontWeight: 700,
                    fontSize: '1.1rem',
                    letterSpacing: '-0.02em',
                  }}
                >
                  MetroEye
                </div>
                <div
                  style={{
                    fontFamily: font.mono,
                    fontSize: '0.6rem',
                    color: palette.textMuted,
                    letterSpacing: '0.12em',
                    textTransform: 'uppercase',
                  }}
                >
                  AI Platform Safety · Detection & Prevention
                </div>
              </div>
            </div>
            {/* View Mode Toggle */}
            <div style={{ display: 'flex', gap: 0, border: `1px solid ${palette.border}`, borderRadius: 0 }}>
              {['grid', 'list'].map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  style={{
                    padding: '8px 18px',
                    fontSize: '0.7rem',
                    fontFamily: font.mono,
                    letterSpacing: '0.12em',
                    border: 'none',
                    cursor: 'pointer',
                    textTransform: 'uppercase',
                    transition: 'all 0.15s ease',
                    background: viewMode === mode ? palette.accent : 'transparent',
                    color: viewMode === mode ? '#fff' : palette.textMuted,
                    borderRadius: 0,
                  }}
                >
                  {mode === 'grid' ? '▦' : '☰'} {mode}
                </button>
              ))}
            </div>
            {/* Stats */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
              <div style={{ textAlign: 'center' }}>
                <div
                  style={{
                    fontFamily: font.mono,
                    fontSize: '1.4rem',
                    fontWeight: 700,
                    color: palette.accent,
                  }}
                >
                  {cameras.length}
                </div>
                <div
                  style={{
                    fontFamily: font.mono,
                    fontSize: '0.55rem',
                    color: palette.textMuted,
                    textTransform: 'uppercase',
                    letterSpacing: '0.15em',
                  }}
                >
                  Cameras
                </div>
              </div>
              <div style={{ width: 1, height: 28, background: palette.border }} />
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div
                  style={{
                    width: 8,
                    height: 8,
                    background: palette.safe,
                    animation: 'status-blink 2s infinite',
                    borderRadius: 0,
                  }}
                />
                <span
                  style={{
                    fontFamily: font.mono,
                    fontSize: '0.65rem',
                    color: palette.safe,
                    letterSpacing: '0.1em',
                    textTransform: 'uppercase',
                  }}
                >
                  ALL ACTIVE
                </span>
              </div>
            </div>
          </div>
        </header>
        {/* ─── MAIN ─── */}
        <main style={{ maxWidth: 1600, margin: '0 auto', padding: '0 24px 40px' }}>
          {/* Status Bar */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '14px 0',
              borderBottom: `1px solid ${palette.border}`,
              marginBottom: 24,
              flexWrap: 'wrap',
              gap: 12,
            }}
          >
            <span
              style={{
                fontFamily: font.mono,
                fontSize: '0.7rem',
                color: palette.textMuted,
                letterSpacing: '0.08em',
              }}
            >
              Monitoring{' '}
              <span style={{ color: palette.text }}>{cameras.length}</span>{' '}
              camera{cameras.length !== 1 ? 's' : ''} · AI Detection Active
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              {[
                { href: '/calibration', label: '◎ Calibration', color: palette.text, borderColor: palette.border },
                { href: '/alerts', label: '▲ Alerts', color: palette.accent, borderColor: 'rgba(224,64,64,0.3)' },
              ].map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  style={{
                    fontFamily: font.mono,
                    fontSize: '0.65rem',
                    color: link.color,
                    textDecoration: 'none',
                    padding: '6px 14px',
                    border: `1px solid ${link.borderColor}`,
                    letterSpacing: '0.1em',
                    textTransform: 'uppercase',
                    transition: 'all 0.15s ease',
                    borderRadius: 0,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = `${link.color}12`;
                    e.currentTarget.style.borderColor = link.color;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.borderColor = link.borderColor;
                  }}
                >
                  {link.label}
                </a>
              ))}
            </div>
          </div>
          {/* ─── GRID VIEW ─── */}
          {viewMode === 'grid' && (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: cameras.length === 1
                  ? '1fr'
                  : cameras.length === 2
                  ? 'repeat(2, 1fr)'
                  : 'repeat(3, 1fr)',
                gap: 20,
              }}
            >
              {cameras.map((camera, i) => (
                <div
                  key={camera.id}
                  style={{
                    background: palette.surface,
                    border: `1px solid ${palette.border}`,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    animation: `slide-up 0.4s ease ${i * 0.1}s both`,
                    position: 'relative',
                    overflow: 'hidden',
                    borderRadius: 0,
                  }}
                  onClick={() => {
                    setFocusedCamera(camera.id);
                    setViewMode('focus');
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = palette.accent;
                    e.currentTarget.style.boxShadow = `0 0 24px ${palette.accentGlow}`;
                    e.currentTarget.style.background = palette.surfaceHover;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = palette.border;
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.background = palette.surface;
                  }}
                >
                  {/* Top accent bar */}
                  <div style={{ height: 2, background: palette.accent, opacity: 0.6 }} />
                  {/* Camera header */}
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '12px 16px',
                      borderBottom: `1px solid ${palette.border}`,
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontSize: '0.85rem' }}>📹</span>
                      <span
                        style={{
                          fontFamily: font.mono,
                          fontSize: '0.7rem',
                          color: palette.text,
                          letterSpacing: '0.1em',
                          textTransform: 'uppercase',
                          fontWeight: 600,
                        }}
                      >
                        Camera {i + 1}
                      </span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div
                        style={{
                          width: 6,
                          height: 6,
                          background: palette.safe,
                          animation: 'status-blink 2s infinite',
                          borderRadius: 0,
                        }}
                      />
                      <span
                        style={{
                          fontFamily: font.mono,
                          fontSize: '0.6rem',
                          color: palette.safe,
                          letterSpacing: '0.1em',
                          textTransform: 'uppercase',
                          background: 'transparent',
                          padding: 0,
                          borderRadius: 0,
                        }}
                      >
                        Live
                      </span>
                    </div>
                  </div>
                  {/* Feed */}
                  <LiveFeed cameraId={camera.id} />
                  {/* Camera info footer — matched to dashboard aesthetic */}
                  <div
                    style={{
                      padding: '12px 16px',
                      borderTop: `1px solid ${palette.border}`,
                      background: palette.bg,
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <div>
                      <div
                        style={{
                          fontFamily: font.mono,
                          fontSize: '0.68rem',
                          color: palette.text,
                          letterSpacing: '0.06em',
                          fontWeight: 600,
                        }}
                      >
                        {camera.name}
                      </div>
                      <div
                        style={{
                          fontFamily: font.mono,
                          fontSize: '0.58rem',
                          color: palette.textMuted,
                          letterSpacing: '0.08em',
                          marginTop: 2,
                          textTransform: 'uppercase',
                        }}
                      >
                        {camera.location}
                      </div>
                    </div>
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                      }}
                    >
                      <div
                        style={{
                          width: 4,
                          height: 4,
                          background: palette.accent,
                          borderRadius: 0,
                        }}
                      />
                      <span
                        style={{
                          fontFamily: font.mono,
                          fontSize: '0.55rem',
                          color: palette.textMuted,
                          letterSpacing: '0.1em',
                          textTransform: 'uppercase',
                        }}
                      >
                        Tracking
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
          {/* ─── LIST VIEW ─── */}
          {viewMode === 'list' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {cameras.map((camera, i) => (
                <div
                  key={camera.id}
                  style={{
                    animation: `slide-up 0.3s ease ${i * 0.08}s both`,
                  }}
                >
                  <div
                    style={{
                      background: palette.surface,
                      border: `1px solid ${palette.border}`,
                      padding: 20,
                      display: 'flex',
                      gap: 24,
                      alignItems: 'flex-start',
                      transition: 'all 0.2s ease',
                      borderRadius: 0,
                    }}
                  >
                    <div
                      style={{
                        width: 320,
                        flexShrink: 0,
                        border: `1px solid ${palette.border}`,
                        overflow: 'hidden',
                        borderRadius: 0,
                      }}
                    >
                      <LiveFeed cameraId={camera.id} />
                    </div>
                    <div style={{ flex: 1 }}>
                      <div>
                        <div
                          style={{
                            fontFamily: font.display,
                            fontWeight: 700,
                            fontSize: '1rem',
                            marginBottom: 4,
                          }}
                        >
                          {camera.name}
                        </div>
                        <div
                          style={{
                            fontFamily: font.mono,
                            fontSize: '0.65rem',
                            color: palette.textMuted,
                            letterSpacing: '0.1em',
                            textTransform: 'uppercase',
                          }}
                        >
                          {camera.location}
                        </div>
                        <div
                          style={{
                            display: 'flex',
                            gap: 24,
                            marginTop: 14,
                            flexWrap: 'wrap',
                          }}
                        >
                          {[
                            { label: 'Status', value: '● Active', color: palette.safe },
                            { label: 'Cam ID', value: camera.id, color: palette.text },
                          ].map((row) => (
                            <div key={row.label} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                              <span
                                style={{
                                  fontFamily: font.mono,
                                  fontSize: '0.6rem',
                                  color: palette.textMuted,
                                  textTransform: 'uppercase',
                                  letterSpacing: '0.1em',
                                }}
                              >
                                {row.label}
                              </span>
                              <span
                                style={{
                                  fontFamily: font.mono,
                                  fontSize: '0.65rem',
                                  color: row.color,
                                }}
                              >
                                {row.value}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          setFocusedCamera(camera.id);
                          setViewMode('focus');
                        }}
                        style={{
                          marginTop: 16,
                          padding: '10px 20px',
                          border: `1px solid ${palette.accent}`,
                          background: 'transparent',
                          color: palette.accent,
                          fontSize: '0.7rem',
                          letterSpacing: '0.15em',
                          fontFamily: font.mono,
                          cursor: 'pointer',
                          transition: 'all 0.15s ease',
                          textTransform: 'uppercase',
                          borderRadius: 0,
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = palette.accent;
                          e.currentTarget.style.color = '#fff';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = 'transparent';
                          e.currentTarget.style.color = palette.accent;
                        }}
                      >
                        Full Screen →
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
          {/* ─── FOCUS VIEW ─── */}
          {viewMode === 'focus' && focusedCamera && (
            <div style={{ animation: 'slide-up 0.3s ease both' }}>
              {/* Scanline on focused view */}
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: 1,
                  background: `linear-gradient(90deg, transparent, ${palette.accent}, transparent)`,
                  animation: 'scanline 3s linear infinite',
                  opacity: 0.5,
                  zIndex: 2,
                  pointerEvents: 'none',
                }}
              />
              <div
                style={{
                  background: palette.surface,
                  border: `1px solid ${palette.border}`,
                  overflow: 'hidden',
                  borderRadius: 0,
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '16px 20px',
                    borderBottom: `1px solid ${palette.border}`,
                  }}
                >
                  <div>
                    <div
                      style={{
                        fontFamily: font.display,
                        fontWeight: 700,
                        fontSize: '1.1rem',
                      }}
                    >
                      {cameras.find((c) => c.id === focusedCamera)?.name}
                    </div>
                    <div
                      style={{
                        fontFamily: font.mono,
                        fontSize: '0.6rem',
                        color: palette.textMuted,
                        letterSpacing: '0.1em',
                        textTransform: 'uppercase',
                        marginTop: 4,
                      }}
                    >
                      Live Feed · AI Monitoring Active
                    </div>
                  </div>
                  <button
                    onClick={() => setViewMode('grid')}
                    style={{
                      padding: '10px 22px',
                      border: `1px solid ${palette.border}`,
                      background: 'transparent',
                      color: palette.textMuted,
                      fontFamily: font.mono,
                      fontSize: '0.7rem',
                      letterSpacing: '0.1em',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                      textTransform: 'uppercase',
                      borderRadius: 0,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = palette.accent;
                      e.currentTarget.style.color = palette.accent;
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = palette.border;
                      e.currentTarget.style.color = palette.textMuted;
                    }}
                  >
                    ✕ Close
                  </button>
                </div>
                <div style={{ padding: 20 }}>
                  <LiveFeed cameraId={focusedCamera} />
                </div>
              </div>
            </div>
          )}
          {/* ─── EMPTY STATE ─── */}
          {cameras.length === 0 && (
            <div
              style={{
                textAlign: 'center',
                padding: '80px 20px',
                border: `1px dashed ${palette.border}`,
                borderRadius: 0,
              }}
            >
              <div
                style={{
                  fontSize: '2.5rem',
                  marginBottom: 16,
                  opacity: 0.3,
                }}
              >
                ◻
              </div>
              <div
                style={{
                  fontFamily: font.display,
                  fontWeight: 600,
                  fontSize: '1rem',
                  marginBottom: 8,
                }}
              >
                No Cameras Detected
              </div>
              <div
                style={{
                  fontFamily: font.mono,
                  fontSize: '0.7rem',
                  color: palette.textMuted,
                  letterSpacing: '0.08em',
                  marginBottom: 24,
                }}
              >
                Connect surveillance feeds to begin monitoring
              </div>
              <button
                style={{
                  padding: '12px 28px',
                  border: `1px solid ${palette.accent}`,
                  background: 'transparent',
                  color: palette.accent,
                  fontFamily: font.mono,
                  fontSize: '0.7rem',
                  letterSpacing: '0.12em',
                  cursor: 'pointer',
                  textTransform: 'uppercase',
                  transition: 'all 0.15s ease',
                  borderRadius: 0,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = palette.accent;
                  e.currentTarget.style.color = '#fff';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.color = palette.accent;
                }}
              >
                + Add Camera
              </button>
            </div>
          )}
        </main>
        {/* ─── FOOTER ─── */}
        <footer
          style={{
            borderTop: `1px solid ${palette.border}`,
            padding: '16px 24px',
            background: `${palette.bg}ee`,
          }}
        >
          <div
            style={{
              maxWidth: 1600,
              margin: '0 auto',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div
              style={{
                fontFamily: font.mono,
                fontSize: '0.6rem',
                color: palette.textMuted,
                letterSpacing: '0.08em',
              }}
            >
              MetroEye · AI-Powered
              <span style={{ margin: '0 8px', opacity: 0.3 }}>|</span>
              Platform Safety & Prevention
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div
                style={{
                  width: 6,
                  height: 6,
                  background: palette.safe,
                  borderRadius: 0,
                }}
              />
              <span
                style={{
                  fontFamily: font.mono,
                  fontSize: '0.6rem',
                  color: palette.safe,
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                }}
              >
                System Operational
              </span>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
}