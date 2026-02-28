/**
 * Multi-Camera Dashboard - Monitor multiple cameras simultaneously
 *
 * Perfect for demo to judges - shows real-time tracking across multiple cameras
 *
 * Features:
 * - Grid layout with multiple camera feeds
 * - Real-time tracking data for each camera
 * - Responsive design (adapts to screen size)
 * - Connection status for all cameras
 * - Quick stats summary
 */

import { useState, useEffect } from 'react';
import axios from 'axios';
import LiveFeed from '../components/LiveFeed';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

// --- Inline styles as constants for the brutalist safety-monitor theme ---
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
          className="min-h-screen flex items-center justify-center"
          style={{ background: palette.bg, fontFamily: font.display }}
        >
          <div className="text-center" style={{ animation: 'slide-up 0.6s ease-out' }}>
            {/* Spinner — square brutalist */}
            <div
              style={{
                width: 56,
                height: 56,
                margin: '0 auto 24px',
                border: `3px solid ${palette.border}`,
                borderTop: `3px solid ${palette.accent}`,
                animation: 'spin 0.8s linear infinite',
              }}
            />
            <p
              style={{
                fontFamily: font.mono,
                fontSize: '0.85rem',
                letterSpacing: '0.25em',
                color: palette.text,
                textTransform: 'uppercase',
              }}
            >
              Initializing Feeds
            </p>
            <p
              style={{
                fontFamily: font.mono,
                fontSize: '0.6rem',
                color: palette.textMuted,
                letterSpacing: '0.3em',
                marginTop: 8,
              }}
            >
              Establishing Secure Connection
            </p>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <style>{keyframesStyle}</style>
      <div
        className="min-h-screen flex flex-col"
        style={{ background: palette.bg, fontFamily: font.display, color: palette.text }}
      >
        {/* Subtle scanline overlay */}
        <div
          style={{
            pointerEvents: 'none',
            position: 'fixed',
            inset: 0,
            zIndex: 9999,
            background:
              'repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(255,255,255,0.008) 3px, rgba(255,255,255,0.008) 4px)',
          }}
        />

        {/* ─── HEADER ─── */}
        <header
          style={{
            background: palette.surface,
            borderBottom: `1px solid ${palette.border}`,
            position: 'sticky',
            top: 0,
            zIndex: 50,
          }}
        >
          <div className="max-w-screen-2xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between flex-wrap gap-4">
              {/* Logo & Title */}
              <div className="flex items-center gap-4">
                <div
                  style={{
                    width: 44,
                    height: 44,
                    border: `2px solid ${palette.accent}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontFamily: font.display,
                    fontSize: '1.1rem',
                    fontWeight: 800,
                    color: palette.accent,
                    letterSpacing: '-0.02em',
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                >
                  M
                  {/* top accent line */}
                  <div
                    style={{
                      position: 'absolute',
                      bottom: 0,
                      left: 0,
                      right: 0,
                      height: 3,
                      background: palette.accent,
                    }}
                  />
                </div>
                <div>
                  <h1
                    style={{
                      fontFamily: font.display,
                      fontSize: '1.25rem',
                      fontWeight: 700,
                      letterSpacing: '0.04em',
                      textTransform: 'uppercase',
                      color: palette.text,
                      margin: 0,
                      lineHeight: 1.2,
                    }}
                  >
                    Metro<span style={{ color: palette.accent }}>Eye</span>
                  </h1>
                  <p
                    style={{
                      fontFamily: font.mono,
                      color: palette.textMuted,
                      fontSize: '0.6rem',
                      letterSpacing: '0.2em',
                      textTransform: 'uppercase',
                      margin: 0,
                    }}
                  >
                    AI Platform Safety · Detection & Prevention
                  </p>
                </div>
              </div>

              {/* View Mode Toggle */}
              <div
                className="flex items-center"
                style={{ border: `1px solid ${palette.border}`, overflow: 'hidden' }}
              >
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
                    }}
                  >
                    {mode === 'grid' ? '▦' : '☰'} {mode}
                  </button>
                ))}
              </div>

              {/* Stats */}
              <div
                className="hidden lg:flex items-center gap-5"
                style={{
                  border: `1px solid ${palette.border}`,
                  padding: '10px 20px',
                  background: palette.accentSoft,
                }}
              >
                <div className="text-center">
                  <div
                    style={{
                      fontSize: '1.4rem',
                      fontWeight: 700,
                      color: palette.text,
                      fontFamily: font.mono,
                    }}
                  >
                    {cameras.length}
                  </div>
                  <div
                    style={{
                      fontSize: '0.55rem',
                      color: palette.textMuted,
                      letterSpacing: '0.25em',
                      textTransform: 'uppercase',
                    }}
                  >
                    Cameras
                  </div>
                </div>
                <div style={{ width: 1, height: 28, background: palette.border }} />
                <div className="text-center flex items-center gap-2">
                  <div
                    style={{
                      width: 8,
                      height: 8,
                      background: palette.safe,
                      borderRadius: '50%',
                      animation: 'status-blink 2s ease-in-out infinite',
                    }}
                  />
                  <div
                    style={{
                      fontSize: '0.65rem',
                      color: palette.safe,
                      letterSpacing: '0.15em',
                      fontFamily: font.mono,
                    }}
                  >
                    ALL ACTIVE
                  </div>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* ─── MAIN ─── */}
        <main className="max-w-screen-2xl mx-auto px-6 py-6 relative z-10 flex-1 w-full">
          {/* Status Bar */}
          <div
            className="mb-5 flex items-center justify-between flex-wrap gap-3"
            style={{
              border: `1px solid ${palette.border}`,
              borderLeft: `3px solid ${palette.accent}`,
              background: palette.surface,
              padding: '14px 20px',
            }}
          >
            <div
              style={{
                fontSize: '0.72rem',
                color: palette.textMuted,
                letterSpacing: '0.1em',
                fontFamily: font.mono,
              }}
            >
              Monitoring{' '}
              <span style={{ color: palette.text, fontWeight: 600 }}>{cameras.length}</span>{' '}
              camera{cameras.length !== 1 ? 's' : ''} · AI Detection Active
            </div>

            <div className="flex items-center gap-2">
              {[
                { href: '/calibration', label: '◎ Calibration', color: palette.text, borderColor: palette.border },
                { href: '/alerts', label: '▲ Alerts', color: palette.accent, borderColor: 'rgba(224,64,64,0.3)' },
              ].map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  style={{
                    padding: '6px 14px',
                    border: `1px solid ${link.borderColor}`,
                    background: 'transparent',
                    color: link.color,
                    fontSize: '0.65rem',
                    letterSpacing: '0.12em',
                    textDecoration: 'none',
                    fontFamily: font.mono,
                    textTransform: 'uppercase',
                    transition: 'all 0.15s ease',
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
            <div className={`grid ${getGridClass()} gap-4`}>
              {cameras.map((camera, i) => (
                <div
                  key={camera.id}
                  className="cursor-pointer group"
                  style={{
                    border: `1px solid ${palette.border}`,
                    background: palette.surface,
                    transition: 'all 0.2s ease',
                    position: 'relative',
                    overflow: 'hidden',
                    animation: `slide-up 0.4s ease-out ${i * 0.08}s both`,
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
                  <div
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: 2,
                      background: `linear-gradient(90deg, ${palette.accent}, transparent)`,
                    }}
                  />

                  {/* Camera header */}
                  <div
                    style={{
                      padding: '10px 14px',
                      borderBottom: `1px solid ${palette.border}`,
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      background: 'rgba(0,0,0,0.25)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontSize: '0.85rem', opacity: 0.5 }}>📹</span>
                      <span
                        style={{
                          fontFamily: font.display,
                          fontSize: '0.82rem',
                          fontWeight: 700,
                          color: palette.text,
                          letterSpacing: '0.06em',
                          textTransform: 'uppercase',
                        }}
                      >
                        Camera {i + 1}
                      </span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div
                        style={{
                          width: 7,
                          height: 7,
                          background: palette.safe,
                          borderRadius: '50%',
                          animation: 'status-blink 2s ease-in-out infinite',
                        }}
                      />
                      <span
                        style={{
                          fontFamily: font.mono,
                          fontSize: '0.6rem',
                          color: palette.safe,
                          letterSpacing: '0.1em',
                        }}
                      >
                        Live
                      </span>
                    </div>
                  </div>

                  {/* Feed */}
                  <LiveFeed cameraId={camera.id} showStats={true} />

                  {/* Camera info footer */}
                  <div
                    style={{
                      padding: '10px 14px',
                      borderTop: `1px solid ${palette.border}`,
                      background: 'rgba(0,0,0,0.2)',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <div>
                      <div
                        style={{
                          fontFamily: font.display,
                          fontSize: '0.78rem',
                          fontWeight: 600,
                          color: palette.text,
                          letterSpacing: '0.02em',
                        }}
                      >
                        {camera.name}
                      </div>
                      <div
                        style={{
                          fontFamily: font.mono,
                          fontSize: '0.55rem',
                          color: palette.textMuted,
                          letterSpacing: '0.18em',
                          textTransform: 'uppercase',
                          marginTop: 2,
                        }}
                      >
                        {camera.location}
                      </div>
                    </div>
                    <div
                      style={{
                        padding: '3px 8px',
                        border: `1px solid ${palette.border}`,
                        fontFamily: font.mono,
                        fontSize: '0.52rem',
                        color: palette.textMuted,
                        letterSpacing: '0.1em',
                      }}
                    >
                      {camera.id.replace('_', ':')}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* ─── LIST VIEW ─── */}
          {viewMode === 'list' && (
            <div className="space-y-3">
              {cameras.map((camera, i) => (
                <div
                  key={camera.id}
                  style={{
                    border: `1px solid ${palette.border}`,
                    borderLeft: `3px solid ${palette.accent}`,
                    background: palette.surface,
                    padding: '20px 24px',
                    position: 'relative',
                    overflow: 'hidden',
                    animation: `slide-up 0.4s ease-out ${i * 0.06}s both`,
                  }}
                >
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <LiveFeed cameraId={camera.id} showStats={false} />
                    </div>
                    <div className="flex flex-col justify-between">
                      <div>
                        <h3
                          style={{
                            fontFamily: font.display,
                            fontSize: '1.15rem',
                            fontWeight: 700,
                            color: palette.text,
                            letterSpacing: '0.03em',
                            margin: '0 0 6px 0',
                          }}
                        >
                          {camera.name}
                        </h3>
                        <p
                          style={{
                            fontFamily: font.mono,
                            color: palette.textMuted,
                            fontSize: '0.65rem',
                            letterSpacing: '0.15em',
                            textTransform: 'uppercase',
                            marginBottom: 16,
                          }}
                        >
                          {camera.location}
                        </p>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                          {[
                            { label: 'Status', value: '● Active', color: palette.safe },
                            { label: 'Cam ID', value: camera.id, color: palette.text },
                          ].map((row) => (
                            <div
                              key={row.label}
                              className="flex justify-between"
                              style={{
                                fontSize: '0.7rem',
                                borderBottom: `1px solid ${palette.border}`,
                                paddingBottom: 6,
                              }}
                            >
                              <span
                                style={{
                                  color: palette.textMuted,
                                  letterSpacing: '0.15em',
                                  textTransform: 'uppercase',
                                  fontFamily: font.mono,
                                }}
                              >
                                {row.label}
                              </span>
                              <span style={{ color: row.color, fontFamily: font.mono }}>
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
            <div
              className="fixed inset-0 z-50 flex items-center justify-center p-6"
              style={{ background: 'rgba(10,10,14,0.97)', animation: 'slide-up 0.3s ease-out' }}
            >
              {/* Scanline */}
              <div
                style={{
                  pointerEvents: 'none',
                  position: 'absolute',
                  inset: 0,
                  background:
                    'repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(255,255,255,0.008) 3px, rgba(255,255,255,0.008) 4px)',
                }}
              />
              <div className="w-full max-w-7xl relative z-10">
                <div className="mb-4 flex justify-between items-center">
                  <div>
                    <h2
                      style={{
                        fontFamily: font.display,
                        fontSize: '1.4rem',
                        fontWeight: 700,
                        color: palette.text,
                        letterSpacing: '0.04em',
                        margin: 0,
                      }}
                    >
                      {cameras.find((c) => c.id === focusedCamera)?.name}
                    </h2>
                    <p
                      style={{
                        fontFamily: font.mono,
                        fontSize: '0.6rem',
                        color: palette.textMuted,
                        letterSpacing: '0.2em',
                        textTransform: 'uppercase',
                        marginTop: 4,
                      }}
                    >
                      Live Feed · AI Monitoring Active
                    </p>
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
                <div
                  style={{
                    border: `1px solid ${palette.border}`,
                    boxShadow: `0 0 40px ${palette.accentGlow}`,
                  }}
                >
                  <LiveFeed cameraId={focusedCamera} showStats={true} className="shadow-2xl" />
                </div>
              </div>
            </div>
          )}

          {/* ─── EMPTY STATE ─── */}
          {cameras.length === 0 && (
            <div
              className="text-center py-20"
              style={{ animation: 'slide-up 0.5s ease-out' }}
            >
              <div
                style={{
                  fontSize: '2.5rem',
                  color: palette.textMuted,
                  marginBottom: 12,
                }}
              >
                ◻
              </div>
              <h3
                style={{
                  fontFamily: font.display,
                  fontSize: '1.2rem',
                  fontWeight: 700,
                  color: palette.text,
                  letterSpacing: '0.06em',
                  marginBottom: 8,
                  textTransform: 'uppercase',
                }}
              >
                No Cameras Detected
              </h3>
              <p
                style={{
                  fontFamily: font.mono,
                  color: palette.textMuted,
                  fontSize: '0.68rem',
                  letterSpacing: '0.15em',
                  marginBottom: 24,
                }}
              >
                Connect surveillance feeds to begin monitoring
              </p>
              <button
                style={{
                  padding: '12px 24px',
                  border: `1px solid ${palette.accent}`,
                  background: 'transparent',
                  color: palette.accent,
                  fontFamily: font.mono,
                  fontSize: '0.72rem',
                  letterSpacing: '0.15em',
                  cursor: 'pointer',
                  textTransform: 'uppercase',
                  transition: 'all 0.15s ease',
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
            background: palette.surface,
            marginTop: 'auto',
          }}
        >
          <div className="max-w-screen-2xl mx-auto px-6 py-4">
            <div
              className="flex items-center justify-between flex-wrap gap-2"
              style={{
                fontSize: '0.58rem',
                color: palette.textMuted,
                letterSpacing: '0.2em',
                textTransform: 'uppercase',
                fontFamily: font.mono,
              }}
            >
              <div>
                Metro<span style={{ color: palette.accent, fontWeight: 600 }}>Eye</span> · AI-Powered
                Platform Safety & Prevention
              </div>
              <div className="flex items-center gap-2">
                <div
                  style={{
                    width: 6,
                    height: 6,
                    background: palette.safe,
                    borderRadius: '50%',
                    animation: 'status-blink 2s ease-in-out infinite',
                  }}
                />
                System Operational
              </div>
            </div>
          </div>
        </footer>
      </div>
    </>
  );
}
