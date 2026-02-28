/**
 * Calibration Page — MetroEye Brutalist Safety-Monitor Theme
 * All internal UI matches the surveillance-system aesthetic.
 */
import { useState } from 'react';
import ManualCalibration from '../components/ManualCalibration';

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
@keyframes slide-up {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes scanline {
  0%   { transform: translateY(-100%); }
  100% { transform: translateY(100vh); }
}
`;

export default function CalibrationPage() {
  const [selectedCamera, setSelectedCamera] = useState('camera_1');

  const cameras = [
    { id: 'camera_1', name: 'Platform 1 — North' },
    { id: 'camera_2', name: 'Platform 2 — South' },
    { id: 'camera_3', name: 'Platform 3 — Central' },
  ];

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
        {/* Scanline overlay */}
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            pointerEvents: 'none',
            zIndex: 50,
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '2px',
              background: `linear-gradient(90deg, transparent, ${palette.accentGlow}, transparent)`,
              animation: 'scanline 4s linear infinite',
              opacity: 0.4,
            }}
          />
        </div>

        {/* ─── HEADER ─── */}
        <header
          style={{
            borderBottom: `1px solid ${palette.border}`,
            padding: '20px 32px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: palette.surface,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div
              style={{
                width: 36,
                height: 36,
                background: palette.accent,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontFamily: font.mono,
                fontWeight: 900,
                fontSize: '1rem',
                color: '#fff',
              }}
            >
              M
            </div>
            <div>
              <div
                style={{
                  fontFamily: font.mono,
                  fontSize: '0.95rem',
                  fontWeight: 700,
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                }}
              >
                MetroEye{' '}
                <span style={{ color: palette.accent, fontSize: '0.65rem' }}>
                  · Calibration
                </span>
              </div>
              <div
                style={{
                  fontFamily: font.mono,
                  fontSize: '0.6rem',
                  color: palette.textMuted,
                  letterSpacing: '0.15em',
                  textTransform: 'uppercase',
                  marginTop: 2,
                }}
              >
                Platform Edge Detection · Configuration
              </div>
            </div>
          </div>

          <a
            href="/"
            style={{
              fontFamily: font.mono,
              fontSize: '0.7rem',
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              color: palette.textMuted,
              textDecoration: 'none',
              padding: '8px 18px',
              border: `1px solid ${palette.border}`,
              transition: 'all 0.15s ease',
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
            ← Back to Dashboard
          </a>
        </header>

        {/* ─── STATUS BAR ─── */}
        <div
          style={{
            padding: '10px 32px',
            borderBottom: `1px solid ${palette.border}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: palette.accentSoft,
          }}
        >
          <span
            style={{
              fontFamily: font.mono,
              fontSize: '0.65rem',
              color: palette.textMuted,
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
            }}
          >
            ◎ Calibrating{' '}
            <span style={{ color: palette.accent }}>
              {cameras.find((c) => c.id === selectedCamera)?.name}
            </span>
          </span>
          <span
            style={{
              fontFamily: font.mono,
              fontSize: '0.65rem',
              color: palette.safe,
              letterSpacing: '0.1em',
            }}
          >
            ● System Ready
          </span>
        </div>

        {/* ─── MAIN CONTENT ─── */}
        <div style={{ padding: '32px', maxWidth: 1400, margin: '0 auto' }}>
          {/* Camera Selector */}
          <div style={{ marginBottom: 28 }}>
            <div
              style={{
                fontFamily: font.mono,
                fontSize: '0.65rem',
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                color: palette.textMuted,
                marginBottom: 12,
              }}
            >
              Select Camera to Calibrate
            </div>
            <div style={{ display: 'flex', gap: 0 }}>
              {cameras.map((camera) => (
                <button
                  key={camera.id}
                  onClick={() => setSelectedCamera(camera.id)}
                  style={{
                    padding: '10px 22px',
                    fontFamily: font.mono,
                    fontSize: '0.7rem',
                    letterSpacing: '0.1em',
                    textTransform: 'uppercase',
                    border: `1px solid ${
                      selectedCamera === camera.id
                        ? palette.accent
                        : palette.border
                    }`,
                    borderRight:
                      camera.id !== cameras[cameras.length - 1].id
                        ? 'none'
                        : undefined,
                    background:
                      selectedCamera === camera.id
                        ? palette.accent
                        : 'transparent',
                    color:
                      selectedCamera === camera.id ? '#fff' : palette.textMuted,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    borderRadius: 0,
                  }}
                  onMouseEnter={(e) => {
                    if (selectedCamera !== camera.id) {
                      e.currentTarget.style.background = palette.accentSoft;
                      e.currentTarget.style.color = palette.text;
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (selectedCamera !== camera.id) {
                      e.currentTarget.style.background = 'transparent';
                      e.currentTarget.style.color = palette.textMuted;
                    }
                  }}
                >
                  {camera.name}
                </button>
              ))}
            </div>
          </div>

          {/* Calibration Component Container */}
          <div
            style={{
              border: `1px solid ${palette.border}`,
              background: palette.surface,
              padding: 0,
              animation: 'slide-up 0.4s ease',
            }}
          >
            {/* Container header */}
            <div
              style={{
                padding: '16px 24px',
                borderBottom: `1px solid ${palette.border}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ color: palette.accent, fontSize: '0.8rem' }}>
                  ◎
                </span>
                <span
                  style={{
                    fontFamily: font.mono,
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    letterSpacing: '0.08em',
                    textTransform: 'uppercase',
                  }}
                >
                  Platform Edge Calibration
                </span>
              </div>
              <span
                style={{
                  fontFamily: font.mono,
                  fontSize: '0.6rem',
                  color: palette.textMuted,
                  letterSpacing: '0.1em',
                  background: palette.bg,
                  padding: '4px 12px',
                  border: `1px solid ${palette.border}`,
                }}
              >
                Camera: {selectedCamera}
              </span>
            </div>

            {/* ManualCalibration lives here — override its internal styles via CSS */}
            <div
              className="calibration-override"
              style={{ padding: 24 }}
            >
              <style>{`
                /* ─── Override ManualCalibration internals ─── */
                .calibration-override * {
                  border-radius: 0 !important;
                }
                .calibration-override button {
                  font-family: ${font.mono} !important;
                  font-size: 0.7rem !important;
                  letter-spacing: 0.12em !important;
                  text-transform: uppercase !important;
                  border-radius: 0 !important;
                  transition: all 0.15s ease !important;
                  cursor: pointer !important;
                }
                /* Primary / accent buttons (Save, Auto Calibrate) */
                .calibration-override button[class*="bg-blue"],
                .calibration-override button[class*="bg-green"],
                .calibration-override button[class*="bg-indigo"],
                .calibration-override button[class*="primary"] {
                  background: ${palette.accent} !important;
                  color: #fff !important;
                  border: 1px solid ${palette.accent} !important;
                  box-shadow: none !important;
                }
                .calibration-override button[class*="bg-blue"]:hover,
                .calibration-override button[class*="bg-green"]:hover,
                .calibration-override button[class*="bg-indigo"]:hover,
                .calibration-override button[class*="primary"]:hover {
                  background: transparent !important;
                  color: ${palette.accent} !important;
                }
                /* Secondary / outline buttons (Reset, etc.) */
                .calibration-override button[class*="bg-red"],
                .calibration-override button[class*="bg-gray"],
                .calibration-override button[class*="bg-yellow"],
                .calibration-override button[class*="secondary"],
                .calibration-override button[class*="outline"],
                .calibration-override button[class*="destructive"] {
                  background: transparent !important;
                  color: ${palette.textMuted} !important;
                  border: 1px solid ${palette.border} !important;
                  box-shadow: none !important;
                }
                .calibration-override button[class*="bg-red"]:hover,
                .calibration-override button[class*="bg-gray"]:hover,
                .calibration-override button[class*="secondary"]:hover,
                .calibration-override button[class*="outline"]:hover {
                  border-color: ${palette.accent} !important;
                  color: ${palette.accent} !important;
                }
                /* Cards / panels / info boxes */
                .calibration-override div[class*="bg-white"],
                .calibration-override div[class*="bg-gray-50"],
                .calibration-override div[class*="bg-gray-100"],
                .calibration-override div[class*="bg-blue-50"],
                .calibration-override div[class*="bg-green-50"],
                .calibration-override div[class*="bg-yellow-50"],
                .calibration-override div[class*="rounded-lg"],
                .calibration-override div[class*="rounded-xl"],
                .calibration-override div[class*="shadow"] {
                  background: ${palette.surface} !important;
                  border: 1px solid ${palette.border} !important;
                  box-shadow: none !important;
                  color: ${palette.text} !important;
                }
                /* Text overrides */
                .calibration-override h1,
                .calibration-override h2,
                .calibration-override h3,
                .calibration-override h4,
                .calibration-override h5 {
                  font-family: ${font.mono} !important;
                  letter-spacing: 0.06em !important;
                  text-transform: uppercase !important;
                  color: ${palette.text} !important;
                }
                .calibration-override p,
                .calibration-override span,
                .calibration-override label {
                  color: ${palette.textMuted} !important;
                  font-family: ${font.mono} !important;
                  font-size: 0.72rem !important;
                }
                .calibration-override input,
                .calibration-override select,
                .calibration-override textarea {
                  background: ${palette.bg} !important;
                  border: 1px solid ${palette.border} !important;
                  color: ${palette.text} !important;
                  font-family: ${font.mono} !important;
                  border-radius: 0 !important;
                  font-size: 0.72rem !important;
                }
                .calibration-override input:focus,
                .calibration-override select:focus {
                  border-color: ${palette.accent} !important;
                  outline: none !important;
                  box-shadow: 0 0 0 1px ${palette.accentGlow} !important;
                }
                /* Status indicators */
                .calibration-override span[class*="text-green"],
                .calibration-override span[class*="text-emerald"] {
                  color: ${palette.safe} !important;
                }
                .calibration-override span[class*="text-red"] {
                  color: ${palette.accent} !important;
                }
                .calibration-override span[class*="text-yellow"],
                .calibration-override span[class*="text-amber"] {
                  color: ${palette.warn} !important;
                }
                /* Canvas container */
                .calibration-override canvas {
                  border: 1px solid ${palette.border} !important;
                  border-radius: 0 !important;
                }
              `}</style>
              <ManualCalibration cameraId={selectedCamera} />
            </div>
          </div>

          {/* ─── Tips Section ─── */}
          <div
            style={{
              marginTop: 28,
              border: `1px solid ${palette.border}`,
              background: palette.surface,
              animation: 'slide-up 0.5s ease',
            }}
          >
            <div
              style={{
                padding: '14px 24px',
                borderBottom: `1px solid ${palette.border}`,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              <span style={{ color: palette.accent }}>▲</span>
              <span
                style={{
                  fontFamily: font.mono,
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  letterSpacing: '0.12em',
                  textTransform: 'uppercase',
                }}
              >
                Calibration Protocol
              </span>
            </div>
            <div style={{ padding: '20px 24px' }}>
              {[
                {
                  label: 'Manual',
                  desc: 'Click along the platform edge to create a boundary line. Use at least 2 points.',
                },
                {
                  label: 'Auto',
                  desc: 'YOLO AI detection automatically finds the platform edge. Works best with clear visibility.',
                },
                {
                  label: 'Resolution',
                  desc: 'Coordinates are normalized — calibration works at any display resolution.',
                },
                {
                  label: 'Re-calibration',
                  desc: 'Click "Reset Points" and start over at any time.',
                },
              ].map((tip, i) => (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    gap: 12,
                    marginBottom: i < 3 ? 14 : 0,
                    alignItems: 'baseline',
                  }}
                >
                  <span
                    style={{
                      fontFamily: font.mono,
                      fontSize: '0.6rem',
                      color: palette.accent,
                      minWidth: 80,
                      textTransform: 'uppercase',
                      letterSpacing: '0.1em',
                    }}
                  >
                    [{tip.label}]
                  </span>
                  <span
                    style={{
                      fontFamily: font.mono,
                      fontSize: '0.68rem',
                      color: palette.textMuted,
                      lineHeight: 1.6,
                    }}
                  >
                    {tip.desc}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ─── FOOTER ─── */}
        <footer
          style={{
            borderTop: `1px solid ${palette.border}`,
            padding: '16px 32px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginTop: 40,
          }}
        >
          <span
            style={{
              fontFamily: font.mono,
              fontSize: '0.6rem',
              color: palette.textMuted,
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
            }}
          >
            MetroEye · AI-Powered Platform Safety & Prevention
          </span>
          <span
            style={{
              fontFamily: font.mono,
              fontSize: '0.6rem',
              color: palette.safe,
              letterSpacing: '0.1em',
            }}
          >
            ● System Operational
          </span>
        </footer>
      </div>
    </>
  );
}
