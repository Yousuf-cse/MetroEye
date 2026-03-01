import { useState, useEffect } from 'react';
import MultiCameraDashboard from './pages/MultiCameraDashboard';
import CalibrationPage from './pages/CalibrationPage';
import AlertList from './components/AlertList';
import './App.css';

// Retro/Brutalist theme
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
  info: '#3b82f6',
};

const font = {
  mono: "'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
  display: "'Inter', 'Helvetica Neue', Arial, sans-serif",
};

function App() {
  const [activeTab, setActiveTab] = useState('dashboard'); // dashboard, calibration, alerts

  useEffect(() => {
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  return (
    <div className="App">
      {activeTab === 'dashboard' && <MultiCameraDashboard />}
      {activeTab === 'calibration' && <CalibrationPage />}
      {activeTab === 'alerts' && (
        <div style={{
          minHeight: '100vh',
          background: palette.bg,
        }}>
          <header style={{
            background: palette.surface,
            borderBottom: `2px solid ${palette.border}`,
            position: 'sticky',
            top: 0,
            zIndex: 40,
          }}>
            <div style={{
              maxWidth: 1400,
              margin: '0 auto',
              padding: '20px 24px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <h1 style={{
                fontSize: '1.5rem',
                fontFamily: font.mono,
                color: palette.text,
                letterSpacing: '0.05em',
                textTransform: 'uppercase',
                fontWeight: 700,
              }}>🚨 Alert Management</h1>
              <button
                onClick={() => setActiveTab('dashboard')}
                style={{
                  padding: '10px 20px',
                  background: 'transparent',
                  border: `1px solid ${palette.border}`,
                  color: palette.text,
                  borderRadius: 0,
                  cursor: 'pointer',
                  fontFamily: font.mono,
                  fontSize: '0.7rem',
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  transition: 'all 0.15s ease',
                }}
                onMouseEnter={(e) => {
                  e.target.style.background = palette.surfaceHover;
                  e.target.style.borderColor = palette.accent;
                  e.target.style.color = palette.accent;
                }}
                onMouseLeave={(e) => {
                  e.target.style.background = 'transparent';
                  e.target.style.borderColor = palette.border;
                  e.target.style.color = palette.text;
                }}
              >
                ← Back to Dashboard
              </button>
            </div>
          </header>
          <main>
            <AlertList />
          </main>
        </div>
      )}

      {/* Floating Navigation */}
      <div style={{
        position: 'fixed',
        bottom: 24,
        right: 24,
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        zIndex: 50,
      }}>
        <button
          onClick={() => setActiveTab('dashboard')}
          style={{
            width: 56,
            height: 56,
            borderRadius: 0,
            boxShadow: activeTab === 'dashboard'
              ? `0 0 20px ${palette.accent}40, 0 4px 12px rgba(0,0,0,0.8)`
              : `0 4px 12px rgba(0,0,0,0.6)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.5rem',
            transition: 'all 0.2s ease',
            background: activeTab === 'dashboard' ? palette.accent : palette.surface,
            color: palette.text,
            border: `2px solid ${activeTab === 'dashboard' ? palette.accent : palette.border}`,
            cursor: 'pointer',
            transform: activeTab === 'dashboard' ? 'scale(1.1)' : 'scale(1)',
          }}
          title="Dashboard"
          onMouseEnter={(e) => {
            if (activeTab !== 'dashboard') {
              e.target.style.background = palette.surfaceHover;
              e.target.style.borderColor = palette.textMuted;
            }
          }}
          onMouseLeave={(e) => {
            if (activeTab !== 'dashboard') {
              e.target.style.background = palette.surface;
              e.target.style.borderColor = palette.border;
            }
          }}
        >
          📹
        </button>
        <button
          onClick={() => setActiveTab('calibration')}
          style={{
            width: 56,
            height: 56,
            borderRadius: 0,
            boxShadow: activeTab === 'calibration'
              ? `0 0 20px ${palette.accent}40, 0 4px 12px rgba(0,0,0,0.8)`
              : `0 4px 12px rgba(0,0,0,0.6)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.5rem',
            transition: 'all 0.2s ease',
            background: activeTab === 'calibration' ? palette.accent : palette.surface,
            color: palette.text,
            border: `2px solid ${activeTab === 'calibration' ? palette.accent : palette.border}`,
            cursor: 'pointer',
            transform: activeTab === 'calibration' ? 'scale(1.1)' : 'scale(1)',
          }}
          title="Calibration"
          onMouseEnter={(e) => {
            if (activeTab !== 'calibration') {
              e.target.style.background = palette.surfaceHover;
              e.target.style.borderColor = palette.textMuted;
            }
          }}
          onMouseLeave={(e) => {
            if (activeTab !== 'calibration') {
              e.target.style.background = palette.surface;
              e.target.style.borderColor = palette.border;
            }
          }}
        >
          🎯
        </button>
        <button
          onClick={() => setActiveTab('alerts')}
          style={{
            width: 56,
            height: 56,
            borderRadius: 0,
            boxShadow: activeTab === 'alerts'
              ? `0 0 20px ${palette.accent}40, 0 4px 12px rgba(0,0,0,0.8)`
              : `0 4px 12px rgba(0,0,0,0.6)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.5rem',
            transition: 'all 0.2s ease',
            background: activeTab === 'alerts' ? palette.accent : palette.surface,
            color: palette.text,
            border: `2px solid ${activeTab === 'alerts' ? palette.accent : palette.border}`,
            cursor: 'pointer',
            transform: activeTab === 'alerts' ? 'scale(1.1)' : 'scale(1)',
          }}
          title="Alerts"
          onMouseEnter={(e) => {
            if (activeTab !== 'alerts') {
              e.target.style.background = palette.surfaceHover;
              e.target.style.borderColor = palette.textMuted;
            }
          }}
          onMouseLeave={(e) => {
            if (activeTab !== 'alerts') {
              e.target.style.background = palette.surface;
              e.target.style.borderColor = palette.border;
            }
          }}
        >
          🚨
        </button>
      </div>
    </div>
  );
}

export default App;