import React, { useState, useEffect } from 'react';
import VideoStream from './components/VideoStream';
import AlertList from './components/AlertList';
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('live'); // live, alerts, dashboard
  const [notificationPermission, setNotificationPermission] = useState('default');

  useEffect(() => {
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission().then((permission) => {
        setNotificationPermission(permission);
      });
    }
  }, []);

  return (
    <div className="App">
      {/* Header */}
      <header style={{
        background: '#1976d2',
        color: 'white',
        padding: '20px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <h1 style={{ margin: 0 }}>🚇 MetroEye - Behavior Detection System</h1>
        <p style={{ margin: '5px 0 0 0', opacity: 0.9 }}>
          Real-time suspicious behavior monitoring
        </p>
      </header>

      {/* Navigation */}
      <nav style={{
        background: 'white',
        padding: '10px 20px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        display: 'flex',
        gap: '10px'
      }}>
        <button
          onClick={() => setActiveTab('live')}
          style={{
            padding: '10px 20px',
            background: activeTab === 'live' ? '#2196f3' : 'transparent',
            color: activeTab === 'live' ? 'white' : '#333',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          📹 Live Feed
        </button>
        <button
          onClick={() => setActiveTab('alerts')}
          style={{
            padding: '10px 20px',
            background: activeTab === 'alerts' ? '#2196f3' : 'transparent',
            color: activeTab === 'alerts' ? 'white' : '#333',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          🚨 Alerts
        </button>
        <button
          onClick={() => setActiveTab('dashboard')}
          style={{
            padding: '10px 20px',
            background: activeTab === 'dashboard' ? '#2196f3' : 'transparent',
            color: activeTab === 'dashboard' ? 'white' : '#333',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontWeight: 'bold'
          }}
        >
          📊 Dashboard
        </button>
      </nav>

      {/* Main Content */}
      <main style={{ padding: '20px', background: '#f5f5f5', minHeight: 'calc(100vh - 200px)' }}>
        {activeTab === 'live' && (
          <div>
            <h2>Live Camera Feed</h2>
            <VideoStream cameraId="camera_1" />
          </div>
        )}

        {activeTab === 'alerts' && (
          <div>
            <h2>Alert Management</h2>
            <AlertList />
          </div>
        )}

        {activeTab === 'dashboard' && (
          <Dashboard />
        )}
      </main>
    </div>
  );
}

export default App;