import { useState, useEffect } from 'react';
import MultiCameraDashboard from './pages/MultiCameraDashboard';
import CalibrationPage from './pages/CalibrationPage';
import AlertList from './components/AlertList';
import './App.css';

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
        <div className="min-h-screen bg-gray-100">
          <header className="bg-white shadow-md">
            <div className="max-w-7xl mx-auto px-6 py-4">
              <h1 className="text-2xl font-bold text-gray-900">🚨 Alert Management</h1>
            </div>
          </header>
          <main className="max-w-7xl mx-auto px-6 py-6">
            <button
              onClick={() => setActiveTab('dashboard')}
              className="mb-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              ← Back to Dashboard
            </button>
            <AlertList />
          </main>
        </div>
      )}

      {/* Floating Navigation */}
      <div className="fixed bottom-6 right-6 flex flex-col gap-3 z-50">
        <button
          onClick={() => setActiveTab('dashboard')}
          className={`w-14 h-14 rounded-full shadow-lg flex items-center justify-center text-2xl transition-all ${
            activeTab === 'dashboard'
              ? 'bg-blue-600 text-white scale-110'
              : 'bg-white text-gray-700 hover:bg-gray-100'
          }`}
          title="Dashboard"
        >
          📹
        </button>
        <button
          onClick={() => setActiveTab('calibration')}
          className={`w-14 h-14 rounded-full shadow-lg flex items-center justify-center text-2xl transition-all ${
            activeTab === 'calibration'
              ? 'bg-blue-600 text-white scale-110'
              : 'bg-white text-gray-700 hover:bg-gray-100'
          }`}
          title="Calibration"
        >
          🎯
        </button>
        <button
          onClick={() => setActiveTab('alerts')}
          className={`w-14 h-14 rounded-full shadow-lg flex items-center justify-center text-2xl transition-all ${
            activeTab === 'alerts'
              ? 'bg-blue-600 text-white scale-110'
              : 'bg-white text-gray-700 hover:bg-gray-100'
          }`}
          title="Alerts"
        >
          🚨
        </button>
      </div>
    </div>
  );
}

export default App;