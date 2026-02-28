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

export default function MultiCameraDashboard() {
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState('grid'); // 'grid', 'list', 'focus'
  const [focusedCamera, setFocusedCamera] = useState(null);

  // Default cameras (you can fetch this from backend in production)
  const defaultCameras = [
    { id: 'camera_1', name: 'Platform 1 - North', location: 'Gate A' },
    { id: 'camera_2', name: 'Platform 2 - South', location: 'Gate B' },
    { id: 'camera_3', name: 'Platform 3 - Central', location: 'Main Hall' },
  ];

  useEffect(() => {
    loadCameras();
  }, []);

  const loadCameras = async () => {
    try {
      // Try to fetch camera list from backend
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
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="border-4 border-blue-600 border-t-transparent rounded-full w-16 h-16 animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading cameras...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-md sticky top-0 z-50">
        <div className="max-w-screen-2xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo & Title */}
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg flex items-center justify-center text-white text-2xl font-bold shadow-lg">
                M
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">MetroEye Dashboard</h1>
                <p className="text-sm text-gray-600">Real-time Platform Monitoring System</p>
              </div>
            </div>

            {/* View Mode Selector */}
            <div className="flex items-center gap-2 bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setViewMode('grid')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  viewMode === 'grid'
                    ? 'bg-white text-blue-600 shadow'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <span className="mr-2">▦</span>
                Grid
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  viewMode === 'list'
                    ? 'bg-white text-blue-600 shadow'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <span className="mr-2">☰</span>
                List
              </button>
            </div>

            {/* Stats Summary */}
            <div className="hidden lg:flex items-center gap-6 bg-blue-50 px-6 py-3 rounded-lg border border-blue-200">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{cameras.length}</div>
                <div className="text-xs text-gray-600">Cameras</div>
              </div>
              <div className="w-px h-8 bg-blue-300"></div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">●</div>
                <div className="text-xs text-gray-600">All Active</div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-screen-2xl mx-auto px-6 py-6">
        {/* Quick Actions */}
        <div className="mb-6 flex items-center justify-between bg-white rounded-lg shadow px-6 py-4">
          <div className="flex items-center gap-4">
            <div className="text-sm text-gray-600">
              Monitoring <span className="font-semibold text-gray-900">{cameras.length}</span> camera
              {cameras.length !== 1 ? 's' : ''}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <a
              href="/calibration"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
            >
              🎯 Calibration
            </a>
            <a
              href="/alerts"
              className="px-4 py-2 bg-orange-600 text-white rounded-lg text-sm font-medium hover:bg-orange-700 transition-colors"
            >
              ⚠️ Alerts
            </a>
          </div>
        </div>

        {/* Camera Grid */}
        {viewMode === 'grid' && (
          <div className={`grid ${getGridClass()} gap-6`}>
            {cameras.map((camera) => (
              <div
                key={camera.id}
                className="transform transition-all hover:scale-105 cursor-pointer"
                onClick={() => {
                  setFocusedCamera(camera.id);
                  setViewMode('focus');
                }}
              >
                <LiveFeed cameraId={camera.id} showStats={true} />
                <div className="mt-2 px-4 py-2 bg-white rounded-lg shadow-sm">
                  <div className="font-semibold text-gray-800">{camera.name}</div>
                  <div className="text-xs text-gray-500">{camera.location}</div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* List View */}
        {viewMode === 'list' && (
          <div className="space-y-4">
            {cameras.map((camera) => (
              <div key={camera.id} className="bg-white rounded-lg shadow-lg p-6">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <LiveFeed cameraId={camera.id} showStats={false} />
                  </div>
                  <div className="flex flex-col justify-between">
                    <div>
                      <h3 className="text-xl font-bold text-gray-900 mb-2">{camera.name}</h3>
                      <p className="text-gray-600 mb-4">{camera.location}</p>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Status:</span>
                          <span className="font-semibold text-green-600">● Active</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Camera ID:</span>
                          <span className="font-mono text-gray-800">{camera.id}</span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        setFocusedCamera(camera.id);
                        setViewMode('focus');
                      }}
                      className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      View Full Screen
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Focus View (Single Camera Fullscreen) */}
        {viewMode === 'focus' && focusedCamera && (
          <div className="fixed inset-0 bg-black bg-opacity-95 z-50 flex items-center justify-center p-6">
            <div className="w-full max-w-7xl">
              <div className="mb-4 flex justify-between items-center">
                <h2 className="text-2xl font-bold text-white">
                  {cameras.find((c) => c.id === focusedCamera)?.name}
                </h2>
                <button
                  onClick={() => setViewMode('grid')}
                  className="px-6 py-2 bg-white text-gray-900 rounded-lg hover:bg-gray-100 transition-colors font-medium"
                >
                  ✕ Close
                </button>
              </div>
              <LiveFeed cameraId={focusedCamera} showStats={true} className="shadow-2xl" />
            </div>
          </div>
        )}

        {/* Empty State */}
        {cameras.length === 0 && (
          <div className="text-center py-20">
            <div className="text-6xl mb-4">📹</div>
            <h3 className="text-xl font-semibold text-gray-800 mb-2">No Cameras Available</h3>
            <p className="text-gray-600 mb-6">Add cameras to start monitoring</p>
            <button className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
              Add Camera
            </button>
          </div>
        )}
      </main>

      {/* Footer Info */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-screen-2xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div>
              <span className="font-semibold">MetroEye</span> - AI-Powered Platform Safety System
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              System Operational
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
