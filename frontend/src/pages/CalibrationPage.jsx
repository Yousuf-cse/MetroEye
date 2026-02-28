/**
 * Calibration Page - Example usage of ManualCalibration component
 *
 * This page demonstrates how to integrate the calibration component
 * into your MetroEye application.
 */

import { useState } from 'react';
import ManualCalibration from '../components/ManualCalibration';

export default function CalibrationPage() {
  const [selectedCamera, setSelectedCamera] = useState('camera_1');

  // List of available cameras (fetch from backend in production)
  const cameras = [
    { id: 'camera_1', name: 'Platform 1 - North' },
    { id: 'camera_2', name: 'Platform 2 - South' },
    { id: 'camera_3', name: 'Platform 3 - Central' },
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Page Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">MetroEye Calibration</h1>
              <p className="text-sm text-gray-600 mt-1">
                Configure platform edge detection for accurate tracking
              </p>
            </div>
            <a
              href="/"
              className="text-blue-600 hover:text-blue-700 font-medium text-sm flex items-center gap-2"
            >
              ← Back to Dashboard
            </a>
          </div>
        </div>
      </header>

      {/* Camera Selector */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <label className="block text-sm font-semibold text-gray-700 mb-3">
            Select Camera to Calibrate:
          </label>
          <div className="flex flex-wrap gap-3">
            {cameras.map((camera) => (
              <button
                key={camera.id}
                onClick={() => setSelectedCamera(camera.id)}
                className={`px-6 py-3 rounded-lg font-medium transition-all ${
                  selectedCamera === camera.id
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {camera.name}
              </button>
            ))}
          </div>
        </div>

        {/* Calibration Component */}
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <ManualCalibration cameraId={selectedCamera} />
        </div>

        {/* Help Section */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">
            🎯 Calibration Tips
          </h3>
          <ul className="space-y-2 text-sm text-blue-800">
            <li className="flex items-start gap-2">
              <span className="font-bold">•</span>
              <span>
                <strong>Manual Calibration:</strong> Click along the platform edge to create a
                boundary line. Use at least 2 points for best results.
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold">•</span>
              <span>
                <strong>Auto Calibration:</strong> Uses YOLO AI detection to automatically find
                the platform edge. Works best with clear platform visibility.
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold">•</span>
              <span>
                <strong>Resolution Independence:</strong> Calibration coordinates are normalized,
                so they work at any display resolution.
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="font-bold">•</span>
              <span>
                <strong>Re-calibration:</strong> You can recalibrate anytime by clicking "Reset
                Points" and starting over.
              </span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
