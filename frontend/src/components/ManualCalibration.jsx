/**
 * Manual Calibration Component for MetroEye Platform Edge Detection
 *
 * Features:
 * - Live MJPEG stream display from Python FastAPI
 * - Canvas overlay for clicking calibration points
 * - Automatic coordinate normalization (0.0-1.0 range)
 * - Manual and auto-calibration support
 * - Real-time visual feedback
 * - Saves to Node.js backend (MongoDB)
 */

import { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const PYTHON_SERVICE_URL = process.env.REACT_APP_PYTHON_URL || 'http://localhost:5000';

export default function ManualCalibration({ cameraId }) {
  // State management
  const [points, setPoints] = useState([]);
  const [videoDimensions, setVideoDimensions] = useState(null);
  const [status, setStatus] = useState('Ready to calibrate');
  const [isLoading, setIsLoading] = useState(false);
  const [existingCalibration, setExistingCalibration] = useState(null);
  const [streamLoaded, setStreamLoaded] = useState(false);
  const [calibrationType, setCalibrationType] = useState(null); // 'manual' or 'auto'
  const [saveFeedback, setSaveFeedback] = useState(null); // Visual save feedback

  // Refs for DOM elements
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Stream URL from Python FastAPI
  const streamUrl = `${PYTHON_SERVICE_URL}/stream/${cameraId}`;

  /**
   * Load existing calibration on mount
   */
  useEffect(() => {
    loadExistingCalibration();
  }, [cameraId]);

  /**
   * Update canvas when video dimensions change
   */
  useEffect(() => {
    if (videoDimensions && canvasRef.current) {
      const canvas = canvasRef.current;
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    }
  }, [videoDimensions]);

  /**
   * Redraw points when they change
   */
  useEffect(() => {
    if (points.length > 0 && videoDimensions) {
      drawPoints(points);
    }
  }, [points, videoDimensions]);

  /**
   * Load existing calibration from backend
   */
  const loadExistingCalibration = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/calibration/${cameraId}`);
      if (response.data.calibrated) {
        setExistingCalibration(response.data.data);
        setStatus(`Loaded ${response.data.data.detection_method} calibration (${response.data.data.num_points} points)`);
      }
    } catch (error) {
      console.log('No existing calibration found');
    }
  };

  /**
   * Handle video stream load to get dimensions
   */
  const handleStreamLoad = (e) => {
    const img = e.target;

    // Wait for image to have natural dimensions
    const checkDimensions = () => {
      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        setVideoDimensions({
          width: img.naturalWidth,
          height: img.naturalHeight
        });
        setStreamLoaded(true);
        setStatus(`Stream loaded: ${img.naturalWidth}x${img.naturalHeight}`);

        // Initialize canvas
        if (canvasRef.current) {
          canvasRef.current.width = canvasRef.current.offsetWidth;
          canvasRef.current.height = canvasRef.current.offsetHeight;
        }
      } else {
        // Keep checking until dimensions are available
        setTimeout(checkDimensions, 100);
      }
    };

    checkDimensions();
  };

  /**
   * Handle canvas click to add calibration point
   */
  const handleCanvasClick = (e) => {
    if (!videoDimensions || !streamLoaded) {
      setStatus('⚠️ Wait for stream to load');
      return;
    }

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    // Calculate click position relative to video's natural size
    const scaleX = videoDimensions.width / rect.width;
    const scaleY = videoDimensions.height / rect.height;

    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    // Add point
    const newPoints = [...points, [x, y]];
    setPoints(newPoints);
    setStatus(`Added point ${newPoints.length}: (${x}, ${y})`);
    drawPoints(newPoints);
  };

  /**
   * Draw calibration points and lines on canvas
   */
  const drawPoints = (pointsToDraw) => {
    const canvas = canvasRef.current;
    if (!canvas || !videoDimensions) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (pointsToDraw.length === 0) return;

    // Calculate scale factors
    const scaleX = rect.width / videoDimensions.width;
    const scaleY = rect.height / videoDimensions.height;

    // Draw line connecting points
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.beginPath();

    pointsToDraw.forEach((point, idx) => {
      const x = point[0] * scaleX;
      const y = point[1] * scaleY;

      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw points as circles
    pointsToDraw.forEach((point, idx) => {
      const x = point[0] * scaleX;
      const y = point[1] * scaleY;

      // Draw circle
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = '#ff0000';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw point number
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 14px Arial';
      ctx.fillText(`${idx + 1}`, x + 10, y - 10);
    });
  };

  /**
   * Save manual calibration to backend
   */
  const handleSaveManual = async () => {
    if (points.length < 2) {
      setStatus('⚠️ Need at least 2 points for calibration');
      return;
    }

    setIsLoading(true);
    setStatus('Saving manual calibration...');
    setSaveFeedback('saving');

    try {
      const response = await axios.post(
        `${BACKEND_URL}/api/calibration/${cameraId}/manual`,
        {
          points: points,
          video_dimensions: videoDimensions,
          video_source: streamUrl
        }
      );

      setCalibrationType('manual');
      setStatus(`✅ Manual calibration saved successfully (${points.length} points)`);
      setExistingCalibration(response.data.data);
      setSaveFeedback('success');

      console.log('✅ Calibration saved:', response.data);
      console.log('  - Points:', points.length);
      console.log('  - Normalized:', response.data.data.platform_edge?.normalized);

      // Clear feedback after 3 seconds
      setTimeout(() => setSaveFeedback(null), 3000);

    } catch (error) {
      setStatus(`❌ Save failed: ${error.response?.data?.error || error.message}`);
      setSaveFeedback('error');
      console.error('Save error:', error);
      setTimeout(() => setSaveFeedback(null), 3000);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Trigger auto-calibration (calls Python YOLO/Hough detection)
   */
  const handleAutoCalibrate = async () => {
    setIsLoading(true);
    setStatus('Running auto-calibration (YOLO + Hough)...');
    setSaveFeedback('saving');

    try {
      // IMPORTANT: Clear manual points before auto-calibration
      handleReset();

      // Call Node.js which will proxy to Python
      const response = await axios.post(
        `${BACKEND_URL}/api/calibration/${cameraId}/auto`,
        {
          video_path: streamUrl,
          method: 'auto',
          num_frames: 15
        },
        { timeout: 60000 } // 60 second timeout
      );

      if (response.data.success) {
        setCalibrationType('auto');
        setStatus(`✅ Auto-calibration complete (${response.data.detection_method})`);
        setExistingCalibration(response.data.data);
        setSaveFeedback('success');

        // Load detected points for visualization
        if (response.data.data.platform_edge?.absolute) {
          const autoPoints = response.data.data.platform_edge.absolute;
          setPoints(autoPoints);
          drawPoints(autoPoints);
          console.log('✅ Auto-calibration points loaded:', autoPoints.length);
        }

        // Clear feedback after 3 seconds
        setTimeout(() => setSaveFeedback(null), 3000);
      } else {
        setStatus(`⚠️ Auto-calibration failed: ${response.data.error}`);
        setSaveFeedback('error');
        setTimeout(() => setSaveFeedback(null), 3000);
      }

    } catch (error) {
      setStatus(`❌ Auto-calibration error: ${error.response?.data?.error || error.message}`);
      setSaveFeedback('error');
      console.error('Auto-calibration error:', error);
      setTimeout(() => setSaveFeedback(null), 3000);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Reset calibration points
   */
  const handleReset = () => {
    setPoints([]);
    setCalibrationType(null);
    setSaveFeedback(null);
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setStatus('Points cleared - ready to recalibrate');
  };

  /**
   * Delete existing calibration
   */
  const handleDelete = async () => {
    if (!window.confirm('Delete existing calibration? This will clear all saved points.')) return;

    setIsLoading(true);
    try {
      await axios.delete(`${BACKEND_URL}/api/calibration/${cameraId}`);
      setExistingCalibration(null);
      setCalibrationType(null);
      setSaveFeedback(null);
      setStatus('✅ Calibration deleted - ready for new calibration');
      handleReset();
    } catch (error) {
      setStatus(`❌ Delete failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6 pb-4 border-b-2 border-gray-200">
        <h2 className="text-3xl font-bold text-gray-800">Platform Edge Calibration</h2>
        <span className="bg-blue-600 text-white px-4 py-2 rounded-full text-sm font-semibold">
          Camera: {cameraId}
        </span>
      </div>

      {/* Video Stream with Canvas Overlay */}
      <div className="relative w-full max-w-4xl mx-auto mb-6 bg-black rounded-lg overflow-hidden shadow-xl">
        <img
          ref={videoRef}
          src={streamUrl}
          className="w-full block rounded-lg"
          alt="Live camera stream"
          onLoad={handleStreamLoad}
          onError={() => setStatus('❌ Stream failed to load')}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full cursor-crosshair z-10"
          onClick={handleCanvasClick}
        />
        {!streamLoaded && (
          <div className="absolute top-0 left-0 w-full h-full bg-black bg-opacity-70 flex flex-col justify-center items-center text-white z-20">
            <div className="border-4 border-white border-t-transparent rounded-full w-12 h-12 animate-spin mb-4"></div>
            <p className="text-lg">Loading stream...</p>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="max-w-4xl mx-auto">
        {/* Status Bar with Save Feedback */}
        <div className="flex justify-between items-center p-4 bg-gray-50 rounded-lg mb-4 border-l-4 border-blue-600">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-gray-700">{status}</span>
            {saveFeedback === 'saving' && (
              <span className="flex items-center gap-2 text-blue-600 text-sm">
                <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                Saving...
              </span>
            )}
            {saveFeedback === 'success' && (
              <span className="flex items-center gap-2 text-green-600 text-sm font-semibold animate-pulse">
                ✅ Saved Successfully!
              </span>
            )}
            {saveFeedback === 'error' && (
              <span className="flex items-center gap-2 text-red-600 text-sm font-semibold animate-pulse">
                ❌ Save Failed
              </span>
            )}
          </div>
          {videoDimensions && (
            <span className="text-xs text-gray-600 bg-white px-3 py-1 rounded-full font-mono">
              {videoDimensions.width} × {videoDimensions.height}
            </span>
          )}
        </div>

        {/* Button Group */}
        <div className="flex flex-wrap gap-3 mb-6">
          <button
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 hover:shadow-lg transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            onClick={handleSaveManual}
            disabled={isLoading || points.length < 2}
          >
            💾 Save Manual ({points.length} points)
          </button>

          <button
            className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white font-semibold rounded-lg shadow-md hover:bg-green-700 hover:shadow-lg transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            onClick={handleAutoCalibrate}
            disabled={isLoading || !streamLoaded}
          >
            🤖 Auto Calibrate (YOLO)
          </button>

          <button
            className="flex items-center gap-2 px-6 py-3 bg-yellow-500 text-gray-900 font-semibold rounded-lg shadow-md hover:bg-yellow-600 hover:shadow-lg transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            onClick={handleReset}
            disabled={isLoading || points.length === 0}
          >
            🔄 Reset Points
          </button>

          {existingCalibration && (
            <button
              className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white font-semibold rounded-lg shadow-md hover:bg-red-700 hover:shadow-lg transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              onClick={handleDelete}
              disabled={isLoading}
            >
              🗑️ Delete Saved
            </button>
          )}
        </div>

        {/* Info Panel */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Instructions */}
          <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
            <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
              📝 Instructions:
            </h4>
            <ol className="list-decimal list-inside text-gray-700 space-y-2 text-sm">
              <li>Wait for stream to load</li>
              <li>Click on the video to mark platform edge points</li>
              <li>Add at least 2 points along the platform edge</li>
              <li>Click "Save Manual" or use "Auto Calibrate"</li>
            </ol>
          </div>

          {/* Current Points */}
          {points.length > 0 && (
            <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
              <h4 className="text-lg font-semibold text-gray-800 mb-3">Current Points:</h4>
              <pre className="bg-white p-3 rounded border border-gray-300 overflow-x-auto text-xs font-mono">
                {JSON.stringify(points, null, 2)}
              </pre>
            </div>
          )}

          {/* Existing Calibration */}
          {existingCalibration && (
            <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
              <h4 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
                ✅ Saved Calibration:
              </h4>
              <div className="space-y-2 text-sm text-gray-700">
                <p>
                  <strong className="font-semibold text-gray-900">Method:</strong>{' '}
                  {existingCalibration.detection_method}
                </p>
                <p>
                  <strong className="font-semibold text-gray-900">Points:</strong>{' '}
                  {existingCalibration.num_points}
                </p>
                <p>
                  <strong className="font-semibold text-gray-900">Resolution:</strong>{' '}
                  {existingCalibration.video_dimensions?.width} ×{' '}
                  {existingCalibration.video_dimensions?.height}
                </p>
                {existingCalibration.platform_edge?.normalized && (
                  <details className="mt-3 p-3 bg-white rounded border border-gray-300">
                    <summary className="cursor-pointer font-semibold text-blue-600 hover:text-blue-700 select-none">
                      Normalized Coordinates
                    </summary>
                    <pre className="mt-2 text-xs font-mono overflow-x-auto">
                      {JSON.stringify(existingCalibration.platform_edge.normalized, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
