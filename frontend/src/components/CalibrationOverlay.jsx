/**
 * Calibration Overlay Component
 *
 * Displays calibration points/lines on top of live feed
 * Shows both manual and auto-calibration results
 */

import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

export default function CalibrationOverlay({ cameraId, streamWidth, streamHeight }) {
  const [calibration, setCalibration] = useState(null);
  const [displayPoints, setDisplayPoints] = useState([]);
  const canvasRef = useRef(null);

  /**
   * Load calibration data for this camera
   */
  useEffect(() => {
    if (!cameraId || !streamWidth || !streamHeight) return;

    loadCalibration();

    // Reload every 5 seconds to catch updates
    const interval = setInterval(loadCalibration, 5000);
    return () => clearInterval(interval);
  }, [cameraId, streamWidth, streamHeight]);

  /**
   * Redraw when points or canvas size changes
   */
  useEffect(() => {
    if (displayPoints.length > 0 && canvasRef.current) {
      drawCalibrationLine();
    }
  }, [displayPoints, streamWidth, streamHeight]);

  /**
   * Load calibration from backend
   */
  const loadCalibration = async () => {
    try {
      const response = await axios.get(
        `${BACKEND_URL}/api/calibration/${cameraId}/for-display`,
        {
          params: {
            width: streamWidth,
            height: streamHeight
          }
        }
      );

      if (response.data.points) {
        setDisplayPoints(response.data.points);
        setCalibration(response.data);
        console.log(`✅ Calibration loaded for ${cameraId}:`, response.data.points.length, 'points');
      }
    } catch (error) {
      // No calibration exists - this is normal
      if (error.response?.status !== 404) {
        console.error('Error loading calibration:', error);
      }
      setDisplayPoints([]);
      setCalibration(null);
    }
  };

  /**
   * Draw calibration line on canvas
   */
  const drawCalibrationLine = () => {
    const canvas = canvasRef.current;
    if (!canvas || displayPoints.length === 0) return;

    const ctx = canvas.getContext('2d');

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set canvas size to match stream
    canvas.width = streamWidth;
    canvas.height = streamHeight;

    // Draw line connecting points
    ctx.strokeStyle = '#00ff00'; // Green line
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 5]); // Dashed line
    ctx.shadowColor = '#00ff00';
    ctx.shadowBlur = 10;

    ctx.beginPath();
    displayPoints.forEach((point, idx) => {
      const [x, y] = point;
      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Reset shadow for circles
    ctx.shadowBlur = 0;

    // Draw points as circles
    displayPoints.forEach((point, idx) => {
      const [x, y] = point;

      // Circle
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = '#ff0000'; // Red dot
      ctx.fill();
      ctx.strokeStyle = '#ffffff'; // White outline
      ctx.lineWidth = 2;
      ctx.setLineDash([]); // Solid line for circle
      ctx.stroke();

      // Point number
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px monospace';
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 3;
      ctx.strokeText(`${idx + 1}`, x + 10, y - 10);
      ctx.fillText(`${idx + 1}`, x + 10, y - 10);
    });
  };

  // If no calibration, don't render
  if (!calibration || displayPoints.length === 0) {
    return null;
  }

  return (
    <>
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 pointer-events-none z-10"
        style={{
          width: '100%',
          height: '100%'
        }}
      />

      {/* Calibration info badge */}
      <div
        className="absolute top-2 left-2 z-20 pointer-events-none"
        style={{
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          border: '1px solid #00ff00',
          padding: '4px 8px',
          borderRadius: '2px',
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: '0.65rem',
          color: '#00ff00',
          letterSpacing: '0.05em',
          textTransform: 'uppercase'
        }}
      >
        ✓ Calibrated ({displayPoints.length} pts)
      </div>
    </>
  );
}
