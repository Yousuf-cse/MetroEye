/**
 * useTracking Hook - WebSocket connection for real-time tracking data
 *
 * Connects to Node.js backend via Socket.io to receive:
 * - Person tracking updates
 * - Object detections
 * - Risk scores
 *
 * Usage:
 *   const { trackingData, connected } = useTracking('camera_1');
 */

import { useEffect, useState } from 'react';
import io from 'socket.io-client';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

export function useTracking(cameraId) {
  const [trackingData, setTrackingData] = useState(null);
  const [connected, setConnected] = useState(false);
  const [socket, setSocket] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!cameraId) return;

    console.log(`Connecting to WebSocket for camera: ${cameraId}`);

    // Create Socket.io connection
    const newSocket = io(BACKEND_URL, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });

    // Connection established
    newSocket.on('connect', () => {
      console.log(`✅ WebSocket connected for ${cameraId}`);
      setConnected(true);
      setError(null);

      // Subscribe to tracking updates for this camera
      newSocket.emit('subscribe_tracking', cameraId);
    });

    // Receive tracking updates
    newSocket.on('tracking_update', (data) => {
      // console.log(`📊 Tracking update for ${cameraId}:`, data);
      setTrackingData(data);
    });

    // Connection error
    newSocket.on('connect_error', (err) => {
      console.error(`❌ WebSocket connection error for ${cameraId}:`, err.message);
      setConnected(false);
      setError(err.message);
    });

    // Disconnected
    newSocket.on('disconnect', (reason) => {
      console.log(`❌ WebSocket disconnected for ${cameraId}:`, reason);
      setConnected(false);
    });

    // Reconnecting
    newSocket.on('reconnect', (attemptNumber) => {
      console.log(`🔄 Reconnected to ${cameraId} after ${attemptNumber} attempts`);
      setConnected(true);
      setError(null);
      newSocket.emit('subscribe_tracking', cameraId);
    });

    setSocket(newSocket);

    // Cleanup on unmount
    return () => {
      console.log(`Disconnecting from ${cameraId}`);
      newSocket.disconnect();
    };
  }, [cameraId]);

  return {
    trackingData,
    connected,
    socket,
    error
  };
}

/**
 * useAlerts Hook - Subscribe to alert notifications
 */
export function useAlerts() {
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const socket = io(BACKEND_URL, {
      transports: ['websocket', 'polling']
    });

    socket.on('connect', () => {
      console.log('✅ Connected to alerts channel');
      setConnected(true);
      socket.emit('subscribe_alerts');
    });

    socket.on('new_alert', (alertData) => {
      console.log('⚠️ New alert:', alertData);
      setAlerts(prev => [alertData, ...prev].slice(0, 50)); // Keep last 50 alerts
    });

    socket.on('disconnect', () => {
      console.log('❌ Disconnected from alerts');
      setConnected(false);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  return { alerts, connected };
}
