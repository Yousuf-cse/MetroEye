import React, { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';

const VideoStream = ({ cameraId }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [trackingData, setTrackingData] = useState([]);
  const [streamStatus, setStreamStatus] = useState('Loading...');

  useEffect(() => {
    // Connect to WebSocket
    const socket = io(process.env.REACT_APP_WS_URL);
    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('✅ WebSocket connected');
      setIsConnected(true);
      socket.emit('subscribe_tracking', cameraId);
    });

    socket.on('disconnect', () => {
      console.log('❌ WebSocket disconnected');
      setIsConnected(false);
    });

    // Receive tracking updates
    socket.on('tracking_update', (data) => {
      if (data.camera_id === cameraId) {
        setTrackingData(data.objects || []);
        drawOverlay(data.objects || []);
      }
    });

    return () => {
      socket.emit('unsubscribe_tracking', cameraId);
      socket.disconnect();
    };
  }, [cameraId]);

  // Update canvas size when video loads
  useEffect(() => {
    const video = videoRef.current;
    if (video) {
      video.onload = () => {
        setStreamStatus('Active');
        const canvas = canvasRef.current;
        if (canvas) {
          canvas.width = video.naturalWidth || video.width;
          canvas.height = video.naturalHeight || video.height;
        }
      };
      
      video.onerror = () => {
        setStreamStatus('Error');
      };
    }
  }, []);

  // Draw bounding boxes and labels
  const drawOverlay = (objects) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw each object
    objects.forEach((obj) => {
      const [x1, y1, x2, y2] = obj.bbox;
      const width = x2 - x1;
      const height = y2 - y1;

      // Color based on risk score
      let color;
      if (obj.risk_score > 70) {
        color = '#ff0000'; // Red - High risk
      } else if (obj.risk_score > 40) {
        color = '#ff9800'; // Orange - Medium risk
      } else {
        color = '#4caf50'; // Green - Low risk
      }

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, width, height);

      // Draw label background
      ctx.fillStyle = color;
      const labelText = `ID:${obj.track_id} Risk:${obj.risk_score}`;
      const textWidth = ctx.measureText(labelText).width + 10;
      ctx.fillRect(x1, y1 - 25, textWidth, 25);

      // Draw label text
      ctx.fillStyle = 'white';
      ctx.font = 'bold 14px Arial';
      ctx.fillText(labelText, x1 + 5, y1 - 8);

      // Draw keypoints if available
      if (obj.keypoints && obj.keypoints.length > 0) {
        ctx.fillStyle = color;
        obj.keypoints.forEach(([x, y]) => {
          if (x > 0 && y > 0) {
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
          }
        });
      }
    });
  };

  const streamUrl = `${process.env.REACT_APP_API_URL}/api/stream/${cameraId}/stream`;

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <div style={{ marginBottom: '10px' }}>
        <span style={{ 
          padding: '5px 10px', 
          borderRadius: '5px',
          background: isConnected ? '#4caf50' : '#f44336',
          color: 'white',
          marginRight: '10px'
        }}>
          {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
        </span>
        <span style={{ 
          padding: '5px 10px', 
          borderRadius: '5px',
          background: streamStatus === 'Active' ? '#4caf50' : '#ff9800',
          color: 'white'
        }}>
          Stream: {streamStatus}
        </span>
      </div>

      <div style={{ position: 'relative' }}>
        <img
          ref={videoRef}
          src={streamUrl}
          alt="Camera stream"
          style={{
            width: '100%',
            maxWidth: '640px',
            border: '2px solid #333',
            display: 'block'
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            pointerEvents: 'none'
          }}
        />
      </div>

      {/* Tracking Info */}
      <div style={{ 
        marginTop: '10px', 
        padding: '10px', 
        background: '#f5f5f5',
        borderRadius: '5px'
      }}>
        <strong>Tracked Objects: {trackingData.length}</strong>
        {trackingData.length > 0 && (
          <div style={{ marginTop: '10px' }}>
            {trackingData.map((obj) => (
              <div 
                key={obj.track_id}
                style={{
                  display: 'inline-block',
                  padding: '5px 10px',
                  margin: '5px',
                  background: obj.risk_score > 70 ? '#ffebee' : '#e8f5e9',
                  border: `2px solid ${obj.risk_score > 70 ? '#f44336' : '#4caf50'}`,
                  borderRadius: '5px',
                  fontSize: '12px'
                }}
              >
                <strong>#{obj.track_id}</strong> | 
                Risk: {obj.risk_score} | 
                Speed: {obj.speed?.toFixed(1) || 0}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoStream;