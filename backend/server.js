const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const http = require('http');
const socketIO = require('socket.io');
require('dotenv').config();

const alertRoutes = require('./routes/alertRoutes');
const streamingRoutes = require('./routes/streamingRoutes');
const trackingRoutes = require('./routes/trackingRoutes');
const calibrationRoutes = require('./routes/calibrationRoutes');
//test route
const testRoutes = require('./routes/testRoutes');



// Initialize Express
const app = express();
const server = http.createServer(app);

// Initialize Socket.IO
const io = socketIO(server, {
  cors: {
    origin: ['http://localhost:3000', 'http://localhost:5173'],
    methods: ['GET', 'POST']
  }
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Make io accessible to routes
app.use((req, res, next) => {
  req.io = io;
  next();
});

// Database connection
console.log('🔗 Connecting to MongoDB:', process.env.MONGODB_URI);
mongoose.connect(process.env.MONGODB_URI)
  .then(() => {
    console.log('✅ MongoDB connected successfully');
    console.log('📂 Database:', mongoose.connection.name);
    console.log('🔗 Host:', mongoose.connection.host);
    console.log('🔌 Port:', mongoose.connection.port);
  })
  .catch(err => {
    console.error('❌ MongoDB connection FAILED:', err.message);
    console.error('Full error:', err);
  });

// Routes
app.get('/', (req, res) => {
  res.json({
    status: 'Metro Detection API is running',
    version: '1.0',
    endpoints: {
      alerts: '/api/alerts',
      stats: '/api/alerts/stats',
      videos: '/api/videos/:filename',
      websocket: 'ws://localhost:8000'
    }
  });
});

app.use('/api/alerts', alertRoutes);
app.use('/api/stream', streamingRoutes);
app.use('/api/tracking', trackingRoutes);
app.use('/api/calibration', calibrationRoutes);
//test routes
app.use('/api/test', testRoutes);

// Serve video files
app.use('/api/videos', express.static(path.join(__dirname, '../shared/alert-videos')));

// WebSocket connection handling
io.on('connection', (socket) => {
  console.log('✅ Client connected:', socket.id);
  
  socket.on('subscribe_tracking', (cameraId) => {
    socket.join(`tracking_${cameraId}`);
    console.log(`📹 Client subscribed to tracking: ${cameraId}`);
  });
  
  socket.on('disconnect', () => {
    console.log('❌ Client disconnected:', socket.id);
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    error: 'Something went wrong!'
  });
});

// Start server
const PORT = process.env.PORT || 8000;
server.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📡 WebSocket server running on ws://localhost:${PORT}`);
});