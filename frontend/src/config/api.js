// API Configuration
// Automatically uses .env.production or .env.development based on build

const config = {
  backendUrl: process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000',
  wsUrl: process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
  visionEngineUrl: process.env.REACT_APP_VISION_ENGINE_URL || 'http://localhost:5000',
};

export default config;
