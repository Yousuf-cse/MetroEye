# 🚇 MetroEye - Ai Safety Intelligence Layer for Existing Metro Infrastructure

*Real-time AI-powered surveillance for metro station safety*

---

## 🎯 Project Overview

MetroEye detects suspicious behaviors at metro/train platforms using:
- *YOLO pose estimation* for person tracking
- *Temporal pattern analysis* for behavior classification
- *Real-time alerts* with video evidence
- *Multi-camera support* for comprehensive coverage

---

## 🚀 Quick Start

### For Python Developer (You)

bash
# 1. Setup environment
cd vision-engine
pip install -r requirements.txt

# 2. Configure cameras (IMPORTANT!)
# Edit config/cameras.json with your CCTV camera RTSP URLs

# 3. Test with video file first (safer for demo)
python app.py --video sample-station.webm --camera platform_cam_1

# 4. Connect to live CCTV (after testing)
python app.py --rtsp rtsp://admin:password@192.168.1.100:554/stream1


### For Backend Developer (Your Friend)

bash
# 1. Setup Node.js
cd backend
npm install

# 2. Configure environment
cp .env.example .env
# Edit .env with database credentials, etc.

# 3. Start server
npm run dev

# 4. Test
curl http://localhost:8000/health


---


## 🎥 CCTV Camera Integration

*⚠️ IMPORTANT: This system is designed for CCTV cameras, not webcams!*

### Camera Requirements:
- IP cameras with RTSP support
- Network-accessible (same network as detection server)
- Credentials (username/password)
- Recommended: 720p or 1080p @ 15-30 FPS

### Supported Camera Brands:
- ✅ Hikvision
- ✅ Dahua
- ✅ Axis
- ✅ Any ONVIF-compliant camera

### Connection Example:
python
# Hikvision
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101

# Dahua
rtsp://admin:password@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0

# Generic
rtsp://admin:password@192.168.1.102:554/stream1


---

## 🔧 Technology Stack

### Python Service (AI/Computer Vision)
- *YOLOv8* - Object detection & pose estimation
- *OpenCV* - Video processing
- *NumPy/SciPy* - Numerical operations
- *XGBoost* - Behavior classification
- *FastAPI* (optional) - HTTP streaming

### Node.js Backend (Web API)
- *Express.js* - REST API server
- *Socket.io* - Real-time WebSocket communication
- *MongoDB* - Database for alerts

### Frontend 
- *React* - Dashboard UI
- *Canvas API* - Real-time overlay rendering
- *Socket.io-client* - WebSocket client

---

## 🎯 Features

### ✅ Implemented
- [x] YOLO-based person detection
- [x] Pose estimation (17 keypoints)
- [x] Temporal feature aggregation (sliding windows)
- [x] Rule-based risk scoring
- [x] Per-frame feature extraction
- [x] CSV data export
- [x] RTSP stream support
- [x] Multi-camera support
- [x] Automatic reconnection logic

### 🚧 In Progress (Backend)
- [ ] REST API endpoints
- [ ] WebSocket real-time streaming
- [ ] MongoDB alert storage
- [ ] Cloud video storage
- [ ] React dashboard

### 🎯 Planned
- [ ] XGBoost model training
- [ ] LLM-powered alert reasoning
- [ ] Email/SMS notifications
- [ ] Analytics dashboard

---

## 📊 Performance

### Current Benchmarks:
- *Single Camera:* 20-30 FPS (1080p)
- *4 Cameras:* 10-15 FPS each (multi-threaded)
- *Detection Latency:* <100ms per frame
- *Alert Generation:* <500ms

### System Requirements:
- *CPU:* Intel i5 or better (8th gen+)
- *RAM:* 8GB minimum, 16GB recommended
- *GPU:* Optional (NVIDIA GTX 1060+ for better FPS)
- *Network:* 100 Mbps for 4 cameras

---

## 🤝 Team Roles

### Python AI Developer (Debasis)
- YOLO detection integration
- Feature extraction
- Risk scoring logic
- CCTV camera connectivity
- Video clip recording

### Backend Developer (Yousuf + Debasis)
- Node.js API server
- WebSocket implementation
- Database management
- Cloud storage integration
- Deployment

### Frontend Developer (Yousuf)
- React dashboard
- Live video display
- Alert management UI
- Real-time overlays

---


## 🐛 Troubleshooting

### Camera won't connect?
bash
# Test RTSP URL with ffplay (VLC or ffplay)
ffplay rtsp://admin:password@192.168.1.100:554/stream1

# Check if camera is reachable
ping 192.168.1.100


### Low FPS?
python
# Reduce resolution
frame = cv2.resize(frame, (640, 480))

# Use lighter model
model = YOLO('yolo26n-pose.pt')  # nano version

# Process every Nth frame
if frame_count % 2 == 0:
    results = model.track(frame)


### Backend won't start?
bash
# Check if port is in use
netstat -ano | findstr :8000

# Use different port
PORT=8001 npm start

---

## 📞 Contact & Resources

- *Team:* The Curavture Core
- *Date:* February 2026 - March 2026
- *Event:* [Diversion 2k26]

### External Resources:
- YOLO: https://docs.ultralytics.com/
- Express.js: https://expressjs.com/
- Socket.io: https://socket.io/
- Digital Ocean: https://docs.digitalocean.com/

---

## 📄 License

MIT License (or your chosen license)

---

*Built with ❤️ for metro safety. Let's make public transport safer together! 🚇*
