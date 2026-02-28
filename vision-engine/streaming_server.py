"""
FastAPI Streaming Server for MetroEye Platform Detection
========================================================

Provides HTTP/MJPEG streaming of processed video frames and calibration endpoints.
Integrates with YOLO detection and runs detection in background thread.

Features:
- MJPEG streaming of annotated frames
- Calibration endpoints (auto/yolo/hough)
- Real-time person tracking and risk detection
- Thread-safe frame updates

Usage:
    python streaming_server.py

Then access:
    - Video stream: http://localhost:5000/stream/camera_1
    - Calibration: http://localhost:5000/calibrate/auto/camera_1
    - Health check: http://localhost:5000/health
"""

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import asyncio
import numpy as np
from typing import Dict
import threading
import time
import logging
import httpx  # For sending data to Node.js

# Import calibration router
from calibration_endpoints import router as calibration_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MetroEye Vision Engine API",
    description="Video streaming and calibration endpoints for platform edge detection",
    version="1.0.0"
)

# Enable CORS for Node.js backend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Node.js backend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include calibration router
app.include_router(calibration_router)

# Store latest frames for each camera (thread-safe)
latest_frames: Dict[str, np.ndarray] = {}
frame_locks: Dict[str, threading.Lock] = {}

# Node.js backend configuration
NODE_BACKEND_URL = "http://localhost:8000"  # Change for production
http_client = httpx.Client(timeout=1.0)  # 1 second timeout for non-blocking


def update_frame(camera_id: str, frame: np.ndarray):
    """
    Update the latest frame for a camera (thread-safe)

    Call this from your detection loop to update the stream.

    Args:
        camera_id: Unique identifier for camera
        frame: Annotated frame (numpy array)
    """
    if camera_id not in frame_locks:
        frame_locks[camera_id] = threading.Lock()

    with frame_locks[camera_id]:
        latest_frames[camera_id] = frame.copy()


def get_latest_frame(camera_id: str) -> np.ndarray:
    """
    Get the latest frame for a camera (thread-safe)

    Args:
        camera_id: Unique identifier for camera

    Returns:
        frame: Latest annotated frame, or None if not available
    """
    if camera_id not in frame_locks:
        return None

    with frame_locks[camera_id]:
        return latest_frames.get(camera_id, None)


def extract_tracking_data(result, camera_id: str) -> dict:
    """
    Extract tracking data from YOLO results

    Args:
        result: YOLO result object
        camera_id: Camera identifier

    Returns:
        dict: Tracking data formatted for Node.js backend
    """
    objects = []

    if result.boxes is not None:
        boxes = result.boxes.cpu().numpy()

        for i, box in enumerate(boxes):
            obj = {
                "track_id": int(box.id[0]) if box.id is not None else i,
                "bbox": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class": "person"
            }

            # Add pose keypoints if available
            if result.keypoints is not None:
                kp = result.keypoints.cpu().numpy()
                if i < len(kp.data):
                    obj["keypoints"] = kp.data[i].tolist()

            objects.append(obj)

    return {
        "camera_id": camera_id,
        "timestamp": time.time(),
        "frame_count": len(objects),
        "objects": objects
    }


def send_tracking_data(camera_id: str, tracking_data: dict):
    """
    Send tracking data to Node.js backend (non-blocking)

    Args:
        camera_id: Camera identifier
        tracking_data: Detection results
    """
    try:
        response = http_client.post(
            f"{NODE_BACKEND_URL}/api/tracking/{camera_id}",
            json=tracking_data
        )
        if response.status_code != 200:
            logger.warning(f"Failed to send tracking data: {response.status_code}")
    except Exception as e:
        # Don't block on errors - tracking data is best-effort
        logger.debug(f"Error sending tracking data: {e}")


async def frame_generator(camera_id: str):
    """
    Generator that yields frames as MJPEG stream

    Args:
        camera_id: Camera to stream

    Yields:
        MJPEG frame bytes
    """
    logger.info(f"Starting stream for camera: {camera_id}")

    while True:
        frame = get_latest_frame(camera_id)

        if frame is None:
            # No frame yet, wait and try again
            await asyncio.sleep(0.033)  # ~30 FPS
            continue

        # Encode frame as JPEG with optimized settings for speed
        _, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, 60,  # Reduced quality for faster encoding/streaming
            cv2.IMWRITE_JPEG_OPTIMIZE, 1   # Enable JPEG optimization
        ])
        frame_bytes = buffer.tobytes()

        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        await asyncio.sleep(0.02)  # ~50 FPS (faster streaming)


@app.get("/stream/{camera_id}")
async def stream_camera(camera_id: str):
    """
    Stream endpoint for a specific camera

    Returns:
        MJPEG stream (multipart/x-mixed-replace)

    Usage:
        Access http://localhost:5000/stream/camera_1 in browser or <img> tag
    """
    return StreamingResponse(
        frame_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/frame/{camera_id}")
async def get_frame(camera_id: str):
    """
    Get single latest frame (for testing)

    Returns:
        JPEG image
    """
    frame = get_latest_frame(camera_id)

    if frame is None:
        return Response(
            content=b'',
            status_code=404,
            media_type="text/plain"
        )

    _, buffer = cv2.imencode('.jpg', frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/health")
async def health():
    """
    Health check endpoint

    Returns:
        Status and list of active cameras
    """
    return {
        "status": "ok",
        "active_cameras": list(latest_frames.keys()),
        "timestamp": time.time(),
        "frame_count": {cam: 1 if get_latest_frame(cam) is not None else 0
                       for cam in latest_frames.keys()}
    }


@app.get("/")
async def root():
    """
    Root endpoint - API information
    """
    return {
        "service": "MetroEye Vision Engine",
        "version": "1.0.0",
        "endpoints": {
            "streaming": "/stream/{camera_id}",
            "frame": "/frame/{camera_id}",
            "calibration_auto": "/calibrate/auto/{camera_id}",
            "calibration_yolo": "/calibrate/yolo/{camera_id}",
            "calibration_hough": "/calibrate/hough/{camera_id}",
            "health": "/health"
        },
        "docs": "/docs"
    }


# Detection loop (runs in background thread)
def run_detection(camera_id: str, video_source: str):
    """
    Main detection loop - runs YOLO tracking and updates frames

    Args:
        camera_id: Unique identifier for this camera
        video_source: Path to video file or RTSP URL
    """
    from ultralytics import YOLO

    logger.info(f"Starting detection for {camera_id}: {video_source}")

    # Load YOLO model
    model = YOLO('yolo26n-pose.pt')

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        logger.error(f"Could not open video source: {video_source}")
        return

    frame_counter = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            logger.warning(f"End of video or read error for {camera_id}")
            # Loop video if it's a file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Run YOLO detection
        results = model.track(frame, persist=True, conf=0.3)

        # Draw annotations
        annotated_frame = results[0].plot()

        # Update frame for streaming
        update_frame(camera_id, annotated_frame)

        # Extract and send tracking data every 10 frames (reduce API calls)
        frame_counter += 1
        if frame_counter % 10 == 0:
            tracking_data = extract_tracking_data(results[0], camera_id)

            # Send to Node.js backend (non-blocking)
            if tracking_data['objects']:
                send_tracking_data(camera_id, tracking_data)

        # Small sleep to prevent CPU overload
        time.sleep(0.001)

    cap.release()


def start_detection_thread(camera_id: str, video_source: str):
    """
    Start detection in background thread

    Args:
        camera_id: Unique identifier
        video_source: Video file path or RTSP URL
    """
    detection_thread = threading.Thread(
        target=run_detection,
        args=(camera_id, video_source),
        daemon=True
    )
    detection_thread.start()
    logger.info(f"Detection thread started for {camera_id}")


if __name__ == "__main__":
    import uvicorn

    # Start detection for multiple cameras (for multi-camera dashboard demo)
    # Modify video sources as needed
    cameras = [
        ("camera_1", "sample-station.webm"),
        # Uncomment to add more cameras:
        # ("camera_2", "sample-station.webm"),  # Use different video sources
        # ("camera_3", "rtsp://your-camera-url"),
    ]

    logger.info("=" * 60)
    logger.info("Starting detection threads for cameras...")
    for camera_id, video_source in cameras:
        start_detection_thread(camera_id, video_source)
        logger.info(f"  ✅ {camera_id}: {video_source}")
    logger.info("=" * 60)

    # Start FastAPI server
    logger.info("MetroEye Vision Engine API Server")
    logger.info("=" * 60)
    logger.info("📹 Video streams:")
    for camera_id, _ in cameras:
        logger.info(f"    http://localhost:5000/stream/{camera_id}")
    logger.info(f"📊 API docs: http://localhost:5000/docs")
    logger.info(f"🏥 Health check: http://localhost:5000/health")
    logger.info(f"🔗 Node.js backend: {NODE_BACKEND_URL}")
    logger.info("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=5000)
