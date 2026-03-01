"""
Frame Reader with Producer-Consumer Pattern
============================================

Separates frame reading from processing to prevent blocking.
Runs in a dedicated thread, continuously reading frames and queuing them.

Features:
- Non-blocking frame reading
- Automatic frame dropping when queue is full
- Auto-loop video files
- Health monitoring
"""

import cv2
import threading
import queue
import time
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class FrameReader:
    """
    Dedicated thread for reading frames from video source

    Runs continuously in background, reading frames as fast as possible.
    Old frames are dropped if processing can't keep up (queue full).
    """

    def __init__(self, video_source: str, frame_queue: queue.Queue, camera_id: str, max_queue_size: int = 3):
        """
        Initialize frame reader

        Args:
            video_source: Path to video file or camera index
            frame_queue: Queue to put frames into
            camera_id: Camera identifier for logging
            max_queue_size: Maximum frames in queue before dropping (default 3)
        """
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.camera_id = camera_id
        self.max_queue_size = max_queue_size

        self.running = False
        self.thread = None
        self.cap = None

        # Statistics
        self.frames_read = 0
        self.frames_dropped = 0
        self.last_read_time = 0
        self.read_fps = 0

        # Health monitoring
        self.last_health_check = time.time()
        self.health_check_interval = 5.0  # Log health every 5 seconds

    def start(self):
        """Start the frame reader thread"""
        if self.running:
            logger.warning(f"{self.camera_id}: Frame reader already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True, name=f"FrameReader-{self.camera_id}")
        self.thread.start()
        logger.info(f"🎬 {self.camera_id}: Frame reader started")

    def stop(self):
        """Stop the frame reader thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info(f"🛑 {self.camera_id}: Frame reader stopped")

    def _read_loop(self):
        """Main frame reading loop (runs in dedicated thread)"""
        consecutive_failures = 0
        max_consecutive_failures = 30

        # Open video capture
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            logger.error(f"❌ {self.camera_id}: Could not open video source: {self.video_source}")
            return

        # Reduce buffer size to minimize lag
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Get video FPS to control reading speed
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0 or video_fps > 60:
            video_fps = 30  # Default to 30 FPS if invalid

        # Slow down slightly (multiply by 1.15 to add 15% delay)
        frame_time = (1.0 / video_fps) * 1.1  # Time between frames with slight slowdown
        logger.info(f"✅ {self.camera_id}: Video capture opened at {video_fps:.1f} FPS (slowed to {video_fps/1.15:.1f} FPS): {self.video_source}")

        last_fps_calc = time.time()
        frames_since_last_calc = 0
        last_frame_time = time.time()

        while self.running:
            read_start = time.time()

            # Throttle reading to match video FPS (prevent fast-forward effect!)
            time_since_last_frame = read_start - last_frame_time
            if time_since_last_frame < frame_time:
                sleep_time = frame_time - time_since_last_frame
                time.sleep(sleep_time)

            # Read frame
            ret, frame = self.cap.read()

            if not ret:
                consecutive_failures += 1
                if consecutive_failures > max_consecutive_failures:
                    # End of video, loop back
                    logger.info(f"🔄 {self.camera_id}: Looping video")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    consecutive_failures = 0
                continue

            consecutive_failures = 0
            self.frames_read += 1
            frames_since_last_calc += 1
            last_frame_time = time.time()

            # Try to put frame in queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                # Queue is full, drop this frame
                self.frames_dropped += 1

                # If queue is consistently full, log warning (but less frequently)
                if self.frames_dropped % 1000 == 0:
                    logger.warning(
                        f"⚠️ {self.camera_id}: Dropped {self.frames_dropped} frames total "
                        f"(processing too slow, queue full)"
                    )

            # Calculate FPS
            now = time.time()
            if now - last_fps_calc >= 1.0:
                self.read_fps = frames_since_last_calc / (now - last_fps_calc)
                frames_since_last_calc = 0
                last_fps_calc = now

            # Health check logging
            if now - self.last_health_check > self.health_check_interval:
                self._log_health()
                self.last_health_check = now

            self.last_read_time = now

        # Cleanup
        if self.cap:
            self.cap.release()

    def _log_health(self):
        """Log health statistics"""
        drop_rate = (self.frames_dropped / self.frames_read * 100) if self.frames_read > 0 else 0

        if drop_rate > 10:
            log_func = logger.warning
            status = "⚠️"
        else:
            log_func = logger.debug
            status = "✅"

        log_func(
            f"{status} {self.camera_id} FrameReader: "
            f"FPS={self.read_fps:.1f}, "
            f"Read={self.frames_read}, "
            f"Dropped={self.frames_dropped} ({drop_rate:.1f}%), "
            f"Queue={self.frame_queue.qsize()}/{self.max_queue_size}"
        )

    def get_stats(self) -> dict:
        """Get current statistics"""
        return {
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "drop_rate": (self.frames_dropped / self.frames_read * 100) if self.frames_read > 0 else 0,
            "read_fps": self.read_fps,
            "queue_size": self.frame_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "is_running": self.running,
            "time_since_last_read": time.time() - self.last_read_time if self.last_read_time > 0 else 0
        }

    def is_healthy(self) -> bool:
        """Check if frame reader is healthy"""
        if not self.running:
            return False

        # Check if we've read frames recently (within last 5 seconds)
        time_since_last_read = time.time() - self.last_read_time if self.last_read_time > 0 else float('inf')
        if time_since_last_read > 5.0:
            return False

        # Check if drop rate is reasonable (< 50%)
        drop_rate = (self.frames_dropped / self.frames_read * 100) if self.frames_read > 0 else 0
        if drop_rate > 50:
            return False

        return True
