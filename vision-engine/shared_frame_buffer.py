"""
Shared Memory Frame Buffer for Multi-Process Video Streaming
==============================================================

Provides zero-copy frame transfer between camera processes and main FastAPI server.
Uses double buffering for lock-free reads.

Usage:
    # In camera process
    buffer = SharedFrameBuffer.create("camera_1", 1920, 1080)
    buffer.write_frame(frame)

    # In main process
    buffer = SharedFrameBuffer.attach("camera_1", 1920, 1080)
    frame = buffer.read_frame()
"""

import numpy as np
from multiprocessing import shared_memory, Lock
import struct
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SharedFrameBuffer:
    """
    Thread-safe and process-safe frame buffer using shared memory

    Memory Layout:
    - 4 bytes: write_index (which buffer is being written to: 0 or 1)
    - 4 bytes: frame_width
    - 4 bytes: frame_height
    - Buffer 0: width * height * 3 bytes (BGR frame)
    - Buffer 1: width * height * 3 bytes (BGR frame)
    """

    HEADER_SIZE = 12  # 3 * 4 bytes for metadata

    def __init__(self, name: str, width: int, height: int, create: bool = False):
        """
        Initialize shared frame buffer

        Args:
            name: Unique identifier for this buffer
            width: Frame width in pixels
            height: Frame height in pixels
            create: If True, create new shared memory; if False, attach to existing
        """
        self.name = f"metroye_frame_{name}"
        self.width = width
        self.height = height
        self.frame_size = width * height * 3  # BGR format
        self.total_size = self.HEADER_SIZE + (2 * self.frame_size)  # Double buffer

        try:
            if create:
                # Create new shared memory
                self.shm = shared_memory.SharedMemory(
                    name=self.name,
                    create=True,
                    size=self.total_size
                )
                # Initialize header
                self._write_header(0, width, height)
                logger.info(f"✅ Created shared memory buffer: {self.name} ({self.total_size / (1024*1024):.2f} MB)")
            else:
                # Attach to existing shared memory
                self.shm = shared_memory.SharedMemory(
                    name=self.name,
                    create=False
                )
                # Read dimensions from header
                _, stored_width, stored_height = self._read_header()
                if stored_width != width or stored_height != height:
                    logger.warning(f"Dimension mismatch in {self.name}: expected {width}x{height}, got {stored_width}x{stored_height}")
                logger.info(f"✅ Attached to shared memory buffer: {self.name}")

        except FileExistsError:
            # Shared memory already exists, attach to it
            self.shm = shared_memory.SharedMemory(name=self.name)
            logger.info(f"✅ Attached to existing shared memory: {self.name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize shared memory {self.name}: {e}")
            raise

        # Create lock for write operations (not needed for reads due to double buffering)
        self.write_lock = Lock()

    @classmethod
    def create(cls, name: str, width: int, height: int) -> 'SharedFrameBuffer':
        """Factory method to create new shared memory buffer"""
        return cls(name, width, height, create=True)

    @classmethod
    def attach(cls, name: str, width: int, height: int) -> 'SharedFrameBuffer':
        """Factory method to attach to existing shared memory buffer"""
        return cls(name, width, height, create=False)

    def _write_header(self, write_index: int, width: int, height: int):
        """Write metadata to header"""
        struct.pack_into('III', self.shm.buf, 0, write_index, width, height)

    def _read_header(self) -> Tuple[int, int, int]:
        """Read metadata from header"""
        return struct.unpack_from('III', self.shm.buf, 0)

    def _get_write_index(self) -> int:
        """Get current write buffer index (0 or 1)"""
        return struct.unpack_from('I', self.shm.buf, 0)[0]

    def _set_write_index(self, index: int):
        """Set write buffer index (0 or 1)"""
        struct.pack_into('I', self.shm.buf, 0, index)

    def _get_buffer_offset(self, buffer_index: int) -> int:
        """Get byte offset for buffer 0 or 1"""
        return self.HEADER_SIZE + (buffer_index * self.frame_size)

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write frame to shared memory (non-blocking, double-buffered)

        Args:
            frame: NumPy array of shape (height, width, 3) in BGR format

        Returns:
            True if write successful, False otherwise
        """
        try:
            # Validate frame dimensions
            if frame.shape != (self.height, self.width, 3):
                logger.error(f"Frame dimension mismatch: expected ({self.height}, {self.width}, 3), got {frame.shape}")
                return False

            with self.write_lock:
                # Get current write index and flip it
                current_write_index = self._get_write_index()
                next_write_index = 1 - current_write_index  # Flip 0->1 or 1->0

                # Write frame to the inactive buffer
                offset = self._get_buffer_offset(next_write_index)
                frame_bytes = frame.tobytes()
                self.shm.buf[offset:offset + self.frame_size] = frame_bytes

                # Atomically update write index to make new frame visible
                self._set_write_index(next_write_index)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to write frame to {self.name}: {e}")
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read frame from shared memory (lock-free for readers)

        Returns:
            NumPy array of shape (height, width, 3) in BGR format, or None if error
        """
        try:
            # Read the current write index (this tells us which buffer has the latest frame)
            write_index = self._get_write_index()

            # Read from the buffer that was last written to
            offset = self._get_buffer_offset(write_index)
            frame_bytes = bytes(self.shm.buf[offset:offset + self.frame_size])

            # Convert back to numpy array
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )

            # Return a copy to avoid data corruption if frame is modified
            return frame.copy()

        except Exception as e:
            logger.error(f"❌ Failed to read frame from {self.name}: {e}")
            return None

    def read_frame_nocopy(self) -> Optional[np.ndarray]:
        """
        Read frame without copying (zero-copy, but read-only!)

        WARNING: Do NOT modify the returned array! It's a view into shared memory.

        Returns:
            Read-only NumPy array, or None if error
        """
        try:
            write_index = self._get_write_index()
            offset = self._get_buffer_offset(write_index)

            # Create read-only view directly into shared memory
            frame = np.ndarray(
                (self.height, self.width, 3),
                dtype=np.uint8,
                buffer=self.shm.buf,
                offset=offset
            )

            # Make it read-only to prevent accidental modification
            frame.flags.writeable = False

            return frame

        except Exception as e:
            logger.error(f"❌ Failed to read frame (no-copy) from {self.name}: {e}")
            return None

    def close(self):
        """Close shared memory (call from both processes when done)"""
        try:
            self.shm.close()
            logger.info(f"✅ Closed shared memory: {self.name}")
        except Exception as e:
            logger.error(f"❌ Failed to close shared memory {self.name}: {e}")

    def unlink(self):
        """Destroy shared memory (call only from creator process)"""
        try:
            self.shm.unlink()
            logger.info(f"✅ Destroyed shared memory: {self.name}")
        except Exception as e:
            logger.error(f"❌ Failed to unlink shared memory {self.name}: {e}")

    def __del__(self):
        """Cleanup on garbage collection"""
        try:
            if hasattr(self, 'shm'):
                self.shm.close()
        except:
            pass


# Convenience functions for common operations

def create_frame_buffer(camera_id: str, width: int = 1920, height: int = 1080) -> SharedFrameBuffer:
    """Create a new shared frame buffer for a camera"""
    return SharedFrameBuffer.create(camera_id, width, height)


def attach_frame_buffer(camera_id: str, width: int = 1920, height: int = 1080) -> SharedFrameBuffer:
    """Attach to an existing shared frame buffer"""
    return SharedFrameBuffer.attach(camera_id, width, height)


def cleanup_frame_buffer(buffer: SharedFrameBuffer):
    """Properly cleanup a frame buffer"""
    buffer.close()
    buffer.unlink()
