"""
FastAPI Calibration Endpoints for Platform Edge Detection
=========================================================

Provides HTTP endpoints for automatic and manual platform edge calibration.
Integrates with existing FastAPI streaming server.

Endpoints:
- POST /calibrate/auto/{camera_id} - Auto-detect using YOLO + Hough
- POST /calibrate/yolo/{camera_id} - YOLO segmentation only
- POST /calibrate/hough/{camera_id} - Hough Transform only

All endpoints return normalized coordinates (0.0-1.0) for resolution independence.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
import logging

from edge_calibrator import YOLOSegEdgeDetector, HoughEdgeDetector

# Create router
router = APIRouter(prefix="/calibrate", tags=["calibration"])

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationRequest(BaseModel):
    """Request body for calibration"""
    video_source: Optional[str] = None  # If None, uses current live frame
    num_frames: int = 15
    visualize: bool = False


class CalibrationResponse(BaseModel):
    """Response structure for calibration"""
    success: bool
    points: List[List[float]]  # Normalized coordinates [[x, y], ...]
    absolute_points: List[List[int]]  # Original pixel coordinates
    video_dimensions: dict  # {width: int, height: int}
    detection_method: str  # 'yolo-seg', 'hough-transform', or 'auto'
    num_points: int


def normalize_points(points: List[tuple], width: int, height: int) -> List[List[float]]:
    """
    Normalize pixel coordinates to 0.0-1.0 range

    Args:
        points: List of (x, y) tuples in pixels
        width: Frame width in pixels
        height: Frame height in pixels

    Returns:
        List of [x, y] pairs normalized to 0.0-1.0
    """
    return [[round(x / width, 4), round(y / height, 4)] for x, y in points]


def get_frame_from_source(video_source: str = None, camera_id: str = None):
    """
    Get a frame from video source or live stream

    Args:
        video_source: Path to video file
        camera_id: Camera ID for live stream (if video_source is None)

    Returns:
        frame: numpy array (H, W, 3)

    Raises:
        HTTPException: If frame cannot be retrieved
    """
    if video_source:
        # Read from video file
        cap = cv2.VideoCapture(video_source)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read frame from video: {video_source}"
            )

        return frame

    elif camera_id:
        # Try to get from live stream (requires streaming_server integration)
        try:
            from streaming_server import get_latest_frame
            frame = get_latest_frame(camera_id)

            if frame is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No live frame available for camera: {camera_id}"
                )

            return frame
        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Live streaming not configured. Please provide video_source."
            )

    else:
        raise HTTPException(
            status_code=400,
            detail="Either video_source or camera_id must be provided"
        )


@router.post("/auto/{camera_id}", response_model=CalibrationResponse)
async def auto_calibrate(camera_id: str, request: CalibrationRequest):
    """
    Auto-detect platform edge using YOLO + Hough fallback

    Returns normalized coordinates (0.0-1.0) for resolution independence.
    Tries YOLO first, falls back to Hough if YOLO fails.
    """
    logger.info(f"Auto calibration requested for camera: {camera_id}")

    try:
        # Get frame
        frame = get_frame_from_source(request.video_source, camera_id)
        h, w = frame.shape[:2]

        # Try YOLO first
        logger.info("Attempting YOLO-seg detection...")
        yolo_detector = YOLOSegEdgeDetector()

        if yolo_detector.available:
            platform_poly = yolo_detector.detect_platform_from_video(
                request.video_source or camera_id,
                num_frames=request.num_frames,
                visualize=request.visualize
            )

            if platform_poly and len(platform_poly) >= 2:
                logger.info(f"YOLO detection successful: {len(platform_poly)} points")

                return CalibrationResponse(
                    success=True,
                    points=normalize_points(platform_poly, w, h),
                    absolute_points=[[int(x), int(y)] for x, y in platform_poly],
                    video_dimensions={"width": w, "height": h},
                    detection_method="yolo-seg",
                    num_points=len(platform_poly)
                )

        # Fallback to Hough Transform
        logger.info("YOLO failed or unavailable, trying Hough Transform...")
        hough_detector = HoughEdgeDetector()
        platform_poly = hough_detector.detect_platform_from_video(
            request.video_source or camera_id,
            num_frames=request.num_frames,
            visualize=request.visualize
        )

        if platform_poly and len(platform_poly) >= 2:
            logger.info(f"Hough detection successful: {len(platform_poly)} points")

            return CalibrationResponse(
                success=True,
                points=normalize_points(platform_poly, w, h),
                absolute_points=[[int(x), int(y)] for x, y in platform_poly],
                video_dimensions={"width": w, "height": h},
                detection_method="hough-transform",
                num_points=len(platform_poly)
            )

        # Both methods failed
        raise HTTPException(
            status_code=404,
            detail="No platform edge detected using any method"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Calibration error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


@router.post("/yolo/{camera_id}", response_model=CalibrationResponse)
async def yolo_calibrate(camera_id: str, request: CalibrationRequest):
    """
    Detect platform edge using YOLO segmentation only
    """
    logger.info(f"YOLO calibration requested for camera: {camera_id}")

    try:
        # Get frame
        frame = get_frame_from_source(request.video_source, camera_id)
        h, w = frame.shape[:2]

        # Run YOLO detection
        yolo_detector = YOLOSegEdgeDetector()

        if not yolo_detector.available:
            raise HTTPException(
                status_code=501,
                detail="YOLO model not available"
            )

        platform_poly = yolo_detector.detect_platform_from_video(
            request.video_source or camera_id,
            num_frames=request.num_frames,
            visualize=request.visualize
        )

        if not platform_poly or len(platform_poly) < 2:
            raise HTTPException(
                status_code=404,
                detail="YOLO could not detect platform edge"
            )

        return CalibrationResponse(
            success=True,
            points=normalize_points(platform_poly, w, h),
            absolute_points=[[int(x), int(y)] for x, y in platform_poly],
            video_dimensions={"width": w, "height": h},
            detection_method="yolo-seg",
            num_points=len(platform_poly)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YOLO calibration error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"YOLO calibration failed: {str(e)}")


@router.post("/hough/{camera_id}", response_model=CalibrationResponse)
async def hough_calibrate(camera_id: str, request: CalibrationRequest):
    """
    Detect platform edge using Hough Transform line detection only
    """
    logger.info(f"Hough calibration requested for camera: {camera_id}")

    try:
        # Get frame
        frame = get_frame_from_source(request.video_source, camera_id)
        h, w = frame.shape[:2]

        # Run Hough detection
        hough_detector = HoughEdgeDetector()
        platform_poly = hough_detector.detect_platform_from_video(
            request.video_source or camera_id,
            num_frames=request.num_frames,
            visualize=request.visualize
        )

        if not platform_poly or len(platform_poly) < 2:
            raise HTTPException(
                status_code=404,
                detail="Hough Transform could not detect platform edge"
            )

        return CalibrationResponse(
            success=True,
            points=normalize_points(platform_poly, w, h),
            absolute_points=[[int(x), int(y)] for x, y in platform_poly],
            video_dimensions={"width": w, "height": h},
            detection_method="hough-transform",
            num_points=len(platform_poly)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hough calibration error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Hough calibration failed: {str(e)}")


@router.get("/status/{camera_id}")
async def calibration_status(camera_id: str):
    """
    Get calibration status for a camera

    Note: This endpoint just reports if calibration endpoints are working.
    Actual calibration data is stored in Node.js MongoDB.
    """
    return {
        "camera_id": camera_id,
        "endpoints_available": True,
        "methods": ["auto", "yolo", "hough"],
        "note": "Calibration data is stored in Node.js backend (MongoDB)"
    }
