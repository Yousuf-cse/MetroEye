"""
Calibration API for MetroEye Platform Edge Detection
===================================================

Provides REST API endpoints for:
- Getting calibration status
- Running calibration with user choice
- Reviewing and approving/rejecting calibrations
- Manual recalibration interface

Usage:
    python calibration_api.py

    Then access:
    - GET  /api/calibration/{camera_id} - Get calibration status
    - POST /api/calibration/{camera_id}/auto - Auto calibrate
    - POST /api/calibration/{camera_id}/manual - Manual calibrate
    - GET  /api/calibration/{camera_id}/preview - Get preview image
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import json
import base64
from pathlib import Path
from edge_calibrator import CalibrationManager
import io

app = Flask(__name__)
CORS(app)

calibration_manager = CalibrationManager()


@app.route('/api/calibration/<camera_id>', methods=['GET'])
def get_calibration_status(camera_id):
    """
    Get current calibration status for a camera

    Returns:
        {
            "camera_id": "platform_cam_1",
            "calibrated": true/false,
            "calibration_data": {...} or null,
            "detection_method": "yolo-seg" / "hough" / "manual" / null
        }
    """
    calibration = calibration_manager.load_calibration(camera_id)

    if calibration is not None:
        config_path = calibration_manager.get_config_path(camera_id)
        with open(config_path, 'r') as f:
            full_config = json.load(f)

        return jsonify({
            "camera_id": camera_id,
            "calibrated": True,
            "calibration_data": full_config,
            "detection_method": full_config.get('detection_method'),
            "vertices": len(calibration)
        })
    else:
        return jsonify({
            "camera_id": camera_id,
            "calibrated": False,
            "calibration_data": None,
            "detection_method": None
        })


@app.route('/api/calibration/<camera_id>/preview', methods=['GET'])
def get_calibration_preview(camera_id):
    """
    Get a preview image showing the calibration overlay

    Query params:
        video_path: Path to video file

    Returns:
        JPEG image with calibration overlay
    """
    video_path = request.args.get('video_path')
    if not video_path:
        return jsonify({"error": "video_path parameter required"}), 400

    calibration = calibration_manager.load_calibration(camera_id)
    if calibration is None:
        return jsonify({"error": "No calibration found"}), 404

    # Read first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Could not read video"}), 400

    # Draw calibration overlay
    cv2.polylines(frame, [np.array(calibration)], True, (0, 255, 0), 3)
    for pt in calibration:
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame)

    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name=f'{camera_id}_preview.jpg'
    )


@app.route('/api/calibration/<camera_id>/methods', methods=['GET'])
def get_available_methods(camera_id):
    """
    Get available calibration methods and their descriptions

    Returns:
        {
            "methods": [
                {
                    "id": "auto",
                    "name": "Automatic Detection",
                    "description": "Tries YOLO-seg, then Hough Transform",
                    "recommended": true
                },
                {
                    "id": "yolo",
                    "name": "YOLO Segmentation Only",
                    "description": "AI-based semantic segmentation"
                },
                ...
            ]
        }
    """
    return jsonify({
        "methods": [
            {
                "id": "auto",
                "name": "Automatic Detection (Recommended)",
                "description": "Tries YOLO-seg first, falls back to Hough Transform if needed, then manual",
                "recommended": True
            },
            {
                "id": "yolo",
                "name": "YOLO Segmentation",
                "description": "AI-based detection - best for complex scenes with people"
            },
            {
                "id": "hough",
                "name": "Hough Transform",
                "description": "Line detection - best for clear straight platform edges"
            },
            {
                "id": "manual",
                "name": "Manual Calibration",
                "description": "Click points manually - always works but requires user input"
            }
        ]
    })


@app.route('/api/calibration/<camera_id>/calibrate', methods=['POST'])
def run_calibration(camera_id):
    """
    Run calibration with specified method

    Request body:
        {
            "video_path": "/path/to/video.mp4",
            "method": "auto" / "yolo" / "hough" / "manual",
            "force": true/false (recalibrate even if exists)
        }

    Returns:
        {
            "success": true/false,
            "camera_id": "platform_cam_1",
            "detection_method": "yolo-seg",
            "vertices": 4,
            "calibration_data": {...},
            "preview_url": "/api/calibration/{camera_id}/preview?video_path=..."
        }
    """
    data = request.json

    if not data or 'video_path' not in data:
        return jsonify({"error": "video_path required"}), 400

    video_path = data['video_path']
    method = data.get('method', 'auto')
    force = data.get('force', False)

    # Run calibration
    try:
        platform_poly = calibration_manager.calibrate(
            video_source=video_path,
            camera_id=camera_id,
            method=method,
            visualize=False,
            force_recalibrate=force
        )

        if platform_poly:
            # Load the saved config
            config_path = calibration_manager.get_config_path(camera_id)
            with open(config_path, 'r') as f:
                full_config = json.load(f)

            return jsonify({
                "success": True,
                "camera_id": camera_id,
                "detection_method": full_config['detection_method'],
                "vertices": len(platform_poly),
                "calibration_data": full_config,
                "preview_url": f"/api/calibration/{camera_id}/preview?video_path={video_path}"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Calibration failed with all methods"
            }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/calibration/<camera_id>/manual', methods=['POST'])
def manual_calibration(camera_id):
    """
    Save manual calibration from user-clicked points

    Request body:
        {
            "video_path": "/path/to/video.mp4",
            "points": [[x1, y1], [x2, y2], ...],
            "video_dimensions": {"width": 1920, "height": 1080}
        }

    Note: Points can be from ANY display size - they'll be normalized automatically

    Returns:
        {
            "success": true,
            "camera_id": "platform_cam_1",
            "normalized_points": [[0.0, 0.15], [0.95, 0.35]],
            "num_points": 2
        }
    """
    data = request.json

    if not data or 'points' not in data or 'video_dimensions' not in data:
        return jsonify({"error": "points and video_dimensions required"}), 400

    points = data['points']
    video_dims = (data['video_dimensions']['width'], data['video_dimensions']['height'])
    video_path = data.get('video_path', '')

    if len(points) < 2:
        return jsonify({"error": "At least 2 points required"}), 400

    # Convert points to tuples
    platform_poly = [(int(x), int(y)) for x, y in points]

    try:
        # Save with normalization
        calibration_manager.save_calibration(
            camera_id=camera_id,
            platform_poly=platform_poly,
            video_source=video_path,
            method='manual',
            video_dims=video_dims
        )

        # Load back to get normalized coords
        config_path = calibration_manager.get_config_path(camera_id)
        with open(config_path, 'r') as f:
            saved_config = json.load(f)

        return jsonify({
            "success": True,
            "camera_id": camera_id,
            "normalized_points": saved_config['platform_edge']['normalized'],
            "num_points": len(points),
            "detection_method": "manual",
            "preview_url": f"/api/calibration/{camera_id}/preview?video_path={video_path}"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/calibration/<camera_id>/delete', methods=['DELETE'])
def delete_calibration(camera_id):
    """
    Delete calibration for a camera

    Returns:
        {
            "success": true,
            "message": "Calibration deleted"
        }
    """
    config_path = calibration_manager.get_config_path(camera_id)

    if config_path.exists():
        config_path.unlink()
        return jsonify({
            "success": True,
            "message": f"Calibration for {camera_id} deleted"
        })
    else:
        return jsonify({
            "success": False,
            "error": "No calibration found"
        }), 404


if __name__ == '__main__':
    print("=" * 60)
    print("MetroEye Calibration API Server")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  GET    /api/calibration/{camera_id}")
    print("  GET    /api/calibration/{camera_id}/preview")
    print("  GET    /api/calibration/{camera_id}/methods")
    print("  POST   /api/calibration/{camera_id}/calibrate")
    print("  DELETE /api/calibration/{camera_id}/delete")
    print("\n" + "=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
