import time
import csv
import os
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------

VIDEO_SOURCE = "sample-station.webm"  # or RTSP stream
CAMERA_ID = "platform_cam_1"
MODEL_PATH = "yolo26n-pose.pt"  # pose model
CONF_THRESH = 0.3
IOU_THRESH = 0.5
WINDOW_SECONDS = 4.0
OUTPUT_CSV = "features_dump.csv"

PLATFORM_POLY = [(50,400),(900,400),(900,720),(50,720)]

# ---------------- UTILITIES ----------------

def bbox_center(bbox):
    x1,y1,x2,y2 = bbox
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def point_to_polygon_distance(point, polygon):
    px, py = point
    min_d = float('inf')
    for i in range(len(polygon)):
        x1,y1 = polygon[i]
        x2,y2 = polygon[(i+1)%len(polygon)]
        dx = x2 - x1
        dy = y2 - y1
        if dx == dy == 0:
            d = np.hypot(px-x1, py-y1)
        else:
            t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy)/(dx*dx + dy*dy)))
            projx = x1 + t*dx
            projy = y1 + t*dy
            d = np.hypot(px-projx, py-projy)
        min_d = min(min_d, d)
    return min_d

def compute_torso_angle(kp):
    try:
        L_sh = kp[5][:2]
        R_sh = kp[6][:2]
        L_hip = kp[11][:2]
        R_hip = kp[12][:2]
        sh_mid = ((L_sh[0]+R_sh[0])/2, (L_sh[1]+R_sh[1])/2)
        hip_mid = ((L_hip[0]+R_hip[0])/2, (L_hip[1]+R_hip[1])/2)
        vec = np.array(hip_mid) - np.array(sh_mid)
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        return float(angle)
    except:
        return None

# ---------------- INITIALIZE ----------------

model = YOLO(MODEL_PATH)

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp","camera_id","track_id",
            "x1","y1","x2","y2",
            "torso_angle","speed_px_s",
            "dist_to_edge_px","dwell_time_s"
        ])

track_history = defaultdict(lambda: deque())
last_positions = {}
first_seen = {}

# ---------------- MAIN LOOP ----------------

cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ts = time.time()

    results = model.track(
        source=frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        persist=True
    )

    if len(results) == 0:
        continue

    r = results[0]

    if r.boxes is None:
        continue
    annotated_frame = results[0].plot()
    boxes = r.boxes.xyxy.cpu().numpy()
    ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None
    keypoints = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None

    for i, box in enumerate(boxes):
        if ids is None:
            continue

        track_id = int(ids[i])
        x1,y1,x2,y2 = map(int, box)

        center = bbox_center((x1,y1,x2,y2))

        # Update temporal history
        track_history[track_id].append((ts, center))
        while (ts - track_history[track_id][0][0]) > WINDOW_SECONDS:
            track_history[track_id].popleft()

        if track_id not in first_seen:
            first_seen[track_id] = ts

        # Compute speed
        speed = None
        if track_id in last_positions:
            last_ts, last_center = last_positions[track_id]
            dt = ts - last_ts if ts - last_ts > 0 else 1e-6
            speed = float(np.hypot(
                center[0]-last_center[0],
                center[1]-last_center[1]
            ) / dt)

        last_positions[track_id] = (ts, center)

        # Compute dwell time
        dwell = ts - first_seen[track_id]

        # Distance to platform edge
        dist_edge = point_to_polygon_distance(center, PLATFORM_POLY)

        # Torso angle
        torso_angle = None
        if keypoints is not None:
            torso_angle = compute_torso_angle(keypoints[i])

        # Write CSV
        with open(OUTPUT_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ts, CAMERA_ID, track_id,
                x1,y1,x2,y2,
                torso_angle, speed,
                dist_edge, dwell
            ])

        # Visualization
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(annotated_frame, f"ID:{track_id}", (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.polylines(annotated_frame, [np.array(PLATFORM_POLY)], True, (0,255,0), 2)
    cv2.imshow("debug", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
