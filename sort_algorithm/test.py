import cv2
import numpy as np
from roboflow import Roboflow
from sort import Sort
from dotenv import load_dotenv
import os
# -----------------------------
# 1. Roboflow model
# -----------------------------
load_dotenv()
api_k = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_k)
project = rf.workspace().project("mot4-fmhpl")
model = project.version("3").model

# -----------------------------
# 2. SORT tracker
# -----------------------------
tracker = Sort(
    max_age=30,    # frame харагдахгүй бол ID хадгалах
    min_hits=3,    # баталгаажих frame
    iou_threshold=0.3
)

# -----------------------------
# 3. Video input
# -----------------------------
cap = cv2.VideoCapture("first.mp4")
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "output_sort_count.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# -----------------------------
# 4. Counting logic
# -----------------------------
counted_ids = set()
total_count = 0

frame_id = 0

# -----------------------------
# 5. Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    print(f"Frame {frame_id}")

    # -----------------------------
    # Roboflow detection
    # -----------------------------
    preds = model.predict(
        frame,
        confidence=40,
        overlap=30
    ).json()

    detections = []

    for p in preds["predictions"]:
        x1 = int(p["x"] - p["width"] / 2)
        y1 = int(p["y"] - p["height"] / 2)
        x2 = int(p["x"] + p["width"] / 2)
        y2 = int(p["y"] + p["height"] / 2)

        score = p["confidence"]

        detections.append([x1, y1, x2, y2, score])

    # -----------------------------
    # SORT tracking
    # -----------------------------
    if len(detections) > 0:
        dets = np.array(detections)
    else:
        dets = np.empty((0, 5))

    tracks = tracker.update(dets)

    # -----------------------------
    # Draw + Count
    # -----------------------------
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)

        # Count once per ID
        if track_id not in counted_ids:
            counted_ids.add(track_id)
            total_count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -----------------------------
    # Display total count
    # -----------------------------
    cv2.putText(
        frame,
        f"TOTAL COUNT: {total_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    out.write(frame)

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()

print("✅ DONE → output_sort_count.mp4")
print("TOTAL COUNT =", total_count)
