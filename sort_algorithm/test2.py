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
    max_age=30,     # frame харагдахгүй бол ID хадгалах
    min_hits=3,     # баталгаажих frame
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
    "output_sort_count_dark.mp4", # Файлын нэрийг өөрчиллөө
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
    # Draw + Count (Өөрчилсөн хэсэг)
    # -----------------------------
    for track in tracks:
        # track-аас coordinates болон ID-г авах
        x1, y1, x2, y2, track_id = map(int, track)
        
        # Coordinates-ийг зөвхөн frame дотор байлгах (шаардлагатай бол)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # ----------------------------------------
        # АЛХАМ 1: 75% Тод / 25% Бүдэг эффект нэмэх
        # ----------------------------------------
        roi = frame[y1:y2, x1:x2]
        
        # ROI хэсэг хоосон биш эсэхийг шалгах
        if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
            # Хар өнгийн маск үүсгэх
            # (Энэ нь ROI-ийн хэмжээтэй адил тэг хар зураг)
            black_mask = np.zeros_like(roi)
            
            # Blend хийх: (Original ROI weight = 0.75, Black Mask weight = 0.25)
            dark_roi = cv2.addWeighted(roi, 0.75, black_mask, 0.25, 0)
            
            # Үндсэн frame дээр бүдэгрүүлсэн ROI-г буцаан байрлуулах
            frame[y1:y2, x1:x2] = dark_roi

        # ----------------------------------------
        # АЛХАМ 2: Хар хүрээ (Outline) зурах
        # ----------------------------------------
        # 4 пикселийн зузаантай хар хүрээ зурах
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # ----------------------------------------
        # АЛХАМ 3: Тод шошго (Label) нэмэх
        # ----------------------------------------
        label = f"ID {track_id}"
        
        # Текстийн хэмжээг тооцоолох
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Шошгоны ард хар дэвсгэр зурах (уншигдахуйц байдлыг сайжруулах)
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10), # Зүүн дээд
            (x1 + text_width + 5, y1),   # Баруун доод
            (0, 0, 0),                   # Хар өнгө
            cv2.FILLED
        )

        # Шошгыг цагаан өнгөөр бичих
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5), # Байрлал
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255), # Цагаан өнгө
            2
        )

        # ----------------------------------------
        # АЛХАМ 4: Объектыг тоолох
        # ----------------------------------------
        # ID-г өмнө тоолсон эсэхийг шалгах, зөвхөн нэг удаа тоолох
        if track_id not in counted_ids:
            # Та энд тоолох шугамыг (counting line) ашиглахгүй байгаа тул
            # объект анх удаа илрэх үед л тоолно.
            counted_ids.add(track_id)
            total_count += 1


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

print("✅ DONE → output_sort_count_dark.mp4")
print("TOTAL COUNT =", total_count)
