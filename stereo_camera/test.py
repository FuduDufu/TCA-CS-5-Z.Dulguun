import cv2
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO

# ============ Камерын параметрүүд ============
FOV_DEG = 96       # Камерын харагдах өнцөг (градус)
BASELINE_M = 0.065 # Хоёр линзний хоорондох зай: 6.5 см
# =============================================

# YOLOv8 + BoT-SORT tracking
yolo = YOLO("yolov8n.pt")
BOTTLE_CLASSES = {39: "bottle", 41: "cup"}
YOLO_CONF = 0.4

camR = cv2.VideoCapture(1)
camR.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
camR.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Эхний frame-ээс focal length тооцоолох
ret_init, frame_init = camR.read()
if not ret_init:
    print("Камер нээгдсэнгүй!")
    exit()
half_w = frame_init.shape[1] // 2
FOCAL_LENGTH_PX = (half_w / 2) / math.tan(math.radians(FOV_DEG / 2))

# Stereo-г 2x жижиг зураг дээр тооцоолж хурдасгах
STEREO_SCALE = 2  # 2x downscale

# StereoSGBM (downscale хийсэн зурагт тааруулсан)
window_size = 5
stereo_left = cv2.StereoSGBM.create(
    minDisparity=0,
    numDisparities=128,  # 256/2 = 128 (downscale-д тааруулсан)
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# WLS filter
stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.5)

# Temporal smoothing
DISP_HISTORY_SIZE = 5
EMA_ALPHA = 0.3
disp_history = []
distance_by_id = {}  # Track ID бүрийн smoothed distance

# Өнгө: track ID бүрт тогтмол өнгө өгөх
TRACK_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 165, 255), (128, 0, 255), (255, 128, 0),
    (0, 255, 128), (128, 255, 0), (255, 0, 128), (0, 128, 255),
]

print("=" * 50)
print("BOTTLE TRACKING + DISTANCE")
print(f"  Resolution:   {half_w}x{frame_init.shape[0]}")
print(f"  Focal length: {FOCAL_LENGTH_PX:.1f} px")
print(f"  Baseline:     {BASELINE_M * 100:.1f} cm")
print("  ESC = гарах")
print("=" * 50)


def get_distance(disparity, y1, y2, x1, x2):
    roi = disparity[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if len(valid) < 10:
        return None
    med = np.median(valid)
    if med <= 0:
        return None
    dist = (FOCAL_LENGTH_PX * BASELINE_M) / med
    if 0.1 < dist < 20.0:
        return dist
    return None


while True:
    ret, frame = camR.read()
    if not ret:
        break

    frame = cv2.flip(frame, -1)  # 180° эргүүлэх (-1: хэвтээ+босоо, 0: босоо, 1: хэвтээ)
    h, w, _ = frame.shape
    left = frame[:, :w // 2]
    right = frame[:, w // 2:]

    # Grayscale + CLAHE + downscale (stereo хурдасгах)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_left_full = clahe.apply(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
    gray_right_full = clahe.apply(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
    gray_left = cv2.resize(gray_left_full, None, fx=1/STEREO_SCALE, fy=1/STEREO_SCALE)
    gray_right = cv2.resize(gray_right_full, None, fx=1/STEREO_SCALE, fy=1/STEREO_SCALE)

    # Stereo disparity (жижиг зураг дээр тооцоолоод буцааж scale хийнэ)
    disp_left = stereo_left.compute(gray_left, gray_right)
    disp_right = stereo_right.compute(gray_right, gray_left)
    disp_small = wls_filter.filter(disp_left, gray_left, None, disp_right)
    disp_small = disp_small.astype(np.float32) / 16.0
    # Disparity-г буцааж анхны хэмжээнд scale хийх (утгыг ч STEREO_SCALE-аар үржүүлнэ)
    disp_raw = cv2.resize(disp_small, (left.shape[1], left.shape[0])) * STEREO_SCALE

    disp_history.append(disp_raw)
    if len(disp_history) > DISP_HISTORY_SIZE:
        disp_history.pop(0)
    disparity = np.mean(disp_history, axis=0)

    left_draw = left.copy()
    lh, lw = left.shape[:2]

    # --- YOLOv8 Track (BoT-SORT) ---
    results = yolo.track(left, verbose=False, conf=YOLO_CONF,
                         persist=True, tracker="botsort.yaml", imgsz=480)

    active_ids = set()
    for r in results:
        if r.boxes is None or r.boxes.id is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in BOTTLE_CLASSES:
                continue
            if box.id is None:
                continue

            track_id = int(box.id[0])
            active_ids.add(track_id)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(lw, x2), min(lh, y2)
            cls_name = BOTTLE_CLASSES[cls_id]

            # Зай тооцоолох + EMA smoothing (track ID-аар)
            raw_dist = get_distance(disparity, y1, y2, x1, x2)
            dist_label = ""
            if raw_dist:
                if track_id in distance_by_id:
                    smoothed = distance_by_id[track_id] * (1 - EMA_ALPHA) + raw_dist * EMA_ALPHA
                else:
                    smoothed = raw_dist
                distance_by_id[track_id] = smoothed
                dist_label = f" {smoothed:.2f}m"

            # Track ID-д тогтмол өнгө
            color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
            label = f"ID:{track_id} {cls_name}{dist_label}"

            cv2.rectangle(left_draw, (x1, y1), (x2, y2), color, 2)
            # Label-ийн дэвсгэр
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(left_draw, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(left_draw, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Алга болсон ID-ийн distance хадгалсаар байна (буцаж ирвэл ашиглана)
    # Хэт олон хуримтлагдахаас сэргийлж хамгийн хуучныг устгана
    if len(distance_by_id) > 50:
        old_ids = [k for k in distance_by_id if k not in active_ids]
        for k in old_ids:
            del distance_by_id[k]

    # Дэлгэцэнд гаргах (Left + Right side by side)
    combined = np.hstack([left_draw, right])
    scale = min(1280 / combined.shape[1], 720 / combined.shape[0])
    display = cv2.resize(combined, None, fx=scale, fy=scale)

    cv2.imshow("Bottle Tracking", display)

    if cv2.waitKey(1) == 27:
        break

camR.release()
cv2.destroyAllWindows()
