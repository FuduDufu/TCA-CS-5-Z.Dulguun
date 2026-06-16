"""
analyze.py
==========
practice.py-гаар авсан бичлэг + pose CSV-г уншиж:
  1. Зүүн камерын frame тус бүрт мод таних  (YOLO + ногооны mask)
  2. Стерео depth тооцоолно (баруун хагасаар)
  3. Тухайн frame-ийн pose ашиглан world coordinate-д хөрвүүлнэ
  4. Map зурж practice_map_... .png болон .mp4 хэлбэрээр хадгална

Ашиглалт:
  python analyze.py practice_cam_20250420_120000.mp4 practice_pose_20250420_120000.csv
"""

import sys
import csv
import math
import time
import numpy as np
import cv2
from ultralytics import YOLO

# =============================================================
# ТОХИРГОО
# =============================================================
FOV_DEG            = 96
BASELINE_M         = 0.065
STEREO_SCALE       = 2
DISP_HISTORY_SIZE  = 3     # offline-д RAM хайрладаггүй тул бага байж болно
EMA_ALPHA          = 0.3
MAP_MAX_DISTANCE   = 15.0  # outdoor-д хол байж болно

LANDMARK_MERGE_THRESHOLD = 1.0   # outdoor-д одометрийн алдаа их → илүү том
LANDMARK_MIN_OBS         = 3
LANDMARK_POS_HISTORY     = 20

# Мод таних — YOLOv8-ийн анги + ногооны HSV mask хослол
YOLO_MODEL    = "../models/yolov8n.pt"   # эсвэл tree-specific model (жишээ: yolov8n_tree.pt)
YOLO_CONF     = 0.35
# YOLOv8n coco80: 58=potted plant (дотор).
# Гадна мод танихад тусгайлсан model шаардагдана.
# HSV ногооны mask мод байж болох газрыг санал болгоно.
TREE_YOLO_CLASSES = {58: "plant"}   # ← custom model ашиглавал өөрчил

# Ногооны HSV range (мод / навч)
GREEN_HSV_LOW  = np.array([35,  40,  40])
GREEN_HSV_HIGH = np.array([90, 255, 255])
GREEN_MIN_AREA = 800   # пиксел талбай — жижиг noise-г хаяна

# Map зургийн хэмжээ
MAP_W, MAP_H = 900, 700

# =============================================================
# LANDMARK
# =============================================================
class Landmark:
    def __init__(self, x, y, label):
        self.label = label
        self.xs    = [x]; self.ys = [y]
        self.x     = x;   self.y  = y
        self.count = 1;    self.confirmed = False

    def update(self, x, y):
        self.xs.append(x); self.ys.append(y)
        if len(self.xs) > LANDMARK_POS_HISTORY:
            self.xs.pop(0); self.ys.pop(0)
        self.x = float(np.median(self.xs))
        self.y = float(np.median(self.ys))
        self.count += 1
        if self.count >= LANDMARK_MIN_OBS:
            self.confirmed = True

class WorldMap:
    def __init__(self):
        self.landmarks = []

    def add(self, wx, wy, label, tid):
        for lm in self.landmarks:
            if lm.label != label: continue
            if math.hypot(lm.x - wx, lm.y - wy) < LANDMARK_MERGE_THRESHOLD:
                lm.update(wx, wy); return
        self.landmarks.append(Landmark(wx, wy, label))

    def confirmed(self):
        return [(lm.x, lm.y, lm.label, lm.count)
                for lm in self.landmarks if lm.confirmed]

    def all(self):
        return [(lm.x, lm.y, lm.label, lm.count, lm.confirmed)
                for lm in self.landmarks]


# =============================================================
# STEREO
# =============================================================
def create_stereo():
    ws = 5
    left = cv2.StereoSGBM.create(
        minDisparity=0, numDisparities=128, blockSize=ws,
        P1=8*3*ws**2, P2=32*3*ws**2, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    right = cv2.ximgproc.createRightMatcher(left)
    wls   = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left)
    wls.setLambda(8000); wls.setSigmaColor(1.5)
    return left, right, wls

def roi_depth(disp, focal, y1, y2, x1, x2):
    roi   = disp[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if len(valid) < 10: return None
    med = np.median(valid)
    if med <= 0: return None
    d = (focal * BASELINE_M) / med
    return d if 0.1 < d < MAP_MAX_DISTANCE else None

def pixel_to_world(cx_rel, depth, focal, rx, ry, rth):
    bearing = math.atan2(cx_rel, focal)
    rfx =  depth * math.cos(bearing)
    rfy = -depth * math.sin(bearing)
    wx  = rx + rfx * math.cos(rth) - rfy * math.sin(rth)
    wy  = ry + rfx * math.sin(rth) + rfy * math.cos(rth)
    return wx, wy


# =============================================================
# НОГООНЫ MASK-ААР МОД ТАНИХ
# =============================================================
def find_green_blobs(frame_bgr):
    """
    HSV ногооны mask дээрх contour-уудыг буцаана.
    Return: list of (cx, cy, x1, y1, x2, y2)
    """
    hsv    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask   = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        if cv2.contourArea(c) < GREEN_MIN_AREA: continue
        bx, by, bw, bh = cv2.boundingRect(c)
        blobs.append((bx + bw//2, by + bh//2, bx, by, bx+bw, by+bh))
    return blobs


# =============================================================
# MAP ЗУРАХ
# =============================================================
COLORS = [
    (0, 200, 100), (255, 100, 0), (0, 200, 255), (200, 0, 255),
    (255, 200, 0), (0, 128, 255), (255, 0, 128), (128, 255, 0),
]

def render_map(world_map_obj, trail, frame_idx, total_frames):
    img = np.full((MAP_H, MAP_W, 3), (24, 24, 30), dtype=np.uint8)

    all_lms = world_map_obj.all()
    conf    = [(x, y, l, n) for (x, y, l, n, c) in all_lms if c]

    all_pts = [(0.0, 0.0)] + [(x, y) for x, y, *_ in all_lms] + trail
    if len(all_pts) < 2:
        return img

    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
    cx = (max(xs) + min(xs)) / 2;  cy = (max(ys) + min(ys)) / 2
    span = max(max(xs)-min(xs), max(ys)-min(ys), 1.0) + 2.0
    margin = 60
    scale  = min((MAP_W - 2*margin) / span, (MAP_H - 60 - 2*margin) / span)

    def w2s(wx, wy):
        sx = int(MAP_W/2 + (wx - cx) * scale)
        sy = int((MAP_H + 60)/2 - (wy - cy) * scale)
        return sx, sy

    # Header
    cv2.rectangle(img, (0, 0), (MAP_W, 40), (55, 55, 75), -1)
    pct = int(100 * frame_idx / max(total_frames, 1))
    cv2.putText(img, f"analyze.py  —  {frame_idx}/{total_frames} ({pct}%)",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 230, 140), 2)

    # Grid
    for xi in range(int(cx-span/2)-1, int(cx+span/2)+2):
        p1 = w2s(xi, cy-span/2); p2 = w2s(xi, cy+span/2)
        cv2.line(img, p1, p2, (50, 50, 65), 1)
    for yi in range(int(cy-span/2)-1, int(cy+span/2)+2):
        p1 = w2s(cx-span/2, yi); p2 = w2s(cx+span/2, yi)
        cv2.line(img, p1, p2, (50, 50, 65), 1)

    # Trail
    if len(trail) >= 2:
        pts = [w2s(px, py) for px, py in trail]
        for k in range(1, len(pts)):
            cv2.line(img, pts[k-1], pts[k], (90, 200, 255), 1, cv2.LINE_AA)

    # Start
    s0 = w2s(0, 0)
    cv2.circle(img, s0, 9, (40, 200, 255), -1)
    cv2.putText(img, "START", (s0[0]+12, s0[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 200, 255), 1)

    # Pending landmarks (бүдэг)
    for (lx, ly, _, _, cf) in all_lms:
        if cf: continue
        cv2.circle(img, w2s(lx, ly), 4, (100, 120, 100), 1)

    # Confirmed landmarks
    for idx, (lx, ly, label, cnt) in enumerate(conf):
        sc = w2s(lx, ly)
        col = COLORS[idx % len(COLORS)]
        cv2.circle(img, sc, 9, col, -1)
        cv2.circle(img, sc, 11, (255, 255, 255), 1)
        cv2.putText(img, f"#{idx+1} {label}", (sc[0]+13, sc[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    cv2.putText(img, f"Trees: {len(conf)} confirmed  ({len(all_lms)} total)",
                (10, MAP_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 200, 160), 1)

    return img


# =============================================================
# ҮНД ЭЭС
# =============================================================
def run(video_path, pose_path):
    # --- CSV унших ---
    poses = {}   # frame_idx → (time_s, x, y, theta)
    with open(pose_path, newline="") as f:
        for row in csv.DictReader(f):
            poses[int(row["frame"])] = (
                float(row["time_s"]),
                float(row["x_m"]),
                float(row["y_m"]),
                float(row["theta_rad"]),
            )
    print(f"[CSV] {len(poses)} frame-ийн pose унших боллоо.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video нээгдсэнгүй: {video_path}"); return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 10.0
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_w = fw // 2
    focal  = (half_w / 2) / math.tan(math.radians(FOV_DEG / 2))
    print(f"[VID] {fw}x{fh}  {fps:.1f}fps  {total} frame  focal={focal:.1f}px")

    # Output writers
    import datetime
    ts_str     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_map_path = f"../output/practice_map_{ts_str}.mp4"
    out_img_path = f"../output/practice_map_{ts_str}.png"
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    map_writer = cv2.VideoWriter(out_map_path, fourcc, fps, (MAP_W, MAP_H))
    print(f"[OUT] map video → {out_map_path}")

    yolo       = YOLO(YOLO_MODEL)
    s_left, s_right, wls = create_stereo()
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    disp_hist  = []
    dist_ema   = {}
    world_map  = WorldMap()
    trail      = []
    frame_idx  = 0

    cv2.namedWindow("Analyze - Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Analyze - Camera", fw, fh)
    cv2.namedWindow("Analyze - Map",    cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Analyze - Map",   MAP_W, MAP_H)
    cv2.moveWindow("Analyze - Camera", 30, 30)
    cv2.moveWindow("Analyze - Map",    30, fh + 60)

    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, -1)
        left  = frame[:, half_w:]
        right = frame[:, :half_w]

        # Тухайн frame-ийн pose
        pose_row = poses.get(frame_idx)
        if pose_row is None:
            frame_idx += 1; continue
        _, rx, ry, rth = pose_row
        trail.append((rx, ry))

        # --- Stereo disparity ---
        gl = clahe.apply(cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY))
        gr = clahe.apply(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
        gls = cv2.resize(gl, None, fx=1/STEREO_SCALE, fy=1/STEREO_SCALE)
        grs = cv2.resize(gr, None, fx=1/STEREO_SCALE, fy=1/STEREO_SCALE)
        dl  = s_left.compute(gls, grs)
        dr  = s_right.compute(grs, gls)
        ds  = wls.filter(dl, gls, None, dr).astype(np.float32) / 16.0
        disp_raw = cv2.resize(ds, (half_w, fh)) * STEREO_SCALE
        disp_hist.append(disp_raw)
        if len(disp_hist) > DISP_HISTORY_SIZE: disp_hist.pop(0)
        disparity = np.mean(disp_hist, axis=0)

        lh, lw = left.shape[:2]
        draw = left.copy()
        tid_counter = 0   # green blob-д pseudo track_id

        # --- YOLO объект таних ---
        results = yolo.track(left, verbose=False, conf=YOLO_CONF,
                             persist=True, tracker="botsort.yaml", imgsz=480)
        for r in results:
            if r.boxes is None or r.boxes.id is None: continue
            for box in r.boxes:
                cid = int(box.cls[0])
                if cid not in TREE_YOLO_CLASSES or box.id is None: continue
                tid  = int(box.id[0])
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1,y1 = max(0,x1),max(0,y1); x2,y2 = min(lw,x2),min(lh,y2)
                raw_d = roi_depth(disparity, focal, y1, y2, x1, x2)
                if raw_d is None: continue
                prev = dist_ema.get(tid)
                sm   = raw_d if prev is None else prev*(1-EMA_ALPHA)+raw_d*EMA_ALPHA
                dist_ema[tid] = sm
                cx_rel = (x1+x2)//2 - lw//2
                wx, wy = pixel_to_world(cx_rel, sm, focal, rx, ry, rth)
                world_map.add(wx, wy, TREE_YOLO_CLASSES[cid], tid)
                cv2.rectangle(draw, (x1,y1),(x2,y2), (0,200,80), 2)
                cv2.putText(draw, f"plant {sm:.1f}m", (x1,y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,80), 2)

        # --- Ногооны mask-ээр мод хайх ---
        blobs = find_green_blobs(left)
        for (bcx, bcy, bx1, by1, bx2, by2) in blobs:
            raw_d = roi_depth(disparity, focal, by1, by2, bx1, bx2)
            if raw_d is None: continue
            key  = f"green_{tid_counter}"; tid_counter += 1
            prev = dist_ema.get(key)
            sm   = raw_d if prev is None else prev*(1-EMA_ALPHA)+raw_d*EMA_ALPHA
            dist_ema[key] = sm
            cx_rel = bcx - lw//2
            wx, wy = pixel_to_world(cx_rel, sm, focal, rx, ry, rth)
            world_map.add(wx, wy, "tree", key)
            cv2.rectangle(draw, (bx1,by1),(bx2,by2), (0,255,120), 1)
            cv2.putText(draw, f"tree? {sm:.1f}m", (bx1, by1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,120), 1)

        pct = int(100 * frame_idx / max(total, 1))
        cv2.putText(draw, f"F:{frame_idx}/{total} ({pct}%)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        combined = np.hstack([draw, right])
        cv2.imshow("Analyze - Camera", combined)

        map_img = render_map(world_map, trail, frame_idx, total)
        map_writer.write(map_img)
        cv2.imshow("Analyze - Map", map_img)

        frame_idx += 1
        if cv2.waitKey(1) == 27: break

        elapsed = time.time() - t0
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total} ({pct}%)  {elapsed:.0f}s  "
                  f"trees: {len(world_map.confirmed())} confirmed")

    cap.release()
    map_writer.release()
    cv2.destroyAllWindows()

    # Эцсийн map зурагт хадгална
    final_map = render_map(world_map, trail, total, total)
    cv2.imwrite(out_img_path, final_map)

    print("\n===== ДҮНГИЙН MAP =====")
    for i, (lx, ly, label, cnt) in enumerate(world_map.confirmed()):
        print(f"  #{i+1} {label}: ({lx:.2f},{ly:.2f})  x{cnt}")
    print(f"\nНийт {len(world_map.confirmed())} confirmed landmark")
    print(f"Map зураг → {out_img_path}")
    print(f"Map видео  → {out_map_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Хэрэглэлт: python analyze.py <video.mp4> <pose.csv>")
        print("Жишээ:     python analyze.py practice_cam_20250420.mp4 practice_pose_20250420.csv")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
