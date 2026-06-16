import serial
import time
import math
import threading
import datetime
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import keyboard
from ultralytics import YOLO

# =============================================================
# ТОХИРГОО
# =============================================================
PORT         = "COM4"
BAUD         = 115200
CMD_SET      = 0x53 | 0x80

CAMERA_INDEX = 1
FOV_DEG      = 96
BASELINE_M   = 0.065

RECORD_VIDEO = True

# Камерын бодит FPS-г эхлүүлэхийн өмнө хэмжинэ — энэ тоог VideoWriter-т ашиглана.
# (VideoWriter-ийн FPS ≈ бодит FPS → бичлэг real-time тоглоно)
FPS_BENCHMARK_FRAMES = 30   # хэдэн frame-ийг хэмжихэд ашиглах

# Map-ыг камерын frame бүрт биш, N frame тутамд шинэчилнэ.
# YOLO+stereo дуусмагц map шинэчлэгдэнэ гэдэг утгаар delay-г бууруулна.
MAP_RENDER_EVERY = 4

# =============================================================
# SERVO PIN + ЗОГСОЛТ
# =============================================================
PIN = {
    "R31": 0,  "R32": 1,  "R33": 2,
    "L31": 3,  "L32": 4,  "L33": 5,
    "R21": 6,  "R22": 7,  "R23": 8,
    "L21": 9,  "L22": 10, "L23": 11,
    "R11": 12, "R12": 13, "R13": 14,
    "L11": 15, "L12": 16, "L13": 17,
}

STAND = {
    "L11": 1525, "L12": 1500, "L13": 1470,
    "L21": 1525, "L22": 1530, "L23": 1430,
    "L31": 1540, "L32": 1480, "L33": 1525,
    "R11": 1500, "R12": 1760, "R13": 1470,
    "R21": 1525, "R22": 1480, "R23": 1550,
    "R31": 1450, "R32": 1500, "R33": 1500,
}

GROUP_A = ["L1", "R2", "L3"]
GROUP_B = ["R1", "L2", "R3"]

# =============================================================
# GAIT ПАРАМЕТРҮҮД
# =============================================================
COXA_SWING           = 150
FEMUR_LIFT           = 200
TIBIA_LIFT           = 150
GROUND_PRESS         = 80
TRANSITION_STEPS     = 15
TRANSITION_DELAY     = 0.03
STEP_FORWARD_M       = 0.04     # нэг мөчлөгт урагш хэр шилжих (m)
TURN_ANGLE_RAD       = math.radians(30)  # нэг мөчлөгт хэр эргэх (rad)

# =============================================================
# YOLO + STEREO
# =============================================================
YOLO_MODEL       = "../models/yolov8n.pt"
YOLO_CONF        = 0.4
BOTTLE_CLASSES   = {39: "bottle", 41: "cup"}
STEREO_SCALE     = 2
DISP_HISTORY_SIZE = 5
EMA_ALPHA        = 0.3
MAP_MAX_DISTANCE = 8.0
# Odometry drift-г харгалзан томсгосон — нэг л bottle-ийг давхар оруулахаас сэргийлнэ
LANDMARK_MERGE_THRESHOLD = 0.80
# Хэдэн удаа тогтвортой харагдсаны дараа map-д confirmed гэж тэмдэглэх
LANDMARK_MIN_OBS = 3
# Detection цэгийг median-аар тогтворжуулахад ашиглах ажиглалтын тоо
LANDMARK_POS_HISTORY = 15
# Робот хөдөлж байх үед stereo frame desync + motion blur → шууд map-д нэмэхгүй,
# харин буферт хадгалаад, зогссоны дараа бүгдийг нь тогтвортой pose-оор commit хийнэ.
DEFERRED_COMMIT_WHILE_MOVING = True
# Хөдөлгөөн зогссоны дараа дараалсан хэдэн frame-г бас алгасах (stereo disp_hist цэвэрлэгдэх хугацаа)
MOTION_SETTLE_FRAMES = 4
MOTION_BUFFER_MAX    = 20
PATH_TRAIL_MAX   = 800

TRACK_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 165, 255), (128, 0, 255), (255, 128, 0),
    (0, 255, 128), (128, 255, 0), (255, 0, 128), (0, 128, 255),
]

# Map цонхны хэмжээ
MAP_W     = 1100
MAP_H     = 800
PANEL_W   = 340

# =============================================================
# MAP UI ӨНГӨНҮҮД
# =============================================================
COL_BG         = (24, 24, 30)
COL_PANEL_BG   = (40, 40, 50)
COL_HEADER_BG  = (55, 55, 75)
COL_DIVIDER    = (90, 90, 110)
COL_TEXT       = (220, 220, 230)
COL_TEXT_DIM   = (160, 160, 175)
COL_TEXT_HEAD  = (250, 230, 140)
COL_GRID_LINE  = (50, 50, 65)
COL_AXIS       = (80, 80, 100)
COL_TRAIL      = (90, 200, 255)
COL_ROBOT      = (60, 255, 80)
COL_START      = (40, 200, 255)

# =============================================================
# ХУВААЛЦСАН БАЙДАЛ
# =============================================================
stop_event = threading.Event()

robot_pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
pose_lock  = threading.Lock()

path_trail = []
path_lock  = threading.Lock()

session_start = None

# Робот одоо хөдөлж байна уу? keyboard_thread нь алхалтын өмнө True,
# зогсоод settle хийгдсэн үед False болгоно. camera_loop эндээс уншиж
# detection-ыг map-д бүртгэх эсэхийг шийднэ.
robot_moving   = False
motion_lock    = threading.Lock()

def set_moving(state):
    global robot_moving
    with motion_lock:
        robot_moving = state

def is_moving():
    with motion_lock:
        return robot_moving


def get_pose():
    with pose_lock:
        return robot_pose["x"], robot_pose["y"], robot_pose["theta"]

def advance_pose(dx=0.0, dy=0.0, dtheta=0.0):
    with pose_lock:
        th = robot_pose["theta"]
        robot_pose["x"]     += dx * math.cos(th) - dy * math.sin(th)
        robot_pose["y"]     += dx * math.sin(th) + dy * math.cos(th)
        robot_pose["theta"] += dtheta
        nx, ny = robot_pose["x"], robot_pose["y"]
    with path_lock:
        path_trail.append((nx, ny))
        if len(path_trail) > PATH_TRAIL_MAX:
            del path_trail[:len(path_trail) - PATH_TRAIL_MAX]


# =============================================================
# LANDMARK MAP
# =============================================================
class Landmark:
    def __init__(self, x, y, cls, tid):
        self.cls        = cls
        self.track_ids  = {tid}
        self.xs         = [x]   # сүүлийн ажиглалтуудын байрлал (median-д)
        self.ys         = [y]
        self.count      = 1
        self.x          = x     # одоогийн тогтворжсон байрлал (median)
        self.y          = y
        self.confirmed  = False # LANDMARK_MIN_OBS хүрсний дараа True

    def update(self, x, y, tid):
        self.xs.append(x); self.ys.append(y)
        if len(self.xs) > LANDMARK_POS_HISTORY:
            self.xs.pop(0); self.ys.pop(0)
        # Median ашиглавал нэг гэнэтийн аутлаер байрлалыг төвлөрлөөс холдуулахгүй
        self.x = float(np.median(self.xs))
        self.y = float(np.median(self.ys))
        self.count += 1
        self.track_ids.add(tid)
        if self.count >= LANDMARK_MIN_OBS:
            self.confirmed = True

class WorldMap:
    def __init__(self):
        self.landmarks = []; self.lock = threading.Lock()

    def add(self, wx, wy, cls, tid):
        with self.lock:
            # 1) Track-ID таарч байгаа landmark байвал хамгийн найдвартай → шинэчилнэ
            for lm in self.landmarks:
                if lm.cls == cls and tid in lm.track_ids:
                    lm.update(wx, wy, tid); return
            # 2) Track алдагдсан ч ойролцоо байгаа бол ижил landmark гэж үзнэ
            best = None; best_d = LANDMARK_MERGE_THRESHOLD
            for lm in self.landmarks:
                if lm.cls != cls: continue
                d = math.hypot(lm.x - wx, lm.y - wy)
                if d < best_d: best_d = d; best = lm
            if best: best.update(wx, wy, tid)
            else:    self.landmarks.append(Landmark(wx, wy, cls, tid))

    def snapshot(self):
        """Зөвхөн confirmed landmark-уудыг буцаана (map-д үзэгдэнэ)."""
        with self.lock:
            return [(lm.x, lm.y, lm.cls, lm.count)
                    for lm in self.landmarks if lm.confirmed]

    def snapshot_all(self):
        """Confirmed + pending landmark (pending-г бүдэг харуулахад хэрэгтэй)."""
        with self.lock:
            return [(lm.x, lm.y, lm.cls, lm.count, lm.confirmed)
                    for lm in self.landmarks]

world_map = WorldMap()


# =============================================================
# SERIAL
# =============================================================
def encode_set(start, values):
    data = bytearray([CMD_SET, start & 0x7F, len(values) & 0x7F])
    for v in values:
        data.append(v & 0x7F); data.append((v >> 7) & 0x7F)
    return data

def send_set(ser, start, values, label=""):
    pkt = encode_set(start, values)
    if label: print(label)
    ser.write(pkt)

def ease_in_out(t):
    return (1 - math.cos(math.pi * t)) / 2

def build_values(pose):
    vals = [1500] * 18
    for name, pulse in pose.items():
        vals[PIN[name]] = pulse
    return vals

def send_pose(ser, pose):
    send_set(ser, 0, build_values(pose))

def interpolate(pa, pb, alpha):
    return {k: int(round(pa[k] + (pb[k] - pa[k]) * alpha)) for k in pa}

def move_smooth(ser, pf, pt, steps=15, delay=0.03):
    for i in range(1, steps + 1):
        if stop_event.is_set(): return pf
        send_pose(ser, interpolate(pf, pt, ease_in_out(i / steps)))
        time.sleep(delay)
    return pt


# =============================================================
# OFFSET + POSE
# =============================================================
def coxa_offset(leg, off):
    s = leg + "1"
    return STAND[s] + off if leg.startswith("L") else STAND[s] - off

def femur_offset(leg, off):
    s = leg + "2"
    return STAND[s] - off if leg.startswith("L") else STAND[s] + off

def tibia_offset(leg, off):
    s = leg + "3"
    return STAND[s] + off if leg.startswith("L") else STAND[s] - off

def stand_pose():
    return dict(STAND)

def press_down(pose, leg):
    pose[leg + "2"] = femur_offset(leg, -GROUND_PRESS)
    pose[leg + "3"] = tibia_offset(leg, -GROUND_PRESS)

def turn_sign(leg, td):
    return td if leg.startswith("L") else -td

# ---- Walk poses ----
def step_pose_walk(swing, push, d):
    pose = stand_pose()
    for leg in swing:
        pose[leg+"2"] = femur_offset(leg, FEMUR_LIFT)
        pose[leg+"3"] = tibia_offset(leg, TIBIA_LIFT)
        pose[leg+"1"] = coxa_offset(leg, d * COXA_SWING)
    for leg in push:
        pose[leg+"1"] = coxa_offset(leg, d * (-COXA_SWING))
        press_down(pose, leg)
    return pose

def down_pose_walk(down, push, d):
    pose = stand_pose()
    for leg in down:
        pose[leg+"1"] = coxa_offset(leg, d * COXA_SWING)
        press_down(pose, leg)
    for leg in push:
        pose[leg+"1"] = coxa_offset(leg, d * (-COXA_SWING))
        press_down(pose, leg)
    return pose

# ---- Turn poses ----
def step_pose_turn(swing, push, td):
    pose = stand_pose()
    for leg in swing:
        pose[leg+"2"] = femur_offset(leg, FEMUR_LIFT)
        pose[leg+"3"] = tibia_offset(leg, TIBIA_LIFT)
        pose[leg+"1"] = coxa_offset(leg, turn_sign(leg, td) * COXA_SWING)
    for leg in push:
        pose[leg+"1"] = coxa_offset(leg, turn_sign(leg, td) * (-COXA_SWING))
        press_down(pose, leg)
    return pose

def down_pose_turn(down, push, td):
    pose = stand_pose()
    for leg in down:
        pose[leg+"1"] = coxa_offset(leg, turn_sign(leg, td) * COXA_SWING)
        press_down(pose, leg)
    for leg in push:
        pose[leg+"1"] = coxa_offset(leg, turn_sign(leg, td) * (-COXA_SWING))
        press_down(pose, leg)
    return pose


# =============================================================
# GAIT МӨЧЛӨГҮҮД
# =============================================================
def do_cycle(ser, cur, make_step, make_down, param):
    cur = move_smooth(ser, cur, make_step(GROUP_A, GROUP_B, param), TRANSITION_STEPS, TRANSITION_DELAY)
    cur = move_smooth(ser, cur, make_down(GROUP_A, GROUP_B, param), TRANSITION_STEPS // 2, TRANSITION_DELAY)
    cur = move_smooth(ser, cur, make_step(GROUP_B, GROUP_A, param), TRANSITION_STEPS, TRANSITION_DELAY)
    cur = move_smooth(ser, cur, make_down(GROUP_B, GROUP_A, param), TRANSITION_STEPS // 2, TRANSITION_DELAY)
    return cur

def return_to_stand(ser, cur):
    stand = stand_pose()
    def lift_place(c, group):
        p = dict(c)
        for leg in group:
            p[leg+"2"] = femur_offset(leg, FEMUR_LIFT)
            p[leg+"3"] = tibia_offset(leg, TIBIA_LIFT)
            p[leg+"1"] = stand[leg+"1"]
        c = move_smooth(ser, c, p, TRANSITION_STEPS, TRANSITION_DELAY)
        p2 = dict(c)
        for leg in group:
            p2[leg+"2"] = stand[leg+"2"]
            p2[leg+"3"] = stand[leg+"3"]
        return move_smooth(ser, c, p2, TRANSITION_STEPS // 2, TRANSITION_DELAY)
    cur = lift_place(cur, GROUP_A)
    cur = lift_place(cur, GROUP_B)
    return cur

def walk_fwd(ser, cur):
    cur = do_cycle(ser, cur, step_pose_walk, down_pose_walk, +1)
    advance_pose(dx=STEP_FORWARD_M)
    return cur

def walk_bwd(ser, cur):
    cur = do_cycle(ser, cur, step_pose_walk, down_pose_walk, -1)
    advance_pose(dx=-STEP_FORWARD_M)
    return cur

def turn_right(ser, cur):
    cur = do_cycle(ser, cur, step_pose_turn, down_pose_turn, +1)
    advance_pose(dtheta=-TURN_ANGLE_RAD)
    return cur

def turn_left(ser, cur):
    cur = do_cycle(ser, cur, step_pose_turn, down_pose_turn, -1)
    advance_pose(dtheta=+TURN_ANGLE_RAD)
    return cur


# =============================================================
# KEYBOARD CONTROLLER THREAD
# =============================================================
def keyboard_thread(ser):
    global session_start
    cur = stand_pose()
    send_pose(ser, cur)
    time.sleep(1.0)

    send_set(ser, 26, [1], "RELAY ON")
    time.sleep(0.5)
    session_start = time.time()

    print("=" * 48)
    print("HEXAPOD CONTROLLER  (камер + map идэвхтэй)")
    print("  W = урагш   S = хойш")
    print("  A = зүүн    D = баруун")
    print("  Q / ESC = гарах")
    print("=" * 48)

    last = None
    try:
        while not stop_event.is_set():
            if keyboard.is_pressed("q") or keyboard.is_pressed("esc"):
                stop_event.set(); break
            w = keyboard.is_pressed("w")
            s = keyboard.is_pressed("s")
            a = keyboard.is_pressed("a")
            d = keyboard.is_pressed("d")
            moving_now = w or s or a or d
            # Хөдөлгөөний үеийн stereo детекцийг алгасахын тулд флагийг
            # gait мөчлөгийн өмнө тавина, зогсоод settle хийнэ.
            if moving_now: set_moving(True)
            if w:
                if last != "W": print("→ УРАГШ");  last = "W"
                cur = walk_fwd(ser, cur)
            elif s:
                if last != "S": print("→ ХОЙШ");  last = "S"
                cur = walk_bwd(ser, cur)
            elif a:
                if last != "A": print("→ ЗҮҮН");  last = "A"
                cur = turn_left(ser, cur)
            elif d:
                if last != "D": print("→ БАРУУН"); last = "D"
                cur = turn_right(ser, cur)
            else:
                if last is not None:
                    print("→ ЗОГСОЛТ"); last = None
                    cur = return_to_stand(ser, cur)
                # settle: motion blur/frame desync-ээс цэвэршсэн хэдэн frame хүлээнэ
                time.sleep(0.25)
                set_moving(False)
                time.sleep(0.05)
    finally:
        set_moving(False)
        return_to_stand(ser, cur)
        send_set(ser, 26, [0], "RELAY OFF")
        time.sleep(0.5)
        print("[ROBOT] thread дууслаа.")


# =============================================================
# STEREO + YOLO ТУСЛАХ
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

def roi_distance(disp, focal, y1, y2, x1, x2):
    roi = disp[y1:y2, x1:x2]; valid = roi[roi > 0]
    if len(valid) < 10: return None
    med = np.median(valid)
    if med <= 0: return None
    d = (focal * BASELINE_M) / med
    return d if 0.1 < d < MAP_MAX_DISTANCE else None

def pixel_to_world(cx_rel, depth, focal, rx, ry, rth):
    bearing = math.atan2(cx_rel, focal)
    rfx = depth * math.cos(bearing)
    rfy = -depth * math.sin(bearing)
    wx = rx + rfx * math.cos(rth) - rfy * math.sin(rth)
    wy = ry + rfx * math.sin(rth) + rfy * math.cos(rth)
    return wx, wy


# =============================================================
# MAP RENDERER  (grid-гүй, чөлөөт харагдац)
# =============================================================
def render_map():
    W, H, pw = MAP_W, MAP_H, PANEL_W
    mw = W - pw
    img = np.full((H, W, 3), COL_BG, dtype=np.uint8)

    # Header
    cv2.rectangle(img, (0, 0), (W, 40), COL_HEADER_BG, -1)
    cv2.putText(img, "DREAM Hexapod  -  Manual Controller",
                (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT_HEAD, 2)

    with path_lock:
        trail = list(path_trail)
    rx, ry, rth = get_pose()
    all_lms   = world_map.snapshot_all()                      # (x,y,cls,cnt,confirmed)
    landmarks = [(x,y,c,n) for (x,y,c,n,cf) in all_lms if cf] # confirmed only

    # Бүх цэгийг нэгтгэж bounding box тооцоолно (pending-ыг ч оруулна)
    all_pts = [(0.0, 0.0), (rx, ry)] + [(x, y) for x, y, _, _, _ in all_lms] + trail
    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
    cx = (max(xs) + min(xs)) / 2; cy = (max(ys) + min(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) + 1.5

    margin = 60
    scale = min((mw - 2*margin) / span, (H - 40 - 2*margin) / span)

    def w2s(wx, wy):
        sx = int(mw/2 + (wx - cx) * scale)
        sy = int((H+40)/2 - (wy - cy) * scale)
        return sx, sy

    # 1m grid lines
    gx0 = math.floor(cx - span/2); gx1 = math.ceil(cx + span/2)
    gy0 = math.floor(cy - span/2); gy1 = math.ceil(cy + span/2)
    for xi in range(int(gx0), int(gx1)+1):
        p1 = w2s(xi, gy0); p2 = w2s(xi, gy1)
        cv2.line(img, p1, p2, COL_GRID_LINE, 1)
        cv2.putText(img, f"{xi}m", (p1[0]+2, H-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_TEXT_DIM, 1)
    for yi in range(int(gy0), int(gy1)+1):
        p1 = w2s(gx0, yi); p2 = w2s(gx1, yi)
        cv2.line(img, p1, p2, COL_GRID_LINE, 1)
        cv2.putText(img, f"{yi}m", (8, p1[1]-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_TEXT_DIM, 1)

    # Trail
    if len(trail) >= 2:
        pts = [w2s(px, py) for px, py in trail]
        for k in range(1, len(pts)):
            cv2.line(img, pts[k-1], pts[k], COL_TRAIL, 1, cv2.LINE_AA)

    # START
    sx0, sy0 = w2s(0, 0)
    cv2.circle(img, (sx0, sy0), 9, COL_START, -1)
    cv2.circle(img, (sx0, sy0), 11, (255,255,255), 1)
    cv2.putText(img, "START", (sx0+13, sy0+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_START, 1)

    # Pending landmarks — бүдэгхэн, confirmed бус тул зөвхөн хүлээгдэж байгаа гэж зурна
    for (lx, ly, cls, cnt, cf) in all_lms:
        if cf: continue
        lsx, lsy = w2s(lx, ly)
        cv2.circle(img, (lsx, lsy), 4, (120, 120, 130), 1)

    # Confirmed landmarks
    for idx, (lx, ly, cls, cnt) in enumerate(landmarks):
        lsx, lsy = w2s(lx, ly)
        col = TRACK_COLORS[idx % len(TRACK_COLORS)]
        cv2.circle(img, (lsx, lsy), 8, col, -1)
        cv2.circle(img, (lsx, lsy), 10, (255,255,255), 1)
        cv2.putText(img, f"#{idx+1}", (lsx+11, lsy-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # Robot
    rsx, rsy = w2s(rx, ry)
    ax = rsx + int(math.cos(rth) * 30)
    ay = rsy - int(math.sin(rth) * 30)
    cv2.circle(img, (rsx, rsy), 12, COL_ROBOT, 2)
    cv2.arrowedLine(img, (rsx, rsy), (ax, ay), COL_ROBOT, 3, tipLength=0.35)

    # ---- Right panel ----
    px0 = mw
    cv2.rectangle(img, (px0, 40), (W, H), COL_PANEL_BG, -1)
    cv2.line(img, (px0, 40), (px0, H), COL_DIVIDER, 1)
    pad, y = 14, 60

    def section(title, y):
        cv2.putText(img, title, (px0+pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT_HEAD, 1)
        cv2.line(img, (px0+pad, y+5), (W-pad, y+5), COL_DIVIDER, 1)
        return y + 25

    def tline(text, y, color=COL_TEXT, sc=0.48):
        cv2.putText(img, text, (px0+pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, color, 1)
        return y + 22

    y = section("ROBOT", y)
    y = tline(f"X:  {rx:+.2f} m", y)
    y = tline(f"Y:  {ry:+.2f} m", y)
    y = tline(f"Th: {math.degrees(rth) % 360:.1f} deg", y)
    y += 6

    y = section(f"LANDMARKS  ({len(landmarks)})", y)
    for idx, (lx, ly, cls, cnt) in enumerate(landmarks[:10]):
        col = TRACK_COLORS[idx % len(TRACK_COLORS)]
        y = tline(f"#{idx+1} {cls:8s} ({lx:+.2f},{ly:+.2f}) x{cnt}",
                  y, col, sc=0.42)
    if len(landmarks) > 10:
        y = tline(f"... +{len(landmarks)-10} more", y, COL_TEXT_DIM)
    y += 6

    y = section("SESSION", y)
    if session_start:
        mm, ss = divmod(int(time.time() - session_start), 60)
        y = tline(f"Elapsed: {mm:02d}:{ss:02d}", y)
    y = tline(f"Trail:   {len(trail)} pts", y)

    y = section("CONTROLS", y)
    for txt in ["W = урагш  S = хойш",
                "A = зүүн   D = баруун",
                "Q / ESC = гарах"]:
        y = tline(txt, y, COL_TEXT_DIM, sc=0.44)

    cv2.imshow("2D Map", img)
    return img


# =============================================================
# КАМЕРЫН ЦИКЛ  (main thread)
# =============================================================
def camera_loop():
    yolo = YOLO(YOLO_MODEL)

    cam = cv2.VideoCapture(CAMERA_INDEX)
    # Resolution бууруулснаар FPS нэмэгдэнэ (нэг нүд: 640×480)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,  480)

    ret, frame0 = cam.read()
    frame0 = cv2.flip(frame0, -1)
    if not ret:
        print("Камер нээгдсэнгүй!"); stop_event.set(); return

    half_w   = frame0.shape[1] // 2
    focal_px = (half_w / 2) / math.tan(math.radians(FOV_DEG / 2))

    # Цонхнууд
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 1280, 480)
    cv2.moveWindow("Camera", 30, 30)
    cv2.namedWindow("2D Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("2D Map", MAP_W, MAP_H)
    cv2.moveWindow("2D Map", 30, 30 + 480 + 20)

    stereo_l, stereo_r, wls = create_stereo()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    disp_hist = []; dist_by_id = {}

    # ---- Бодит FPS хэмжих (VideoWriter-ийн FPS = бодит FPS → real-time бичлэг) ----
    print(f"[CAM] FPS хэмжиж байна ({FPS_BENCHMARK_FRAMES} frame)...")
    stereo_bench, _, _ = create_stereo()
    t_bench = time.time()
    for _ in range(FPS_BENCHMARK_FRAMES):
        rok, frm = cam.read()
        if not rok: continue
        frm = cv2.flip(frm, -1)
        lf = frm[:, frm.shape[1]//2:]
        gf = cv2.resize(cv2.cvtColor(lf, cv2.COLOR_BGR2GRAY),
                        None, fx=1/STEREO_SCALE, fy=1/STEREO_SCALE)
        stereo_bench.compute(gf, gf)
        yolo.track(lf, verbose=False, conf=YOLO_CONF,
                   persist=False, imgsz=320)
    measured_fps = FPS_BENCHMARK_FRAMES / max(time.time() - t_bench, 0.1)
    measured_fps = max(1.0, round(measured_fps, 1))
    print(f"[CAM] бодит FPS: {measured_fps:.1f}  →  VideoWriter FPS = {measured_fps:.1f}")

    # Video writers  (measured_fps ашиглана → бичлэг real-time тоглоно)
    cam_w = map_w = None
    if RECORD_VIDEO:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cam_w = cv2.VideoWriter(f"ctrl_cam_{ts}.mp4", fourcc, measured_fps,
                                (2*half_w, frame0.shape[0]))
        map_w = cv2.VideoWriter(f"ctrl_map_{ts}.mp4", fourcc, measured_fps,
                                (MAP_W, MAP_H))
        if cam_w.isOpened(): print(f"[REC] cam  → ctrl_cam_{ts}.mp4")
        if map_w.isOpened(): print(f"[REC] map  → ctrl_map_{ts}.mp4")

    print(f"[CAM] focal={focal_px:.1f}px  res={half_w}x{frame0.shape[0]}")

    frame_count  = 0
    map_img_last = None
    settle_left  = 0
    prev_skip    = False
    motion_buf   = {}   # tid → [(dist, cx_rel, rx, ry, rth), ...]
    class_by_id  = {}

    try:
        while not stop_event.is_set():
            ret, frame = cam.read()
            if not ret: continue
            frame = cv2.flip(frame, -1)
            frame_count += 1

            moving_now = is_moving()
            if moving_now:
                settle_left = MOTION_SETTLE_FRAMES
            elif settle_left > 0:
                settle_left -= 1
            skip_detect  = DEFERRED_COMMIT_WHILE_MOVING and (moving_now or settle_left > 0)
            just_stopped = prev_skip and not skip_detect
            prev_skip    = skip_detect

            w = frame.shape[1]
            left  = frame[:, w//2:]
            right = frame[:, :w//2]

            # Stereo
            gl = clahe.apply(cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY))
            gr = clahe.apply(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
            gls = cv2.resize(gl, None, fx=1/STEREO_SCALE, fy=1/STEREO_SCALE)
            grs = cv2.resize(gr, None, fx=1/STEREO_SCALE, fy=1/STEREO_SCALE)
            dl = stereo_l.compute(gls, grs)
            dr = stereo_r.compute(grs, gls)
            disp_s = wls.filter(dl, gls, None, dr).astype(np.float32) / 16.0
            disp_raw = cv2.resize(disp_s, (left.shape[1], left.shape[0])) * STEREO_SCALE
            disp_hist.append(disp_raw)
            if len(disp_hist) > DISP_HISTORY_SIZE: disp_hist.pop(0)
            disparity = np.mean(disp_hist, axis=0)

            # YOLO
            ld = left.copy()
            lh, lw = left.shape[:2]
            results = yolo.track(left, verbose=False, conf=YOLO_CONF,
                                 persist=True, tracker="botsort.yaml", imgsz=480)
            active = set()
            rx, ry, rth = get_pose()
            for r in results:
                if r.boxes is None or r.boxes.id is None: continue
                for box in r.boxes:
                    cid = int(box.cls[0])
                    if cid not in BOTTLE_CLASSES or box.id is None: continue
                    tid = int(box.id[0]); active.add(tid)
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x1,y1 = max(0,x1),max(0,y1); x2,y2 = min(lw,x2),min(lh,y2)
                    cls_name = BOTTLE_CLASSES[cid]
                    raw_d = roi_distance(disparity, focal_px, y1,y2,x1,x2)
                    dlabel = ""
                    if raw_d is not None:
                        prev = dist_by_id.get(tid)
                        sm = raw_d if prev is None else prev*(1-EMA_ALPHA)+raw_d*EMA_ALPHA
                        dist_by_id[tid] = sm; dlabel = f" {sm:.2f}m"
                        cx_rel = (x1+x2)//2 - lw//2
                        class_by_id[tid] = cls_name
                        if skip_detect:
                            buf = motion_buf.setdefault(tid, [])
                            buf.append((sm, cx_rel, rx, ry, rth))
                            if len(buf) > MOTION_BUFFER_MAX: buf.pop(0)
                        else:
                            wx, wy = pixel_to_world(cx_rel, sm, focal_px, rx, ry, rth)
                            world_map.add(wx, wy, cls_name, tid)
                    col = TRACK_COLORS[tid % len(TRACK_COLORS)]
                    lab = f"ID:{tid} {cls_name}{dlabel}"
                    cv2.rectangle(ld, (x1,y1),(x2,y2), col, 2)
                    (tw,th_),_ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(ld, (x1,y1-th_-10),(x1+tw,y1), col, -1)
                    cv2.putText(ld, lab, (x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            if len(dist_by_id) > 50:
                for k in [k for k in dist_by_id if k not in active]:
                    del dist_by_id[k]

            if just_stopped and motion_buf:
                for tid, samples in motion_buf.items():
                    cls_name = class_by_id.get(tid)
                    if not samples or cls_name is None: continue
                    wxs, wys = [], []
                    for (sm, cxr, rx_s, ry_s, rth_s) in samples:
                        wx, wy = pixel_to_world(cxr, sm, focal_px, rx_s, ry_s, rth_s)
                        wxs.append(wx); wys.append(wy)
                    wx_med = float(np.median(wxs)); wy_med = float(np.median(wys))
                    kx = [x for x,y in zip(wxs,wys) if math.hypot(x-wx_med,y-wy_med)<0.5]
                    ky = [y for x,y in zip(wxs,wys) if math.hypot(x-wx_med,y-wy_med)<0.5]
                    if kx: wx_med, wy_med = float(np.median(kx)), float(np.median(ky))
                    world_map.add(wx_med, wy_med, cls_name, tid)
                motion_buf.clear()

            if skip_detect:
                cv2.putText(ld, "MOVING  (detection paused)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)

            combined = np.hstack([ld, right])
            if cam_w and cam_w.isOpened(): cam_w.write(combined)
            cv2.imshow("Camera", combined)

            # Map-ыг MAP_RENDER_EVERY frame тутамд л шинэчилнэ
            if frame_count % MAP_RENDER_EVERY == 0:
                map_img_last = render_map()
            elif map_img_last is not None:
                cv2.imshow("2D Map", map_img_last)
            if map_w and map_w.isOpened() and map_img_last is not None:
                map_w.write(map_img_last)

            if cv2.waitKey(1) == 27:
                stop_event.set(); break

    finally:
        if cam_w: cam_w.release()
        if map_w: map_w.release()
        cam.release()
        cv2.destroyAllWindows()
        print("\n===== FINAL MAP =====")
        for i,(lx,ly,cls,cnt) in enumerate(world_map.snapshot()):
            print(f"  #{i+1} {cls}: ({lx:.2f},{ly:.2f})  x{cnt}")


# =============================================================
# АЖИЛЛУУЛАХ
# =============================================================
if __name__ == "__main__":
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    try:
        kb = threading.Thread(target=keyboard_thread, args=(ser,), daemon=True)
        kb.start()
        camera_loop()
        stop_event.set()
        kb.join(timeout=5.0)
    finally:
        ser.close()
        print("Дууслаа.")
