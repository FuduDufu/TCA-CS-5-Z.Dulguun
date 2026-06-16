import serial
import time
import math
import threading
import datetime
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from ultralytics import YOLO

# =============================================================
# БИЧЛЭГ
# =============================================================
RECORD_VIDEO = True
RECORD_FPS = 20.0          # бичлэгийн frame rate (бодит loop-ын ойролцоогоор)

# =============================================================
# SERIAL / КАМЕР ТОХИРГОО
# =============================================================
PORT = "COM4"
BAUD = 115200
CMD_SET = 0x53 | 0x80

CAMERA_INDEX = 1
FOV_DEG = 96
BASELINE_M = 0.065

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
COXA_SWING = 150
FEMUR_LIFT = 200
TIBIA_LIFT = 150

# Биеийн жингээр дооно суух үед газар дээрх хөлийг илүү дооно түлхэж нөхдөг параметр.
GROUND_PRESS = 80

TRANSITION_STEPS = 15
TRANSITION_DELAY = 0.03

# =============================================================
# ODOMETRY КАЛИБРАЦ
# =============================================================
# Нэг бүрэн A+B мөчлөгт робот урагш хэр шилжих (метр)
STEP_FORWARD_PER_CYCLE = 0.04     # ← туршилтаар тохируулах

# Нэг бүрэн A+B эргэлтийн мөчлөгт робот хэр зэрэг эргэх (радиан).
# COXA_SWING = 150 μs, нэг мөчлөгт coxa 300 μs sweep ≈ 30°.
# Хэрэв 90°-д хэтэрхий олон/цөөн эргэж байвал энэ утгыг тохируулна.
TURN_ANGLE_PER_CYCLE = math.radians(30)   # ← туршилтаар тохируулах

# =============================================================
# GRID ПАРАМЕТРҮҮД
# =============================================================
CELL_SIZE = 0.45          # 45 см — нэг нүдний хэмжээ (робот 42 см → багтана)

# =============================================================
# ОБЪЕКТЫН MAP
# =============================================================
# Odometry drift-г харгалзан томсгосон — нэг л bottle-ийг давхар оруулахаас сэргийлнэ
LANDMARK_MERGE_THRESHOLD = 0.80
# Confirmed landmark: ≥ энэ тоотой удаа тогтвортой ажиглагдсан үед л map-д гарна
LANDMARK_MIN_OBS       = 3
# Median-д ашиглах ажиглалтын тоо (outlier байрлалуудаас хамгаална)
LANDMARK_POS_HISTORY   = 15
# Робот хөдөлж байх үед шууд map-д нэмэхгүй — буферт хадгалж, зогссоны дараа
# тухайн үеийн pose-уудыг ашиглан нэг удаа commit хийнэ.
DEFERRED_COMMIT_WHILE_MOVING = True
# Зогссоны дараа хэдэн frame-ийг бас алгасах (stereo disp_history цэвэрлэгдэх)
MOTION_SETTLE_FRAMES   = 4
MOTION_BUFFER_MAX      = 20
MAP_WINDOW_W = 1100
MAP_WINDOW_H = 800
MAP_PANEL_W = 380           # баруун талын мэдээлэлийн самбар
MAP_MAX_DISTANCE = 8.0

# Замын ул мөр
PATH_TRAIL_MAX = 600

# =============================================================
# ТОГТОЦ
# =============================================================
stop_event = threading.Event()

robot_pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
pose_lock = threading.Lock()

# Зохицуулалт: grid-ийн хэмжээс + очсон нүднүүд
grid_config = {"rows": 0, "cols": 0}
visited_cells = set()
visited_lock = threading.Lock()

# Замын ул мөр (робот хаагуур явсан)
path_trail = []
path_lock = threading.Lock()

# Эхэлсэн хугацаа (UI-д харуулах зориулалттай)
session_start_time = None

# Робот одоо хөдөлж байна уу? Gait мөчлөгийн үед True болгосноор
# камерын цикл нь тухайн үеийн (motion blur-тай, odometry drift-тэй)
# stereo detection-ыг map-д бүртгэхгүй.
robot_moving = False
motion_lock  = threading.Lock()

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

def advance_pose(dx_robot, dy_robot=0.0, dtheta=0.0):
    with pose_lock:
        th = robot_pose["theta"]
        robot_pose["x"] += dx_robot * math.cos(th) - dy_robot * math.sin(th)
        robot_pose["y"] += dx_robot * math.sin(th) + dy_robot * math.cos(th)
        robot_pose["theta"] += dtheta
        nx, ny = robot_pose["x"], robot_pose["y"]
    with path_lock:
        path_trail.append((nx, ny))
        if len(path_trail) > PATH_TRAIL_MAX:
            del path_trail[:len(path_trail) - PATH_TRAIL_MAX]

def mark_visited(row, col):
    with visited_lock:
        visited_cells.add((row, col))


# =============================================================
# LANDMARK MAP
# =============================================================
class Landmark:
    def __init__(self, x, y, cls, track_id):
        self.cls       = cls
        self.track_ids = {track_id}
        self.xs        = [x]
        self.ys        = [y]
        self.count     = 1
        self.x         = x
        self.y         = y
        self.confirmed = False

    def update(self, x, y, track_id):
        self.xs.append(x); self.ys.append(y)
        if len(self.xs) > LANDMARK_POS_HISTORY:
            self.xs.pop(0); self.ys.pop(0)
        # Median → нэг гэнэтийн буруу байрлал бүх зогсоолыг зөөхгүй
        self.x = float(np.median(self.xs))
        self.y = float(np.median(self.ys))
        self.count += 1
        self.track_ids.add(track_id)
        if self.count >= LANDMARK_MIN_OBS:
            self.confirmed = True


class WorldMap:
    def __init__(self):
        self.landmarks = []
        self.lock = threading.Lock()

    def add_detection(self, wx, wy, cls, track_id):
        with self.lock:
            # 1) Track-ID таарч байгаа байвал хамгийн найдвартай
            for lm in self.landmarks:
                if lm.cls == cls and track_id in lm.track_ids:
                    lm.update(wx, wy, track_id); return
            # 2) Track алдагдсан ч ойролцоо байгаа бол ижил landmark гэж үзнэ
            best = None; best_d = LANDMARK_MERGE_THRESHOLD
            for lm in self.landmarks:
                if lm.cls != cls: continue
                d = math.hypot(lm.x - wx, lm.y - wy)
                if d < best_d:
                    best_d = d; best = lm
            if best is not None:
                best.update(wx, wy, track_id)
            else:
                self.landmarks.append(Landmark(wx, wy, cls, track_id))

    def snapshot(self):
        """Зөвхөн confirmed landmark-уудыг буцаана (map-д харагдана)."""
        with self.lock:
            return [(lm.x, lm.y, lm.cls, lm.count)
                    for lm in self.landmarks if lm.confirmed]

    def snapshot_all(self):
        """Confirmed + pending landmark (UI-д тус тусад нь зурах)."""
        with self.lock:
            return [(lm.x, lm.y, lm.cls, lm.count, lm.confirmed)
                    for lm in self.landmarks]

world_map = WorldMap()


# =============================================================
# SERIAL ФУНКЦҮҮД
# =============================================================
def encode_set(start, values):
    data = bytearray([CMD_SET, start & 0x7F, len(values) & 0x7F])
    for v in values:
        data.append(v & 0x7F)
        data.append((v >> 7) & 0x7F)
    return data

def send_set(ser, start, values, label=""):
    pkt = encode_set(start, values)
    if label: print(label)
    ser.write(pkt)

def ease_in_out(t):
    return (1 - math.cos(math.pi * t)) / 2

def build_values(pose):
    values = [1500] * 18
    for name, pulse in pose.items():
        values[PIN[name]] = pulse
    return values

def send_pose(ser, pose):
    send_set(ser, 0, build_values(pose))

def interpolate(pa, pb, alpha):
    return {k: int(round(pa[k] + (pb[k] - pa[k]) * alpha)) for k in pa}

def move_smooth(ser, pf, pt, steps=15, delay=0.03):
    for i in range(1, steps + 1):
        if stop_event.is_set(): return pf
        alpha = ease_in_out(i / steps)
        send_pose(ser, interpolate(pf, pt, alpha))
        time.sleep(delay)
    return pt


# =============================================================
# OFFSET + POSE
# =============================================================
def coxa_offset(leg, offset):
    s = leg + "1"
    return STAND[s] + offset if leg.startswith("L") else STAND[s] - offset

def femur_offset(leg, offset):
    s = leg + "2"
    return STAND[s] - offset if leg.startswith("L") else STAND[s] + offset

def tibia_offset(leg, offset):
    s = leg + "3"
    return STAND[s] + offset if leg.startswith("L") else STAND[s] - offset

def stand_pose():
    return dict(STAND)

def press_down(pose, leg):
    """Газар дээрх хөлийг STAND-аас илүү дооно түлхэж биеийн жинг нөхнө."""
    pose[leg + "2"] = femur_offset(leg, -GROUND_PRESS)
    pose[leg + "3"] = tibia_offset(leg, -GROUND_PRESS)

# ---- Урагш/Хойш ----
def step_pose_walk(swing, push, direction):
    pose = stand_pose()
    for leg in swing:
        pose[leg + "2"] = femur_offset(leg, FEMUR_LIFT)
        pose[leg + "3"] = tibia_offset(leg, TIBIA_LIFT)
        pose[leg + "1"] = coxa_offset(leg, direction * COXA_SWING)
    for leg in push:
        pose[leg + "1"] = coxa_offset(leg, direction * (-COXA_SWING))
        press_down(pose, leg)
    return pose

def down_pose_walk(down, push, direction):
    pose = stand_pose()
    for leg in down:
        pose[leg + "1"] = coxa_offset(leg, direction * COXA_SWING)
        press_down(pose, leg)
    for leg in push:
        pose[leg + "1"] = coxa_offset(leg, direction * (-COXA_SWING))
        press_down(pose, leg)
    return pose

# ---- Эргэлт ----
def turn_sign(leg, turn_dir):
    return turn_dir if leg.startswith("L") else -turn_dir

def step_pose_turn(swing, push, turn_dir):
    pose = stand_pose()
    for leg in swing:
        pose[leg + "2"] = femur_offset(leg, FEMUR_LIFT)
        pose[leg + "3"] = tibia_offset(leg, TIBIA_LIFT)
        pose[leg + "1"] = coxa_offset(leg, turn_sign(leg, turn_dir) * COXA_SWING)
    for leg in push:
        pose[leg + "1"] = coxa_offset(leg, turn_sign(leg, turn_dir) * (-COXA_SWING))
        press_down(pose, leg)
    return pose

def down_pose_turn(down, push, turn_dir):
    pose = stand_pose()
    for leg in down:
        pose[leg + "1"] = coxa_offset(leg, turn_sign(leg, turn_dir) * COXA_SWING)
        press_down(pose, leg)
    for leg in push:
        pose[leg + "1"] = coxa_offset(leg, turn_sign(leg, turn_dir) * (-COXA_SWING))
        press_down(pose, leg)
    return pose


# =============================================================
# НЭГ A+B МӨЧЛӨГ
# =============================================================
def do_cycle(ser, current, make_step, make_down, param):
    p1 = make_step(GROUP_A, GROUP_B, param)
    current = move_smooth(ser, current, p1, TRANSITION_STEPS, TRANSITION_DELAY)
    p2 = make_down(GROUP_A, GROUP_B, param)
    current = move_smooth(ser, current, p2, TRANSITION_STEPS // 2, TRANSITION_DELAY)
    p3 = make_step(GROUP_B, GROUP_A, param)
    current = move_smooth(ser, current, p3, TRANSITION_STEPS, TRANSITION_DELAY)
    p4 = make_down(GROUP_B, GROUP_A, param)
    current = move_smooth(ser, current, p4, TRANSITION_STEPS // 2, TRANSITION_DELAY)
    return current

def return_to_stand(ser, current):
    """Хөлөө өргөөд STAND руу буцна (чирэхгүй).
    GROUP_A-г өргөөд coxa-г STAND руу шилжүүлж буулгаад, GROUP_B-г мөн адил."""
    stand = stand_pose()

    def lift_and_place(pose_from, group):
        p_up = dict(pose_from)
        for leg in group:
            p_up[leg + "2"] = femur_offset(leg, FEMUR_LIFT)
            p_up[leg + "3"] = tibia_offset(leg, TIBIA_LIFT)
            p_up[leg + "1"] = stand[leg + "1"]
        pose_from = move_smooth(ser, pose_from, p_up, TRANSITION_STEPS, TRANSITION_DELAY)

        p_dn = dict(pose_from)
        for leg in group:
            p_dn[leg + "2"] = stand[leg + "2"]
            p_dn[leg + "3"] = stand[leg + "3"]
        pose_from = move_smooth(ser, pose_from, p_dn, TRANSITION_STEPS // 2, TRANSITION_DELAY)
        return pose_from

    current = lift_and_place(current, GROUP_A)
    current = lift_and_place(current, GROUP_B)
    return current


def walk_forward_cycle(ser, current):
    set_moving(True)
    current = do_cycle(ser, current, step_pose_walk, down_pose_walk, +1)
    advance_pose(STEP_FORWARD_PER_CYCLE)
    return current

def walk_backward_cycle(ser, current):
    set_moving(True)
    current = do_cycle(ser, current, step_pose_walk, down_pose_walk, -1)
    advance_pose(-STEP_FORWARD_PER_CYCLE)
    return current

def turn_right_cycle(ser, current):
    # Баруун эргэх (cw, дэлхий фрэйм-д theta буурна)
    set_moving(True)
    current = do_cycle(ser, current, step_pose_turn, down_pose_turn, +1)
    advance_pose(0, 0, -TURN_ANGLE_PER_CYCLE)
    return current

def turn_left_cycle(ser, current):
    set_moving(True)
    current = do_cycle(ser, current, step_pose_turn, down_pose_turn, -1)
    advance_pose(0, 0, +TURN_ANGLE_PER_CYCLE)
    return current


# =============================================================
# НҮД/90°-ЫН МЕТА ФУНКЦҮҮД
# =============================================================
def walk_one_cell(ser, current):
    """45 см урагш. STEP_FORWARD_PER_CYCLE-тэй тохируулан мөчлөг давтана."""
    n = max(1, int(round(CELL_SIZE / STEP_FORWARD_PER_CYCLE)))
    for _ in range(n):
        if stop_event.is_set(): return current
        current = walk_forward_cycle(ser, current)
    # Нэг нүд дуусав → зогсоод settle хийхийн тулд motion flag-ийг унтраана
    set_moving(False)
    time.sleep(0.3)
    return current

def turn_right_90(ser, current):
    n = max(1, int(round((math.pi / 2) / TURN_ANGLE_PER_CYCLE)))
    for _ in range(n):
        if stop_event.is_set(): return current
        current = turn_right_cycle(ser, current)
    set_moving(False)
    time.sleep(0.3)
    return current

def turn_left_90(ser, current):
    n = max(1, int(round((math.pi / 2) / TURN_ANGLE_PER_CYCLE)))
    for _ in range(n):
        if stop_event.is_set(): return current
        current = turn_left_cycle(ser, current)
    set_moving(False)
    time.sleep(0.3)
    return current


# =============================================================
# LAWNMOWER (ЗИГЗАГ) PATTERN
# =============================================================
# rows = нэг баганад хэдэн нүд (урагш чиглэлд)
# cols = нийт хэдэн багана (баруун чиглэлд)
#
# Робот эхлэх нүд: (0, 0). Эхлээд +X (урагш) чиглэнэ.
# Нүд (i, j) → world (i*CELL, -j*CELL)   [+Y=зүүн тул баруун тийш j нэмэгдэхэд y буурна]

def cell_index_from_pose():
    """Одоогийн robot pose-оос ойролцоо нүдний индекс (row, col).
    Робот эхлэх чиглэл (+X) = багана (col) нэмэгдэх зүг.
    Row 0 дээр, баруун тийш эргэхэд +row зүг (-Y)."""
    x, y, _ = get_pose()
    col = int(round(x / CELL_SIZE))
    row = int(round(-y / CELL_SIZE))
    return row, col

def execute_lawnmower(ser, rows, cols):
    current = stand_pose()
    send_pose(ser, current)
    time.sleep(1.0)

    # Эхлэх нүдийг visited болгоно: (row=0, col=0)
    mark_visited(0, 0)

    # Мөр тус бүрийг snake байдлаар туулна
    for row in range(rows):
        if stop_event.is_set(): break

        # Нэг мөрд cols-1 нүд урагш алхана (эхлэл аль хэдийн тэр мөрд байна)
        for _ in range(cols - 1):
            if stop_event.is_set(): break
            current = walk_one_cell(ser, current)
            r, c = cell_index_from_pose()
            mark_visited(r, c)
            print(f"[GRID] cell ({r},{c})")

        # Сүүлийн мөр биш бол дараагийн мөр рүү шилжинэ
        if row < rows - 1 and not stop_event.is_set():
            if row % 2 == 0:
                # Баруун тийш явсан (6-н нүдний төгсгөлд) →
                # баруун эргэж 1 нүд → дахин баруун эргэж зүүн рүү харна
                print("[GRID] дараагийн мөр рүү шилжиж байна (↓ →←)")
                current = turn_right_90(ser, current)
                current = walk_one_cell(ser, current)
                current = turn_right_90(ser, current)
            else:
                # Зүүн тийш явсан → зүүн эргэж 1 нүд → дахин зүүн эргэж баруун руу
                print("[GRID] дараагийн мөр рүү шилжиж байна (↓ →→)")
                current = turn_left_90(ser, current)
                current = walk_one_cell(ser, current)
                current = turn_left_90(ser, current)

            r, c = cell_index_from_pose()
            mark_visited(r, c)

    # Зогсолт — хөлөө өргөөд STAND руу буцна (чирэхгүй)
    return_to_stand(ser, current)
    print("[GRID] зам дууслаа.")


STARTUP_DELAY_SEC = 130

def movement_thread(ser, rows, cols):
    try:
        # Камер болон map цонх бэлдэх хугацаа — энэ үед робот зогсож байна
        print(f"[MOVEMENT] camera/map бэлдэх {STARTUP_DELAY_SEC} сек хүлээнэ...")
        for s in range(STARTUP_DELAY_SEC, 0, -1):
            if stop_event.is_set():
                print("[MOVEMENT] хүлээлтийн үед зогсоов.")
                return
            if s % 10 == 0 or s <= 5:
                print(f"[MOVEMENT] ... {s} сек")
            time.sleep(1.0)

        send_set(ser, 26, [1], "RELAY ON")
        time.sleep(1.0)
        execute_lawnmower(ser, rows, cols)
    except Exception as e:
        print(f"[MOVEMENT] алдаа: {e}")
    finally:
        send_set(ser, 26, [0], "RELAY OFF")
        time.sleep(0.5)
        print("[MOVEMENT] thread дууслаа.")


# =============================================================
# YOLO + STEREO
# =============================================================
YOLO_MODEL = "../models/yolov8n.pt"
YOLO_CONF = 0.4
BOTTLE_CLASSES = {39: "bottle", 41: "cup"}

STEREO_SCALE = 2
DISP_HISTORY_SIZE = 5
EMA_ALPHA = 0.3

TRACK_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 165, 255), (128, 0, 255), (255, 128, 0),
    (0, 255, 128), (128, 255, 0), (255, 0, 128), (0, 128, 255),
]


def create_stereo():
    ws = 5
    left = cv2.StereoSGBM.create(
        minDisparity=0, numDisparities=128, blockSize=ws,
        P1=8 * 3 * ws ** 2, P2=32 * 3 * ws ** 2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    right = cv2.ximgproc.createRightMatcher(left)
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left)
    wls.setLambda(8000); wls.setSigmaColor(1.5)
    return left, right, wls

def roi_distance(disparity, focal_px, y1, y2, x1, x2):
    roi = disparity[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if len(valid) < 10: return None
    med = np.median(valid)
    if med <= 0: return None
    d = (focal_px * BASELINE_M) / med
    return d if 0.1 < d < MAP_MAX_DISTANCE else None

def pixel_to_world(cx_rel, depth, focal_px, rx, ry, rth):
    bearing = math.atan2(cx_rel, focal_px)
    rfx = depth * math.cos(bearing)
    rfy = -depth * math.sin(bearing)
    wx = rx + rfx * math.cos(rth) - rfy * math.sin(rth)
    wy = ry + rfx * math.sin(rth) + rfy * math.cos(rth)
    return wx, wy


# =============================================================
# 2D GRID MAP RENDERER
# =============================================================
# Өнгө сан
COL_BG          = (24, 24, 30)
COL_PANEL_BG    = (40, 40, 50)
COL_HEADER_BG   = (55, 55, 75)
COL_GRID_BORDER = (90, 90, 110)
COL_CELL_LINE   = (75, 75, 90)
COL_CELL_VISIT  = (45, 110, 70)
COL_CELL_CUR    = (60, 200, 255)
COL_TEXT        = (220, 220, 230)
COL_TEXT_DIM    = (160, 160, 175)
COL_TEXT_HEAD   = (250, 230, 140)
COL_ROBOT       = (60, 255, 80)
COL_PATH        = (90, 200, 255)
COL_START       = (40, 200, 255)
COL_LANDMARK    = (255, 200, 60)


def snake_number(row, col, cols):
    """Snake/lawnmower дарааллаар нүдийг 1-ээс дугаарлана."""
    if row % 2 == 0:
        return row * cols + col + 1
    return (row + 1) * cols - col


def render_map():
    rows = grid_config["rows"]
    cols = grid_config["cols"]

    W = MAP_WINDOW_W
    H = MAP_WINDOW_H
    panel_w = MAP_PANEL_W
    map_w = W - panel_w

    img = np.full((H, W, 3), COL_BG, dtype=np.uint8)

    # ---------- Header ----------
    cv2.rectangle(img, (0, 0), (W, 40), COL_HEADER_BG, -1)
    cv2.putText(img, "DREAM Hexapod  -  Object Density Mapping",
                (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT_HEAD, 2)

    # ---------- Map хэсгийн scale ----------
    margin = 50
    map_top = 60
    map_bottom = H - 30
    avail_w = map_w - 2 * margin
    avail_h = map_bottom - map_top - 2 * margin

    world_w_m = cols * CELL_SIZE
    world_h_m = rows * CELL_SIZE

    if world_w_m <= 0 or world_h_m <= 0:
        cv2.imshow("2D Grid Map", img)
        return img

    scale_x = avail_w / world_w_m
    scale_y = avail_h / world_h_m
    scale = min(scale_x, scale_y)

    grid_w_px = world_w_m * scale
    grid_h_px = world_h_m * scale
    grid_left = (map_w - grid_w_px) / 2
    grid_top  = map_top + (avail_h + 2 * margin - grid_h_px) / 2

    # World → screen хувирал.
    # Нүд (0,0)-ын ЗҮҮН ДЭЭД булан (world: -0.5·CELL, +0.5·CELL) нь
    # screen-д (grid_left, grid_top) болно.
    grid_world_x_min = -0.5 * CELL_SIZE
    grid_world_y_max =  0.5 * CELL_SIZE

    def world_to_screen(wx, wy):
        sx = int(grid_left + (wx - grid_world_x_min) * scale)
        sy = int(grid_top  + (grid_world_y_max - wy) * scale)
        return sx, sy

    def cell_to_screen_rect(i, j):
        wx1 = (j - 0.5) * CELL_SIZE
        wx2 = (j + 0.5) * CELL_SIZE
        wy1 = (-i + 0.5) * CELL_SIZE
        wy2 = (-i - 0.5) * CELL_SIZE
        x1, y1 = world_to_screen(wx1, wy1)
        x2, y2 = world_to_screen(wx2, wy2)
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    # ---------- Snapshot-ууд ----------
    with visited_lock:
        visited_copy = set(visited_cells)
    with path_lock:
        path_copy = list(path_trail)
    rx, ry, rth = get_pose()
    cur_row = int(round(-ry / CELL_SIZE))
    cur_col = int(round(rx / CELL_SIZE))
    all_lms   = world_map.snapshot_all()
    landmarks = [(x, y, c, n) for (x, y, c, n, cf) in all_lms if cf]

    # ---------- Багана/мөрийн толгой ----------
    cell_px = scale * CELL_SIZE
    for j in range(cols):
        cx = int(grid_left + (j + 0.5) * cell_px)
        cv2.putText(img, f"C{j}", (cx - 12, int(grid_top) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT_DIM, 1)
    for i in range(rows):
        cy = int(grid_top + (i + 0.5) * cell_px)
        cv2.putText(img, f"R{i}", (int(grid_left) - 38, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT_DIM, 1)

    # ---------- Нүднүүд ----------
    for i in range(rows):
        for j in range(cols):
            x1, y1, x2, y2 = cell_to_screen_rect(i, j)
            if (i, j) in visited_copy:
                cv2.rectangle(img, (x1, y1), (x2, y2), COL_CELL_VISIT, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), COL_CELL_LINE, 1)
            num = snake_number(i, j, cols)
            cv2.putText(img, str(num), (x1 + 6, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1)

    # ---------- Одоогийн нүд ----------
    if 0 <= cur_row < rows and 0 <= cur_col < cols:
        x1, y1, x2, y2 = cell_to_screen_rect(cur_row, cur_col)
        cv2.rectangle(img, (x1, y1), (x2, y2), COL_CELL_CUR, 2)

    # ---------- Гадна хүрээ ----------
    gl, gt = int(grid_left), int(grid_top)
    gr_, gb_ = int(grid_left + grid_w_px), int(grid_top + grid_h_px)
    cv2.rectangle(img, (gl - 2, gt - 2), (gr_ + 2, gb_ + 2), COL_GRID_BORDER, 2)

    # ---------- Замын ул мөр ----------
    if len(path_copy) >= 2:
        pts = [world_to_screen(px, py) for (px, py) in path_copy]
        for k in range(1, len(pts)):
            cv2.line(img, pts[k - 1], pts[k], COL_PATH, 1, cv2.LINE_AA)

    # ---------- START ----------
    sx0, sy0 = world_to_screen(0.0, 0.0)
    cv2.circle(img, (sx0, sy0), 9, COL_START, -1)
    cv2.circle(img, (sx0, sy0), 11, (255, 255, 255), 1)
    cv2.putText(img, "START", (sx0 + 12, sy0 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_START, 1)

    # ---------- Робот ----------
    rsx, rsy = world_to_screen(rx, ry)
    arr_x = rsx + int(math.cos(rth) * 32)
    arr_y = rsy - int(math.sin(rth) * 32)
    cv2.circle(img, (rsx, rsy), 12, COL_ROBOT, 2)
    cv2.arrowedLine(img, (rsx, rsy), (arr_x, arr_y),
                    COL_ROBOT, 3, tipLength=0.35)

    # ---------- Pending (confirmed биш) — бүдэгхэн жижиг тойрог ----------
    for (lx, ly, cls, count, cf) in all_lms:
        if cf: continue
        lsx, lsy = world_to_screen(lx, ly)
        if gl - 30 <= lsx <= gr_ + 30 and gt - 30 <= lsy <= gb_ + 30:
            cv2.circle(img, (lsx, lsy), 4, (120, 120, 130), 1)

    # ---------- Confirmed landmark ----------
    for idx, (lx, ly, cls, count) in enumerate(landmarks):
        lsx, lsy = world_to_screen(lx, ly)
        if gl - 30 <= lsx <= gr_ + 30 and gt - 30 <= lsy <= gb_ + 30:
            color = TRACK_COLORS[idx % len(TRACK_COLORS)]
            cv2.circle(img, (lsx, lsy), 8, color, -1)
            cv2.circle(img, (lsx, lsy), 10, (255, 255, 255), 1)
            cv2.putText(img, f"#{idx+1}", (lsx + 11, lsy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ============================================================
    # БАРУУН САМБАР
    # ============================================================
    px0 = map_w
    cv2.rectangle(img, (px0, 40), (W, H), COL_PANEL_BG, -1)
    cv2.line(img, (px0, 40), (px0, H), COL_GRID_BORDER, 1)

    pad = 14
    y = 60

    def section(title, y):
        cv2.putText(img, title, (px0 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT_HEAD, 1)
        cv2.line(img, (px0 + pad, y + 5), (W - pad, y + 5), COL_GRID_BORDER, 1)
        return y + 25

    def text_line(text, y, color=COL_TEXT, scale_=0.48):
        cv2.putText(img, text, (px0 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale_, color, 1)
        return y + 22

    # GRID
    y = section("GRID", y)
    y = text_line(f"Size:    {rows} x {cols}  ({rows*cols} cells)", y)
    y = text_line(f"Cell:    {int(CELL_SIZE*100)} cm", y)
    y = text_line(f"Total:   {rows*CELL_SIZE:.1f} m x {cols*CELL_SIZE:.1f} m", y)
    visited_pct = 100.0 * len(visited_copy) / max(1, rows * cols)
    y = text_line(f"Visited: {len(visited_copy)}/{rows*cols}  ({visited_pct:.0f}%)",
                  y, COL_CELL_VISIT)
    y += 6

    # ROBOT
    y = section("ROBOT", y)
    y = text_line(f"X:  {rx:+.2f} m", y)
    y = text_line(f"Y:  {ry:+.2f} m", y)
    y = text_line(f"Th: {(math.degrees(rth) % 360):.1f} deg", y)
    y = text_line(f"Cell: R{cur_row}, C{cur_col}", y, COL_CELL_CUR)
    y += 6

    # LANDMARKS
    y = section(f"LANDMARKS  ({len(landmarks)})", y)
    show_max = 8
    for idx, (lx, ly, cls, count) in enumerate(landmarks[:show_max]):
        color = TRACK_COLORS[idx % len(TRACK_COLORS)]
        y = text_line(f"#{idx+1} {cls:8s}  ({lx:+.2f},{ly:+.2f})  x{count}",
                      y, color, scale_=0.42)
    if len(landmarks) > show_max:
        y = text_line(f"... +{len(landmarks)-show_max} more", y, COL_TEXT_DIM)
    y += 6

    # SESSION
    y = section("SESSION", y)
    if session_start_time is not None:
        elapsed = time.time() - session_start_time
        mm, ss = divmod(int(elapsed), 60)
        y = text_line(f"Elapsed: {mm:02d}:{ss:02d}", y)
    y = text_line(f"Trail:   {len(path_copy)} pts", y)

    # Footer
    cv2.putText(img, "ESC = exit", (px0 + pad, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_TEXT_DIM, 1)

    cv2.imshow("2D Grid Map", img)
    return img


# =============================================================
# КАМЕРЫН ЦИКЛ
# =============================================================
def camera_loop():
    yolo = YOLO(YOLO_MODEL)

    cam = cv2.VideoCapture(CAMERA_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    ret_init, frame_init = cam.read()
    frame_init = cv2.flip(frame_init, -1)
    if not ret_init:
        print("Камер нээгдсэнгүй!"); stop_event.set(); return

    half_w = frame_init.shape[1] // 2
    focal_px = (half_w / 2) / math.tan(math.radians(FOV_DEG / 2))

    # --- Цонхуудыг тодорхой хэмжээтэйгээр нээнэ ---
    # (өмнөх run-аас хадгалагдсан хэмжээг дарж шинэчилнэ)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 1280, 720)
    cv2.moveWindow("Camera", 30, 30)

    cv2.namedWindow("2D Grid Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("2D Grid Map", MAP_WINDOW_W, MAP_WINDOW_H)
    cv2.moveWindow("2D Grid Map", 30 + 1280 + 20, 30)

    stereo_left, stereo_right, wls = create_stereo()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    disp_history = []
    distance_by_id = {}

    # --- VIDEO WRITERS (камер + map) ---
    cam_writer = None
    cam_path = None
    map_writer = None
    map_path = None
    if RECORD_VIDEO:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        cam_path = f"hexapod_cam_{ts}.mp4"
        out_w = 2 * half_w
        out_h = frame_init.shape[0]
        cam_writer = cv2.VideoWriter(cam_path, fourcc, RECORD_FPS, (out_w, out_h))
        if not cam_writer.isOpened():
            print(f"[REC] camera writer нээгдсэнгүй: {cam_path}")
            cam_writer = None
        else:
            print(f"[REC] camera бичлэг → {cam_path}")

        map_path = f"hexapod_map_{ts}.mp4"
        map_writer = cv2.VideoWriter(map_path, fourcc, RECORD_FPS,
                                     (MAP_WINDOW_W, MAP_WINDOW_H))
        if not map_writer.isOpened():
            print(f"[REC] map writer нээгдсэнгүй: {map_path}")
            map_writer = None
        else:
            print(f"[REC] map бичлэг    → {map_path}")

    # Session timer-г эхлүүлнэ (UI-д харуулах)
    global session_start_time
    session_start_time = time.time()

    print("=" * 55)
    print("HEXAPOD LAWNMOWER + BOTTLE TRACKING + 2D GRID MAP")
    print(f"  Resolution:   {half_w}x{frame_init.shape[0]}")
    print(f"  Focal length: {focal_px:.1f} px")
    print("  ESC = гарах")
    print("=" * 55)

    settle_left  = 0
    prev_skip    = False
    motion_buf   = {}
    class_by_id  = {}

    try:
        while not stop_event.is_set():
            ret, frame = cam.read()
            if not ret: continue
            frame = cv2.flip(frame, -1)

            moving_now = is_moving()
            if moving_now:
                settle_left = MOTION_SETTLE_FRAMES
            elif settle_left > 0:
                settle_left -= 1
            skip_detect  = DEFERRED_COMMIT_WHILE_MOVING and (moving_now or settle_left > 0)
            just_stopped = prev_skip and not skip_detect
            prev_skip    = skip_detect

            # Камер 180° эргүүлсний улмаас зүүн/баруун камерын хагас нь
            # байршлаа сольсон — тиймээс стерео болон YOLO-д зориулж
            # зүүн/баруун утгыг солих шаардлагатай.
            w = frame.shape[1]
            left = frame[:, w // 2:]
            right = frame[:, :w // 2]

            # --- Stereo ---
            gl_full = clahe.apply(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
            gr_full = clahe.apply(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
            gl = cv2.resize(gl_full, None, fx=1/STEREO_SCALE, fy=1/STEREO_SCALE)
            gr = cv2.resize(gr_full, None, fx=1/STEREO_SCALE, fy=1/STEREO_SCALE)

            dl = stereo_left.compute(gl, gr)
            dr = stereo_right.compute(gr, gl)
            disp_small = wls.filter(dl, gl, None, dr).astype(np.float32) / 16.0
            disp_raw = cv2.resize(disp_small, (left.shape[1], left.shape[0])) * STEREO_SCALE

            disp_history.append(disp_raw)
            if len(disp_history) > DISP_HISTORY_SIZE:
                disp_history.pop(0)
            disparity = np.mean(disp_history, axis=0)

            # --- YOLO ---
            left_draw = left.copy()
            lh, lw = left.shape[:2]
            img_cx = lw // 2

            results = yolo.track(left, verbose=False, conf=YOLO_CONF,
                                 persist=True, tracker="botsort.yaml", imgsz=480)

            active_ids = set()
            rx, ry, rth = get_pose()

            for r in results:
                if r.boxes is None or r.boxes.id is None: continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in BOTTLE_CLASSES: continue
                    if box.id is None: continue

                    track_id = int(box.id[0])
                    active_ids.add(track_id)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(lw, x2), min(lh, y2)
                    cls_name = BOTTLE_CLASSES[cls_id]

                    raw_d = roi_distance(disparity, focal_px, y1, y2, x1, x2)
                    dist_label = ""
                    if raw_d is not None:
                        prev = distance_by_id.get(track_id)
                        smoothed = (raw_d if prev is None
                                    else prev * (1 - EMA_ALPHA) + raw_d * EMA_ALPHA)
                        distance_by_id[track_id] = smoothed
                        dist_label = f" {smoothed:.2f}m"

                        cx_rel = (x1 + x2) // 2 - img_cx
                        class_by_id[track_id] = cls_name
                        if skip_detect:
                            buf = motion_buf.setdefault(track_id, [])
                            buf.append((smoothed, cx_rel, rx, ry, rth))
                            if len(buf) > MOTION_BUFFER_MAX: buf.pop(0)
                        else:
                            wx, wy = pixel_to_world(cx_rel, smoothed, focal_px, rx, ry, rth)
                            world_map.add_detection(wx, wy, cls_name, track_id)

                    color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
                    label = f"ID:{track_id} {cls_name}{dist_label}"
                    cv2.rectangle(left_draw, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(left_draw, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                    cv2.putText(left_draw, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if len(distance_by_id) > 50:
                for k in [k for k in distance_by_id if k not in active_ids]:
                    del distance_by_id[k]

            if just_stopped and motion_buf:
                for tid, samples in motion_buf.items():
                    cls_name_b = class_by_id.get(tid)
                    if not samples or cls_name_b is None: continue
                    wxs, wys = [], []
                    for (sm, cxr, rx_s, ry_s, rth_s) in samples:
                        wx, wy = pixel_to_world(cxr, sm, focal_px, rx_s, ry_s, rth_s)
                        wxs.append(wx); wys.append(wy)
                    wx_med = float(np.median(wxs)); wy_med = float(np.median(wys))
                    kx = [x for x,y in zip(wxs,wys) if math.hypot(x-wx_med,y-wy_med)<0.5]
                    ky = [y for x,y in zip(wxs,wys) if math.hypot(x-wx_med,y-wy_med)<0.5]
                    if kx: wx_med, wy_med = float(np.median(kx)), float(np.median(ky))
                    world_map.add_detection(wx_med, wy_med, cls_name_b, tid)
                motion_buf.clear()

            if skip_detect:
                cv2.putText(left_draw, "MOVING  (detection paused)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)

            combined = np.hstack([left_draw, right])

            if cam_writer is not None:
                cam_writer.write(combined)

            scale = min(1280 / combined.shape[1], 720 / combined.shape[0])
            cv2.imshow("Camera", cv2.resize(combined, None, fx=scale, fy=scale))

            map_img = render_map()
            if map_writer is not None and map_img is not None:
                map_writer.write(map_img)

            if cv2.waitKey(1) == 27:
                stop_event.set(); break

    finally:
        if cam_writer is not None:
            cam_writer.release()
            print(f"[REC] camera бичлэг хадгалагдлаа: {cam_path}")
        if map_writer is not None:
            map_writer.release()
            print(f"[REC] map бичлэг хадгалагдлаа: {map_path}")
        cam.release()
        cv2.destroyAllWindows()
        print("Камер хаагдлаа.")

        print("\n===== FINAL MAP =====")
        for i, (lx, ly, cls, count) in enumerate(world_map.snapshot()):
            print(f"  #{i+1} {cls}: ({lx:.2f}m, {ly:.2f}m)  seen x{count}")


# =============================================================
# АЖИЛЛУУЛАХ
# =============================================================
def main():
    print("Grid-ийн хэмжээсийг оруул (мөр багана).")
    print("Жишээ: 4 6   → 4 мөр × 6 багана")
    user_input = input("> ").strip().replace(",", " ").replace("[", "").replace("]", "")
    parts = user_input.split()
    if len(parts) != 2:
        print("Буруу форматтай. '4 6' гэж оруул."); return
    rows, cols = int(parts[0]), int(parts[1])
    if rows < 1 or cols < 1:
        print("Хэмжээ 1-ээс их байх ёстой."); return

    grid_config["rows"] = rows
    grid_config["cols"] = cols
    print(f"→ Grid: {rows} мөр × {cols} багана,  нийт {rows*cols} нүд "
          f"({rows*CELL_SIZE:.2f}m × {cols*CELL_SIZE:.2f}m)")

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    try:
        mover = threading.Thread(target=movement_thread, args=(ser, rows, cols), daemon=True)
        mover.start()

        camera_loop()

        stop_event.set()
        mover.join(timeout=5.0)

    finally:
        ser.close()
        print("Serial хаагдлаа.")


if __name__ == "__main__":
    main()
