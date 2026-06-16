"""
practice.py
===========
Хексапод WASD хяналт + камерын стерео бичлэг.
YOLO / stereo processing / map огт байхгүй → хурдан, өндөр FPS.

Гаралт:
  practice_cam_YYYYMMDD_HHMMSS.mp4   — стерео видео бичлэг (зүүн+баруун хагас)
  practice_pose_YYYYMMDD_HHMMSS.csv  — frame тус бүрийн timestamp + robot pose
                                       (analyze.py-д уншиж map хийхэд хэрэгтэй)

Хожим analyze.py:
  1. CSV + MP4 уншина
  2. Зүүн хагасаас модыг таних (YOLO / өнгөний segmentation)
  3. Баруун хагасаар stereo depth тооцоолно
  4. Тухайн frame-ийн pose-оор world coordinate-д хөрвүүлнэ
  5. Map зурна
"""

import serial
import time
import math
import csv
import threading
import datetime
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import keyboard

# =============================================================
# ТОХИРГОО
# =============================================================
PORT         = "COM4"
BAUD         = 115200
CMD_SET      = 0x53 | 0x80

CAMERA_INDEX = 1
# Камерын нийт өргөн (хоёр нүд). 1280→нэг нүд 640×480
CAM_W        = 1280
CAM_H        = 480

# Бичлэгийн FPS-г бодитоор хэмжинэ — real-time playback хийхэд хэрэгтэй
FPS_BENCH_FRAMES = 30

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
COXA_SWING       = 150
FEMUR_LIFT       = 200
TIBIA_LIFT       = 150
GROUND_PRESS     = 80
TRANSITION_STEPS = 15
TRANSITION_DELAY = 0.03
STEP_FORWARD_M   = 0.04
TURN_ANGLE_RAD   = math.radians(30)

# =============================================================
# ХУВААЛЦСАН БАЙДАЛ
# =============================================================
stop_event = threading.Event()

robot_pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
pose_lock  = threading.Lock()

session_start = None


def get_pose():
    with pose_lock:
        return robot_pose["x"], robot_pose["y"], robot_pose["theta"]

def advance_pose(dx=0.0, dy=0.0, dtheta=0.0):
    with pose_lock:
        th = robot_pose["theta"]
        robot_pose["x"]     += dx * math.cos(th) - dy * math.sin(th)
        robot_pose["y"]     += dx * math.sin(th) + dy * math.cos(th)
        robot_pose["theta"] += dtheta


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

# --- Walk ---
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

# --- Turn ---
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
# KEYBOARD THREAD
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
    print("PRACTICE RECORDER  (WASD + стерео бичлэг)")
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
                time.sleep(0.05)
    finally:
        return_to_stand(ser, cur)
        send_set(ser, 26, [0], "RELAY OFF")
        time.sleep(0.5)
        print("[ROBOT] thread дууслаа.")


# =============================================================
# КАМЕРЫН ЦИКЛ  (main thread)
# — YOLO/stereo/map огт байхгүй, зөвхөн бичлэг + pose CSV
# =============================================================
def camera_loop():
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    ret, frame0 = cam.read()
    if not ret:
        print("Камер нээгдсэнгүй!"); stop_event.set(); return
    frame0 = cv2.flip(frame0, -1)
    fh, fw = frame0.shape[:2]

    cv2.namedWindow("Practice Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Practice Camera", fw, fh)
    cv2.moveWindow("Practice Camera", 30, 30)

    # --- Бодит FPS хэмжих ---
    print(f"[CAM] FPS хэмжиж байна ({FPS_BENCH_FRAMES} frame)...")
    t0 = time.time()
    for _ in range(FPS_BENCH_FRAMES):
        ok, f = cam.read()
        if ok: cv2.flip(f, -1)
    fps = FPS_BENCH_FRAMES / max(time.time() - t0, 0.1)
    fps = max(1.0, round(fps, 1))
    print(f"[CAM] бодит FPS: {fps:.1f}")

    # --- Файлын нэрс ---
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    vid_path  = f"practice_cam_{ts}.mp4"
    pose_path = f"practice_pose_{ts}.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(vid_path, fourcc, fps, (fw, fh))
    if writer.isOpened():
        print(f"[REC] video  → {vid_path}")
    else:
        print("[REC] VideoWriter нээгдсэнгүй!"); writer = None

    # CSV: frame_idx, timestamp_s, x_m, y_m, theta_rad
    pose_file = open(pose_path, "w", newline="")
    csv_writer = csv.writer(pose_file)
    csv_writer.writerow(["frame", "time_s", "x_m", "y_m", "theta_rad"])
    print(f"[REC] pose   → {pose_path}")

    frame_idx = 0
    t_start   = time.time()

    print("[CAM] Бичлэг эхэллээ. Q/ESC → гарах.")

    try:
        while not stop_event.is_set():
            ret, frame = cam.read()
            if not ret: continue
            frame = cv2.flip(frame, -1)

            rx, ry, rth = get_pose()
            t_now = time.time() - t_start

            # CSV-д pose хадгална
            csv_writer.writerow([frame_idx, f"{t_now:.4f}",
                                  f"{rx:.4f}", f"{ry:.4f}", f"{rth:.6f}"])
            frame_idx += 1

            # Дэлгэцэнд overlay нэмнэ (бичлэгт ч орно)
            disp = frame.copy()
            mm, ss = divmod(int(t_now), 60)
            cv2.rectangle(disp, (0, 0), (360, 70), (0, 0, 0), -1)
            cv2.putText(disp, f"REC  {mm:02d}:{ss:02d}  F:{frame_idx}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
            cv2.putText(disp, f"X:{rx:+.2f}m  Y:{ry:+.2f}m  Th:{math.degrees(rth)%360:.0f}deg",
                        (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1)

            if writer:
                writer.write(disp)
            cv2.imshow("Practice Camera", disp)

            if cv2.waitKey(1) == 27:
                stop_event.set(); break

    finally:
        pose_file.flush(); pose_file.close()
        if writer: writer.release()
        cam.release()
        cv2.destroyAllWindows()
        print(f"\n[REC] Дууслаа.  {frame_idx} frame,  {frame_idx/fps:.1f}s")
        print(f"      video → {vid_path}")
        print(f"      pose  → {pose_path}")
        print(f"      analyze.py-д эдгээр хоёр файлыг өгч map хийнэ.")


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
