"""
HEXAPOD — Stereo Camera Hand Wave Detection
============================================
Камер хүний гар wave-г илрүүлнэ → Робот R1 хөлөөр hand wave хийнэ.
Q / ESC = гарах
"""

import serial
import time
import math
import threading
import cv2
import numpy as np
from collections import deque
import mediapipe as mp

# =============================================================
# SERIAL / ROBOT ТОХИРГОО
# =============================================================
PORT = "COM4"
BAUD = 115200
CMD_SET = 0x53 | 0x80

# =============================================================
# SERVO PIN
# =============================================================
PIN = {
    "R31": 0,  "R32": 1,  "R33": 2,
    "L31": 3,  "L32": 4,  "L33": 5,
    "R21": 6,  "R22": 7,  "R23": 8,
    "L21": 9,  "L22": 10, "L23": 11,
    "R11": 12, "R12": 13, "R13": 14,
    "L11": 15, "L12": 16, "L13": 17,
}

# =============================================================
# ЗОГСОЛТЫН БАЙРЛАЛ
# =============================================================
STAND = {
    "L11": 1525, "L12": 1500, "L13": 1470,
    "L21": 1525, "L22": 1530, "L23": 1430,
    "L31": 1540, "L32": 1480, "L33": 1525,
    "R11": 1500, "R12": 1760, "R13": 1470,
    "R21": 1525, "R22": 1480, "R23": 1550,
    "R31": 1450, "R32": 1500, "R33": 1500,
}

# =============================================================
# АЛХАЛТЫН ПАРАМЕТРҮҮД (hand wave-д хэрэглэгдэнэ)
# =============================================================
FEMUR_LIFT = 200
TIBIA_LIFT = 150

TRANSITION_STEPS = 15
TRANSITION_DELAY = 0.03

# =============================================================
# HAND WAVE ПАРАМЕТРҮҮД (R1 хөл)
# =============================================================
WAVE_COXA_SWING = 333    # R11: ±pulse (60° орчим)
WAVE_FEMUR_LIFT = 400    # R12: дээш өргөх pulse
WAVE_TIBIA_LIFT = 400    # R13: шуу дэлгэх pulse
WAVE_STEPS      = 20     # Нэг хөдөлгөөний алхам
WAVE_DELAY      = 0.025  # Алхам хоорондын хугацаа
WAVE_REPEATS    = 2      # Баруун-зүүн давталт

# =============================================================
# КАМЕРЫН ТОХИРГОО
# =============================================================
CAMERA_INDEX   = 1
FRAME_WIDTH    = 2560
FRAME_HEIGHT   = 960
FOV_DEG        = 96
BASELINE_M     = 0.065

# Wave илрүүлэх босго
WAVE_MOVE_THRESHOLD       = 80   # pixel — гарын хөдөлгөөний өргөн
WAVE_DIR_CHANGE_THRESHOLD = 2    # чиглэл өөрчлөлтийн тоо
WAVE_HISTORY_LEN          = 15   # хадгалах frame тоо
WAVE_COOLDOWN_SEC         = 3.0  # wave илрүүлсэний дараах хүлээлтийн хугацаа

# =============================================================
# SERIAL ТУСЛАХ ФУНКЦҮҮД
# =============================================================
def encode_set(start, values):
    data = bytearray([CMD_SET, start & 0x7F, len(values) & 0x7F])
    for v in values:
        data.append(v & 0x7F)
        data.append((v >> 7) & 0x7F)
    return data

def send_set(ser, start, values, label=""):
    pkt = encode_set(start, values)
    if label:
        print(label)
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

def interpolate(pose_a, pose_b, alpha):
    return {k: int(round(pose_a[k] + (pose_b[k] - pose_a[k]) * alpha))
            for k in pose_a}

def move_smooth(ser, pose_from, pose_to, steps=15, delay=0.03):
    for i in range(1, steps + 1):
        alpha = ease_in_out(i / steps)
        send_pose(ser, interpolate(pose_from, pose_to, alpha))
        time.sleep(delay)
    return pose_to

def stand_pose():
    return dict(STAND)

def femur_offset(leg, offset):
    s = leg + "2"
    return STAND[s] - offset if leg.startswith("L") else STAND[s] + offset

def tibia_offset(leg, offset):
    s = leg + "3"
    return STAND[s] + offset if leg.startswith("L") else STAND[s] - offset

# =============================================================
# HAND WAVE — R1 хөл
# =============================================================
def hand_wave(ser, current):
    """R1 хөлөөр hand wave хийнэ, дуусаад STAND руу буцна."""
    print("🤖 ROBOT HAND WAVE!")

    stand = stand_pose()

    # 1. Гарыг өргөнө
    pose_up = dict(current)
    pose_up["R12"] = STAND["R12"] + WAVE_FEMUR_LIFT
    pose_up["R13"] = STAND["R13"] - WAVE_TIBIA_LIFT
    pose_up["R11"] = STAND["R11"]
    current = move_smooth(ser, current, pose_up, WAVE_STEPS, WAVE_DELAY)

    # 2. Wave: баруун ↔ зүүн WAVE_REPEATS удаа
    pose_right = dict(pose_up)
    pose_right["R11"] = STAND["R11"] + WAVE_COXA_SWING

    pose_left = dict(pose_up)
    pose_left["R11"] = STAND["R11"] - WAVE_COXA_SWING

    for _ in range(WAVE_REPEATS):
        current = move_smooth(ser, current, pose_right, WAVE_STEPS, WAVE_DELAY)
        current = move_smooth(ser, current, pose_left,  WAVE_STEPS, WAVE_DELAY)

    # Coxa дунд байрлалд буцаана
    pose_center = dict(pose_up)
    pose_center["R11"] = STAND["R11"]
    current = move_smooth(ser, current, pose_center, WAVE_STEPS, WAVE_DELAY)

    # 3. Гарыг буулгана
    current = move_smooth(ser, current, stand, WAVE_STEPS, WAVE_DELAY)

    print("✓ Wave дуусав")
    return current

# =============================================================
# RETURN TO STAND
# =============================================================
def return_to_stand(ser, current):
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

    current = lift_and_place(current, ["L1", "R2", "L3"])
    current = lift_and_place(current, ["R1", "L2", "R3"])
    return current

# =============================================================
# STEREO КАМЕР — WAVE ИЛРҮҮЛЭХ
# =============================================================
def get_distance(disparity, x, y):
    h, w = disparity.shape
    x1, y1 = max(0, x - 10), max(0, y - 10)
    x2, y2 = min(w, x + 10), min(h, y + 10)
    roi   = disparity[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if len(valid) < 5:
        return None
    med = np.median(valid)
    if med <= 0:
        return None
    # focal length камерын эхлэлд тооцоолно (глобал хувьсагч)
    dist = (FOCAL_PX * BASELINE_M) / med
    return dist if 0.1 < dist < 10 else None

def detect_wave(history):
    if len(history) < 10:
        return False
    xs = [p[0] for p in history]
    movement = max(xs) - min(xs)
    direction_changes = 0
    for i in range(1, len(xs) - 1):
        if (xs[i] - xs[i-1]) * (xs[i+1] - xs[i]) < 0:
            direction_changes += 1
    return movement > WAVE_MOVE_THRESHOLD and direction_changes >= WAVE_DIR_CHANGE_THRESHOLD

# =============================================================
# РОБОТ THREAD — wave event хүлээж, hand_wave дуудна
# =============================================================
wave_event   = threading.Event()   # камер → robot thread
stop_event   = threading.Event()   # гарах дохио

def robot_thread_func(ser):
    current = stand_pose()
    send_pose(ser, current)
    time.sleep(1.0)
    print("✓ Робот бэлэн. Гараа wave хийнэ үү!")

    while not stop_event.is_set():
        # wave_event-г хүлээнэ (0.1 секунд timeout-тай)
        triggered = wave_event.wait(timeout=0.1)
        if triggered:
            wave_event.clear()
            if not stop_event.is_set():
                current = hand_wave(ser, current)

    # Гарахдаа STAND руу буцна
    print("Зогсолтод буцаж байна...")
    return_to_stand(ser, current)

# =============================================================
# КАМЕР THREAD — wave илрүүлж, event set хийнэ
# =============================================================
def camera_thread_func():
    global FOCAL_PX

    cam = cv2.VideoCapture(CAMERA_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    ret, frame_init = cam.read()
    if not ret:
        print("Камер нээгдсэнгүй!")
        stop_event.set()
        return

    half_w   = frame_init.shape[1] // 2
    FOCAL_PX = (half_w / 2) / math.tan(math.radians(FOV_DEG / 2))

    # Stereo matcher
    ws     = 5
    stereo = cv2.StereoSGBM.create(
        minDisparity=0, numDisparities=128, blockSize=ws,
        P1=8 * 3 * ws**2, P2=32 * 3 * ws**2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2
    )

    # MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands    = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    )

    hand_positions = deque(maxlen=WAVE_HISTORY_LEN)
    last_wave_time = 0.0   # cooldown хяналт

    print("=" * 50)
    print("КАМЕР АЖИЛЛАЖ БАЙНА")
    print("Гараа wave хийхэд робот хариу өгнө")
    print("ESC = гарах")
    print("=" * 50)

    while not stop_event.is_set():
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, -1)
        h, w, _ = frame.shape

        left  = frame[:, :w // 2]
        right = frame[:, w // 2:]

        # Stereo disparity
        grayL     = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
        grayR     = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

        # Hand detection
        rgb     = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        wave_detected_now = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(left, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cx = int(hand_landmarks.landmark[9].x * left.shape[1])
                cy = int(hand_landmarks.landmark[9].y * left.shape[0])
                hand_positions.append((cx, cy))

                dist = get_distance(disparity, cx, cy)
                if dist:
                    cv2.putText(left, f"{dist:.2f}m", (cx, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if detect_wave(hand_positions):
                    wave_detected_now = True

        # Wave илрүүлсэн + cooldown өнгөрсөн + робот wave хийхгүй байна
        now = time.time()
        if wave_detected_now and (now - last_wave_time) > WAVE_COOLDOWN_SEC:
            if not wave_event.is_set():   # давхар trigger хийхгүй
                print("👋 Wave илрүүлэв! Роботыг идэвхжүүлж байна...")
                hand_positions.clear()     # history цэвэрлэнэ
                last_wave_time = now
                wave_event.set()

            cv2.putText(left, "WAVE!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)

        # Дэлгэц
        combined = np.hstack([left, right])
        scale    = min(1280 / combined.shape[1], 720 / combined.shape[0])
        display  = cv2.resize(combined, None, fx=scale, fy=scale)

        # Cooldown үлдсэн хугацааг харуулна
        remaining = WAVE_COOLDOWN_SEC - (time.time() - last_wave_time)
        if 0 < remaining < WAVE_COOLDOWN_SEC:
            cv2.putText(display, f"Cooldown: {remaining:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

        cv2.imshow("Wave Detection", display)

        key = cv2.waitKey(1)
        if key == 27:   # ESC
            stop_event.set()
            break

    cam.release()
    cv2.destroyAllWindows()
    stop_event.set()   # robot thread-г ч зогсооно

# =============================================================
# АЖИЛЛУУЛАХ
# =============================================================
if __name__ == "__main__":
    # Focal length глобал (camera thread тооцоолно)
    FOCAL_PX = 0.0

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    try:
        send_set(ser, 26, [1], "RELAY ON")
        time.sleep(1.0)

        # Камер thread — wave илрүүлнэ
        cam_thread = threading.Thread(target=camera_thread_func, daemon=True)
        cam_thread.start()

        # Робот thread — wave event хүлээж, hand_wave хийнэ
        # (main thread дээр ажиллуулна, serial thread-safe байхын тулд)
        robot_thread_func(ser)

    finally:
        stop_event.set()
        send_set(ser, 26, [0], "RELAY OFF")
        time.sleep(0.5)
        ser.close()
        print("Гарлаа.")