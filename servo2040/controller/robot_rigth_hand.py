import serial
import time
import math
import keyboard   # pip install keyboard

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
# TRIPOD БҮЛГҮҮД
# =============================================================
GROUP_A = ["L1", "R2", "L3"]
GROUP_B = ["R1", "L2", "R3"]

# =============================================================
# АЛХАЛТЫН ПАРАМЕТРҮҮД
# =============================================================
COXA_SWING = 150
FEMUR_LIFT = 200
TIBIA_LIFT = 150

GROUND_PRESS = 80

TRANSITION_STEPS = 15
TRANSITION_DELAY = 0.03

# =============================================================
# HAND WAVE ПАРАМЕТРҮҮД (R1 хөл)
# =============================================================
# 30 градус ≈ 333 pulse (стандарт servo: 1000–2000 us = 180°, 1°≈5.56 pulse)
WAVE_COXA_SWING  = 333   # R11: баруун-зүүн хазайлт (±333 pulse)
WAVE_FEMUR_LIFT  = 400   # R12: гар өргөх хэмжээ (STAND-аас хасна, учир нь R тал)
WAVE_TIBIA_LIFT  = 450   # R13: шуу дэлгэх хэмжээ
WAVE_STEPS       = 20    # Нэг хөдөлгөөний алхам
WAVE_DELAY       = 0.025 # Алхам хоорондын хугацаа (секунд)
WAVE_REPEATS     = 2     # Баруун-зүүн сэлгэлтийн тоо (2 = 2 бүрэн давталт)

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


# =============================================================
# HAND WAVE — R1 хөл (баруун урд гар)
# =============================================================
def hand_wave(ser, current):
    """
    R1 хөлөөр hand wave хийнэ:
      1. R12 (femur) + R13 (tibia) өргөж гарыг дээш гаргана
      2. R11 (coxa) баруун-зүүн 2 удаа сэлгэнэ (wave)
      3. Анхны STAND байрлалд буцна

    R сервогийн чиглэл:
      - R12 дээшлэх  → STAND["R12"] - WAVE_FEMUR_LIFT  (femur_offset R: STAND+offset → буруу, тиймд шууд хасна)
      - R13 дэлгэх   → STAND["R13"] + WAVE_TIBIA_LIFT  (tibia_offset R: STAND-offset → буруу, тиймд шууд нэмнэ)
      - R11 баруун   → STAND["R11"] + WAVE_COXA_SWING  (coxa_offset R: STAND-offset → буруу, баруун = нэмнэ)
      - R11 зүүн     → STAND["R11"] - WAVE_COXA_SWING

    Тайлбар: coxa_offset("R1", +N) = STAND["R11"] - N  (робот коорд)
             Гэхдээ бид энд SERVO чиглэлийг шууд тодорхойлно.
             R servo-д pulse ихсэх = CW эргэлт = баруун тийш хазайлт гэж үзнэ.
    """
    print("→ HAND WAVE (R1)")

    stand = stand_pose()

    # --- Алхам 1: Гарыг өргөнө (femur + tibia өөрчилнө, coxa хэвээр) ---
    # R12: дээшлэхийн тулд pulse-г багасгана (R femur: их pulse = газрын чиглэл)
    # R13: дэлгэхийн тулд pulse-г ихэсгэнэ
    pose_up = dict(current)
    pose_up["R12"] = STAND["R12"] + WAVE_FEMUR_LIFT   # гарыг дээш өргөнө
    pose_up["R13"] = STAND["R13"] - WAVE_TIBIA_LIFT   # шуу дэлгэнэ
    pose_up["R11"] = STAND["R11"]                       # coxa хэвээр

    current = move_smooth(ser, current, pose_up, WAVE_STEPS, WAVE_DELAY)

    # --- Алхам 2: WAVE — баруун-зүүн 2 удаа ---
    # Нэг давталт = баруун → зүүн → (дараагийн давталт эхлэлийн байрлал)
    # Эцсийн давталтад дунд байрлалд (STAND R11) зогсоно
    pose_right = dict(pose_up)
    pose_right["R11"] = STAND["R11"] + WAVE_COXA_SWING  # баруун

    pose_left = dict(pose_up)
    pose_left["R11"] = STAND["R11"] - WAVE_COXA_SWING   # зүүн

    for i in range(WAVE_REPEATS):
        # Баруун тийш
        current = move_smooth(ser, current, pose_right, WAVE_STEPS, WAVE_DELAY)
        # Зүүн тийш
        current = move_smooth(ser, current, pose_left, WAVE_STEPS, WAVE_DELAY)

    # Coxa-г дунд байрлалд (STAND) буцаана — wave дуусав
    pose_center = dict(pose_up)
    pose_center["R11"] = STAND["R11"]
    current = move_smooth(ser, current, pose_center, WAVE_STEPS, WAVE_DELAY)

    # --- Алхам 3: Гарыг буулгана (STAND руу буцна) ---
    current = move_smooth(ser, current, stand, WAVE_STEPS, WAVE_DELAY)

    print("→ WAVE ДУУСАВ")
    return current


# =============================================================
# POSE ҮҮСГЭХ — УРАГШ/ХОЙШ (forward-backward gait)
# =============================================================
def press_down(pose, leg):
    """Газар дээрх хөлийг STAND-аас илүү дооно түлхэнэ (биеийн суусыг нөхнө)."""
    pose[leg + "2"] = femur_offset(leg, -GROUND_PRESS)
    pose[leg + "3"] = tibia_offset(leg, -GROUND_PRESS)

def make_step_pose_walk(swing_group, push_group, direction):
    pose = stand_pose()
    for leg in swing_group:
        pose[leg + "2"] = femur_offset(leg, FEMUR_LIFT)
        pose[leg + "3"] = tibia_offset(leg, TIBIA_LIFT)
        pose[leg + "1"] = coxa_offset(leg, direction * COXA_SWING)
    for leg in push_group:
        pose[leg + "1"] = coxa_offset(leg, direction * (-COXA_SWING))
        press_down(pose, leg)
    return pose

def make_down_pose_walk(down_group, push_group, direction):
    pose = stand_pose()
    for leg in down_group:
        pose[leg + "1"] = coxa_offset(leg, direction * COXA_SWING)
        press_down(pose, leg)
    for leg in push_group:
        pose[leg + "1"] = coxa_offset(leg, direction * (-COXA_SWING))
        press_down(pose, leg)
    return pose


# =============================================================
# POSE ҮҮСГЭХ — ЭРГЭЛТ (turning gait, in place)
# =============================================================
def turn_sign(leg, turn_dir):
    return turn_dir if leg.startswith("L") else -turn_dir

def make_step_pose_turn(swing_group, push_group, turn_dir):
    pose = stand_pose()
    for leg in swing_group:
        pose[leg + "2"] = femur_offset(leg, FEMUR_LIFT)
        pose[leg + "3"] = tibia_offset(leg, TIBIA_LIFT)
        pose[leg + "1"] = coxa_offset(leg, turn_sign(leg, turn_dir) * COXA_SWING)
    for leg in push_group:
        pose[leg + "1"] = coxa_offset(leg, turn_sign(leg, turn_dir) * (-COXA_SWING))
        press_down(pose, leg)
    return pose

def make_down_pose_turn(down_group, push_group, turn_dir):
    pose = stand_pose()
    for leg in down_group:
        pose[leg + "1"] = coxa_offset(leg, turn_sign(leg, turn_dir) * COXA_SWING)
        press_down(pose, leg)
    for leg in push_group:
        pose[leg + "1"] = coxa_offset(leg, turn_sign(leg, turn_dir) * (-COXA_SWING))
        press_down(pose, leg)
    return pose


# =============================================================
# НЭГ БҮРЭН A+B МӨЧЛӨГ ГҮЙЦЭТГЭХ
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
    return do_cycle(ser, current, make_step_pose_walk, make_down_pose_walk, +1)

def walk_backward_cycle(ser, current):
    return do_cycle(ser, current, make_step_pose_walk, make_down_pose_walk, -1)

def turn_right_cycle(ser, current):
    return do_cycle(ser, current, make_step_pose_turn, make_down_pose_turn, +1)

def turn_left_cycle(ser, current):
    return do_cycle(ser, current, make_step_pose_turn, make_down_pose_turn, -1)


# =============================================================
# ТОВЧЛУУРААР УДИРДАХ ЦИКЛ
# =============================================================
# W = урагш,  S = хойш
# A = зүүн эргэх,  D = баруун эргэх
# R = hand wave (R1 хөл)
# Q эсвэл ESC = гарах

def controller_loop(ser):
    current = stand_pose()
    send_pose(ser, current)
    time.sleep(1.0)

    print("=" * 50)
    print("HEXAPOD CONTROLLER")
    print("  W = урагш   S = хойш")
    print("  A = зүүн    D = баруун")
    print("  R = hand wave (R1 хөл)")
    print("  Q / ESC = гарах")
    print("=" * 50)

    last_command = None
    wave_pressed = False   # Edge-trigger: нэг дарахад нэг wave хийнэ

    while True:
        # Гарах
        if keyboard.is_pressed("q") or keyboard.is_pressed("esc"):
            break

        # Товчлуурын төлөв
        w = keyboard.is_pressed("w")
        s = keyboard.is_pressed("s")
        a = keyboard.is_pressed("a")
        d = keyboard.is_pressed("d")
        r = keyboard.is_pressed("r")

        # R — hand wave (edge trigger: дарагдаж эхэлсэн мөчид л нэг удаа ажиллана)
        if r:
            if not wave_pressed:
                wave_pressed = True
                last_command = "R"
                current = hand_wave(ser, current)
                # Wave дуусаад last_command-г цэвэрлэнэ (зогсолтод буцахгүй, аль хэдийн STAND)
                last_command = None
        else:
            wave_pressed = False

        # Хөдөлгөөний товчлуурууд (R дарагдаагүй үед л шалгана)
        if not r:
            if w:
                if last_command != "W":
                    print("→ УРАГШ")
                    last_command = "W"
                current = walk_forward_cycle(ser, current)
            elif s:
                if last_command != "S":
                    print("→ ХОЙШ")
                    last_command = "S"
                current = walk_backward_cycle(ser, current)
            elif a:
                if last_command != "A":
                    print("→ ЗҮҮН эргэх")
                    last_command = "A"
                current = turn_left_cycle(ser, current)
            elif d:
                if last_command != "D":
                    print("→ БАРУУН эргэх")
                    last_command = "D"
                current = turn_right_cycle(ser, current)
            else:
                if last_command is not None:
                    print("→ ЗОГСОЛТ")
                    last_command = None
                    current = return_to_stand(ser, current)
                time.sleep(0.05)

    print("Зогсолтод буцаж байна...")
    return_to_stand(ser, current)


# =============================================================
# АЖИЛЛУУЛАХ
# =============================================================
if __name__ == "__main__":
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)

    try:
        send_set(ser, 26, [1], "RELAY ON")
        time.sleep(1.0)

        controller_loop(ser)

    finally:
        send_set(ser, 26, [0], "RELAY OFF")
        time.sleep(0.5)
        ser.close()
        print("Гарлаа.")