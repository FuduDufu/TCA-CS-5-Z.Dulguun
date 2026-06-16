import serial
import time
import math

PORT = "COM4"   # <-- COM port-оо солино
BAUD = 115200

CMD_SET = 0x53 | 0x80  # 0xD3

def encode_set(start, values):
    data = bytearray([CMD_SET, start & 0x7F, len(values) & 0x7F])
    for v in values:
        data.append(v & 0x7F)
        data.append((v >> 7) & 0x7F)
    return data

def send_set(ser, start, values, label=""):
    pkt = encode_set(start, values)
    print(label, " ".join(f"{b:02X}" for b in pkt))
    ser.write(pkt)

def clamp(x, a, b):
    return max(a, min(b, x))

def deg2us(deg, neg45_us, pos45_us):
    center_us = (neg45_us + pos45_us) / 2.0
    us_per_deg = (pos45_us - neg45_us) / 90.0
    return int(round(center_us + deg * us_per_deg))

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
COXA_LEN = 43
FEMUR_LEN = 80
TIBIA_LEN = 134

LEG_CONNECTION_Z = -10
LEG_SITTING_Z = -40

COXA_ATTACH_ANGLE = -8
FEMUR_ATTACH_ANGLE = 35
TIBIA_ATTACH_ANGLE = 68

# MODE_BLOCK
LEG_RADIUS = 185
CORNER_LEG_ANGLE = 30
ELONGATION = 1.07
BODY_LIFT = -40

# z target
Z_TARGET = LEG_SITTING_Z - BODY_LIFT   # -40 - (-40) = 0

# --------------------------------------------------
# CALIBRATION
# --------------------------------------------------
CAL = {
    "L11": (1985, 1065),
    "L12": (1960, 1040),
    "L13": (1930, 1010),

    "L21": (1985, 1065),
    "L22": (1990, 1070),
    "L23": (1890, 970),

    "L31": (2000, 1080),
    "L32": (1940, 1020),
    "L33": (1985, 1065),

    "R11": (1960, 1040),
    "R12": (2220, 1300),
    "R13": (1930, 1010),

    "R21": (1985, 1065),
    "R22": (1940, 1020),
    "R23": (2010, 1090),

    "R31": (1910, 990),
    "R32": (1960, 1040),
    "R33": (1960, 1040),
}

PIN = {
    "R31": 0,  "R32": 1,  "R33": 2,
    "L31": 3,  "L32": 4,  "L33": 5,
    "R21": 6,  "R22": 7,  "R23": 8,
    "L21": 9,  "L22": 10, "L23": 11,
    "R11": 12, "R12": 13, "R13": 14,
    "L11": 15, "L12": 16, "L13": 17,
}

# --------------------------------------------------
# LEG TARGET ANGLES (body frame)
# --------------------------------------------------
# L1 front-left, L2 mid-left, L3 rear-left
# R1 front-right, R2 mid-right, R3 rear-right
#
# Hexapod body coordinate гэж үзээд:
#   front-left  = +30
#   mid-left    = +90
#   rear-left   = +150
#   front-right = -30
#   mid-right   = -90
#   rear-right  = -150
#
LEG_COXA_WORLD_ANGLE = {
    "L1":  30,
    "L2":   0,
    "L3": -30,
    "R1": -30,
    "R2":   0,
    "R3":  30,
}

# --------------------------------------------------
# IK for one leg
# --------------------------------------------------
def solve_leg(leg_name, coxa_servo, femur_servo, tibia_servo, invert_ft=False):
    r = LEG_RADIUS * ELONGATION
    ang = math.radians(LEG_COXA_WORLD_ANGLE[leg_name])

    x = r * math.cos(ang)
    y = r * math.sin(ang)
    z = Z_TARGET

    # world coxa angle
    coxa_world_deg = math.degrees(math.atan2(y, x))

    # planar IK after coxa length
    horizontal = math.sqrt(x * x + y * y)
    px = horizontal - COXA_LEN
    pz = z - LEG_CONNECTION_Z

    d = math.sqrt(px * px + pz * pz)
    d = clamp(d, 1.0, FEMUR_LEN + TIBIA_LEN - 1e-6)

    # knee
    cos_knee = clamp(
        (FEMUR_LEN**2 + TIBIA_LEN**2 - d**2) / (2 * FEMUR_LEN * TIBIA_LEN),
        -1.0, 1.0
    )
    knee_inner_deg = math.degrees(math.acos(cos_knee))

    # femur
    cos_femur = clamp(
        (FEMUR_LEN**2 + d**2 - TIBIA_LEN**2) / (2 * FEMUR_LEN * d),
        -1.0, 1.0
    )
    femur_part_deg = math.degrees(math.acos(cos_femur))
    line_deg = math.degrees(math.atan2(pz, px))
    femur_world_deg = line_deg + femur_part_deg

    # servo angles
    coxa_servo_deg = coxa_world_deg - COXA_ATTACH_ANGLE
    femur_servo_deg = femur_world_deg - FEMUR_ATTACH_ANGLE
    tibia_servo_deg = (180 - knee_inner_deg) - TIBIA_ATTACH_ANGLE

    # R side дээр femur/tibia inversion
    if invert_ft:
        femur_servo_deg = -femur_servo_deg
        tibia_servo_deg = -tibia_servo_deg

    # safety clamp
    coxa_servo_deg = clamp(coxa_servo_deg, -45, 45)
    femur_servo_deg = clamp(femur_servo_deg, -45, 45)
    tibia_servo_deg = clamp(tibia_servo_deg, -45, 45)

    p_coxa = deg2us(coxa_servo_deg, *CAL[coxa_servo])
    p_femur = deg2us(femur_servo_deg, *CAL[femur_servo])
    p_tibia = deg2us(tibia_servo_deg, *CAL[tibia_servo])

    print(
        f"{leg_name}: "
        f"deg=({coxa_servo_deg:.1f}, {femur_servo_deg:.1f}, {tibia_servo_deg:.1f}) "
        f"us=({p_coxa}, {p_femur}, {p_tibia})"
    )

    return {
        coxa_servo: p_coxa,
        femur_servo: p_femur,
        tibia_servo: p_tibia,
    }

# --------------------------------------------------
# Build all 18 values
# --------------------------------------------------
servo_values = {}

# Left side: normal
servo_values.update(solve_leg("L1", "L11", "L12", "L13", invert_ft=False))
servo_values.update(solve_leg("L2", "L21", "L22", "L23", invert_ft=False))
servo_values.update(solve_leg("L3", "L31", "L32", "L33", invert_ft=False))

# Right side: femur/tibia inverted
servo_values.update(solve_leg("R1", "R11", "R12", "R13", invert_ft=True))
servo_values.update(solve_leg("R2", "R21", "R22", "R23", invert_ft=True))
servo_values.update(solve_leg("R3", "R31", "R32", "R33", invert_ft=True))

# P00..P17 packet
values_p0_p17 = [1500] * 18
for servo_name, pulse in servo_values.items():
    values_p0_p17[PIN[servo_name]] = pulse

print("P00..P17 values:")
for i, v in enumerate(values_p0_p17):
    print(f"P{i:02d} = {v}")

# --------------------------------------------------
# SEND
# --------------------------------------------------
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

try:
    # Relay ON
    send_set(ser, 26, [1], "RELAY ON:")
    time.sleep(1)

    # All servos to MODE_BLOCK pose
    send_set(ser, 0, values_p0_p17, "ALL LEGS MODE_BLOCK:")
    time.sleep(3)

    print("Done. Robot isytion. Press Ctrl+C to stop.")

    # Servo-нууд байрлалаа барьж байхын тулд relay ON байх ёстой.
    # Гарахдаа Ctrl+C дарна, тэгвэл finally блок relay унтраана.
    while True:
        time.sleep(1)

finally:
    # Ctrl+C эсвэл алдаа гарвал relay унтраагаад serial хаана
    send_set(ser, 26, [0], "RELAY OFF:")
    time.sleep(0.5)
    ser.close()