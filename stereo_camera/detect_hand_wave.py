import cv2
import numpy as np
import math
import mediapipe as mp
import time
from collections import deque

# ============ Камерын параметрүүд ============
FOV_DEG = 96
BASELINE_M = 0.065
# ============================================

cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

ret, frame_init = cam.read()
if not ret:
    print("Камер нээгдсэнгүй!")
    exit()

half_w = frame_init.shape[1] // 2

FOCAL_LENGTH_PX = (half_w / 2) / math.tan(math.radians(FOV_DEG / 2))

# ================= Stereo ===================
window_size = 5

stereo = cv2.StereoSGBM.create(
    minDisparity=0,
    numDisparities=128,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2
)

# =============== MediaPipe ==================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Wave detect history
hand_positions = deque(maxlen=15)

print("=" * 50)
print("HAND WAVE DETECTION")
print("Wave hand -> HI")
print("ESC = Exit")
print("=" * 50)


def get_distance(disparity, x, y):
    h, w = disparity.shape

    x1 = max(0, x - 10)
    y1 = max(0, y - 10)
    x2 = min(w, x + 10)
    y2 = min(h, y + 10)

    roi = disparity[y1:y2, x1:x2]

    valid = roi[roi > 0]

    if len(valid) < 5:
        return None

    med = np.median(valid)

    if med <= 0:
        return None

    dist = (FOCAL_LENGTH_PX * BASELINE_M) / med

    if 0.1 < dist < 10:
        return dist

    return None


def detect_wave(history):
    if len(history) < 10:
        return False

    xs = [p[0] for p in history]

    movement = max(xs) - min(xs)

    direction_changes = 0

    for i in range(1, len(xs)-1):
        d1 = xs[i] - xs[i-1]
        d2 = xs[i+1] - xs[i]

        if d1 * d2 < 0:
            direction_changes += 1

    if movement > 80 and direction_changes >= 2:
        return True

    return False


while True:

    ret, frame = cam.read()

    if not ret:
        break

    frame = cv2.flip(frame, -1)

    h, w, _ = frame.shape

    left = frame[:, :w // 2]
    right = frame[:, w // 2:]

    # ============= Stereo disparity ============
    grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # ============= Hand detection =============
    rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                left,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Palm center
            cx = int(hand_landmarks.landmark[9].x * left.shape[1])
            cy = int(hand_landmarks.landmark[9].y * left.shape[0])

            hand_positions.append((cx, cy))

            # Distance
            dist = get_distance(disparity, cx, cy)

            if dist:
                cv2.putText(
                    left,
                    f"{dist:.2f}m",
                    (cx, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # Wave detect
            if detect_wave(hand_positions):

                cv2.putText(
                    left,
                    "HI",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 255),
                    5
                )

    # ==========================================
    combined = np.hstack([left, right])

    scale = min(1280 / combined.shape[1], 720 / combined.shape[0])

    display = cv2.resize(combined, None, fx=scale, fy=scale)

    cv2.imshow("Wave Detection", display)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()