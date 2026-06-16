import cv2
import numpy as np
import math

# =============================================================
# ОБЪЕКТУУДЫН БАЙРЛАЛ
# Robot (0,0)-д, P1 0.44m, P2 0.41m зайд, хоорондоо 0.05m
# =============================================================
# P1=(0, 0.44), P2 нь P1-ээс 0.05m, эх цэгээс 0.41m байхаар тооцсон
# x²+(y-0.44)²=0.05² ба x²+y²=0.41² → x≈0.039, y≈0.408
P2 = {"label": "P2", "x":  0.030, "y": 0.500}
P1 = {"label": "P1", "x":  -0.309, "y": 0.664}

objects = [P1, P2]

ROBOT = {"x": 0.0, "y": 0.0}

# =============================================================
# MAP ТОХИРГОО
# =============================================================
MAP_W, MAP_H = 900, 700

COL_BG          = (24,  24,  30)
COL_GRID        = (50,  50,  65)
COL_GRID_FINE   = (35,  35,  45)
COL_AXIS        = (90,  90, 110)
COL_TEXT_HEAD   = (250, 230, 140)
COL_TEXT        = (220, 220, 230)
COL_TEXT_DIM    = (140, 140, 155)
COL_HEADER_BG   = (55,  55,  75)
COL_ROBOT       = (255, 220,  60)
COL_P1          = (0,   220,  90)
COL_P2          = (0,   180, 255)
COL_DIST_LINE   = (200, 200,  60)
COL_GAP_LINE    = (255, 120,  60)

# =============================================================
# WORLD → SCREEN
# =============================================================
def make_transform(all_pts, w, h, margin=100, header=50):
    xs = [p["x"] for p in all_pts]
    ys = [p["y"] for p in all_pts]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    span = max(max(xs)-min(xs), max(ys)-min(ys), 0.3) + 0.3
    aw = w - 2*margin
    ah = h - header - 2*margin
    scale = min(aw / span, ah / span)

    def w2s(wx, wy):
        sx = int(w/2 + (wx - cx) * scale)
        sy = int((h + header)/2 - (wy - cy) * scale)
        return sx, sy

    return w2s, cx, cy, span, scale

# =============================================================
# ТУСЛАХ: distance label шугамын дунд
# =============================================================
def draw_dist_label(img, p1s, p2s, text, color, offset=(0, -10)):
    mx = (p1s[0] + p2s[0]) // 2 + offset[0]
    my = (p1s[1] + p2s[1]) // 2 + offset[1]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(img, (mx - tw//2 - 3, my - th - 2),
                       (mx + tw//2 + 3, my + 3), COL_BG, -1)
    cv2.putText(img, text, (mx - tw//2, my),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

# =============================================================
# RENDER
# =============================================================
def render():
    img = np.full((MAP_H, MAP_W, 3), COL_BG, dtype=np.uint8)

    # Header
    cv2.rectangle(img, (0, 0), (MAP_W, 46), COL_HEADER_BG, -1)
    d1 = math.hypot(P1["x"] - ROBOT["x"], P1["y"] - ROBOT["y"])
    d2 = math.hypot(P2["x"] - ROBOT["x"], P2["y"] - ROBOT["y"])
    gap = math.hypot(P1["x"] - P2["x"], P1["y"] - P2["y"])
    cv2.putText(img,
        f"Object Map   P1={d1:.3f}m   P2={d2:.3f}m   Gap={gap:.3f}m",
        (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, COL_TEXT_HEAD, 2)

    all_pts = objects + [ROBOT]
    w2s, cx, cy, span, scale = make_transform(all_pts, MAP_W, MAP_H)

    # Fine grid (0.05m крок)
    step_fine = 0.05
    gx0 = cx - span/2 - step_fine
    gx1 = cx + span/2 + step_fine
    gy0 = cy - span/2 - step_fine
    gy1 = cy + span/2 + step_fine
    v = gx0
    while v <= gx1 + 1e-9:
        p1s = w2s(v, gy0); p2s = w2s(v, gy1)
        col = COL_AXIS if abs(v) < 1e-9 else COL_GRID_FINE
        cv2.line(img, p1s, p2s, col, 1)
        v = round(v + step_fine, 6)
    v = gy0
    while v <= gy1 + 1e-9:
        p1s = w2s(gx0, v); p2s = w2s(gx1, v)
        col = COL_AXIS if abs(v) < 1e-9 else COL_GRID_FINE
        cv2.line(img, p1s, p2s, col, 1)
        v = round(v + step_fine, 6)

    # Major grid (0.1m крок) + axis labels
    step = 0.1
    v = round(math.floor(gx0 / step) * step, 6)
    while v <= gx1 + 1e-9:
        p1s = w2s(v, gy0); p2s = w2s(v, gy1)
        col = COL_AXIS if abs(v) < 1e-9 else COL_GRID
        cv2.line(img, p1s, p2s, col, 1)
        lp = w2s(v, gy0)
        cv2.putText(img, f"{v:.1f}", (lp[0]+2, lp[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL_TEXT_DIM, 1)
        v = round(v + step, 6)
    v = round(math.floor(gy0 / step) * step, 6)
    while v <= gy1 + 1e-9:
        p1s = w2s(gx0, v); p2s = w2s(gx1, v)
        col = COL_AXIS if abs(v) < 1e-9 else COL_GRID
        cv2.line(img, p1s, p2s, col, 1)
        lp = w2s(gx0, v)
        cv2.putText(img, f"{v:.1f}", (lp[0]-32, lp[1]-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL_TEXT_DIM, 1)
        v = round(v + step, 6)

    rs = w2s(ROBOT["x"], ROBOT["y"])
    p1s = w2s(P1["x"], P1["y"])
    p2s = w2s(P2["x"], P2["y"])

    # Robot → P1 зайн шугам
    cv2.line(img, rs, p1s, COL_DIST_LINE, 1, cv2.LINE_AA)
    draw_dist_label(img, rs, p1s, f"{d1:.3f} m", COL_DIST_LINE, (-30, -8))

    # Robot → P2 зайн шугам
    cv2.line(img, rs, p2s, COL_DIST_LINE, 1, cv2.LINE_AA)
    draw_dist_label(img, rs, p2s, f"{d2:.3f} m", COL_DIST_LINE, (30, -8))

    # P1 ↔ P2 зазрын шугам
    cv2.line(img, p1s, p2s, COL_GAP_LINE, 2, cv2.LINE_AA)
    draw_dist_label(img, p1s, p2s, f"gap {gap:.3f} m", COL_GAP_LINE, (0, -12))

    # Robot
    cv2.circle(img, rs, 10, COL_ROBOT, -1)
    cv2.circle(img, rs, 12, (255, 255, 255), 1)
    cv2.putText(img, "ROBOT", (rs[0] - 22, rs[1] + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_ROBOT, 1)
    cv2.putText(img, "(0.000, 0.000)", (rs[0] - 40, rs[1] + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_TEXT_DIM, 1)

    # P1
    cv2.circle(img, p1s, 13, COL_P1, -1)
    cv2.circle(img, p1s, 15, (255, 255, 255), 1)
    cv2.putText(img, "P1", (p1s[0] - 10, p1s[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_P1, 2)
    cv2.putText(img, f"({P1['x']:.3f}, {P1['y']:.3f})",
                (p1s[0] - 42, p1s[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_TEXT_DIM, 1)

    # P2
    cv2.circle(img, p2s, 13, COL_P2, -1)
    cv2.circle(img, p2s, 15, (255, 255, 255), 1)
    cv2.putText(img, "P2", (p2s[0] - 10, p2s[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_P2, 2)
    cv2.putText(img, f"({P2['x']:.3f}, {P2['y']:.3f})",
                (p2s[0] - 42, p2s[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_TEXT_DIM, 1)

    # Legend
    lx, ly = MAP_W - 200, 60
    cv2.putText(img, "LEGEND", (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT_HEAD, 1)
    ly += 22
    entries = [
        (COL_ROBOT,     "Robot  (0, 0)"),
        (COL_P1,        f"P1  d={d1:.3f} m"),
        (COL_P2,        f"P2  d={d2:.3f} m"),
        (COL_DIST_LINE, "Distance to robot"),
        (COL_GAP_LINE,  f"Gap = {gap:.3f} m"),
    ]
    for col, lbl in entries:
        cv2.circle(img, (lx + 8, ly - 6), 6, col, -1)
        cv2.putText(img, lbl, (lx + 20, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COL_TEXT, 1)
        ly += 20

    return img

# =============================================================
# MAIN
# =============================================================
img = render()

cv2.namedWindow("Object Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Map", MAP_W, MAP_H)
cv2.imshow("Object Map", img)

out = "map_objects_result.png"
cv2.imwrite(out, img)
print(f"Saved: {out}")

d1 = math.hypot(P1["x"], P1["y"])
d2 = math.hypot(P2["x"], P2["y"])
gap = math.hypot(P1["x"] - P2["x"], P1["y"] - P2["y"])
print(f"P1 = ({P1['x']:.3f}, {P1['y']:.3f})  → robot-аас {d1:.4f} m")
print(f"P2 = ({P2['x']:.3f}, {P2['y']:.3f})  → robot-аас {d2:.4f} m")
print(f"Gap P1↔P2 = {gap:.4f} m")

cv2.waitKey(0)
cv2.destroyAllWindows()
