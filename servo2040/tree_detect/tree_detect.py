"""
tree_detect.py
==============
Зургаас мод таниад label-тэй хайрцагаар тэмдэглэнэ.

Арга:
  1. Зүүн хагасыг (stereo-ийн зүүн нүд) ашиглана
  2. HSV mask — ногоон + хүрэн/бор өнгө (навч + иш)
  3. Contour → bbox → дээрээс доош уртассан хэлбэртэй бол мод гэж үзнэ
  4. Үр дүнг tree_result.png болгон хадгална

Ашиглалт:
  python tree_detect.py gadaad_orchind_mod_tanilt.png
  python tree_detect.py  (argument-гүй бол ижил folder-ийн default зургийг авна)
"""

import sys
import cv2
import numpy as np

# =============================================================
# ТОХИРГОО
# =============================================================
DEFAULT_IMAGE  = "gadaad_orchind_mod_tanilt.png"
OUTPUT_IMAGE   = "tree_result.png"

# Ногооны HSV range (навч, модны оройн хэсэг)
GREEN_LOW  = np.array([30,  25,  25])
GREEN_HIGH = np.array([95, 255, 255])

# Хүрэн/бор өнгө (иш, гишүү)
BROWN_LOW  = np.array([5,  30,  20])
BROWN_HIGH = np.array([30, 200, 180])

# Contour шүүлт
MIN_AREA        = 600    # хэт жижиг noise хаяна
MIN_ASPECT      = 1.2    # өндөр/өргөн — мод нь босоо урт байна
MAX_ASPECT      = 15.0
MIN_HEIGHT_PX   = 40     # хэт намхан объектыг хаяна

# Ойролцоо bbox-уудыг нэгтгэх зай (пиксел)
MERGE_DIST      = 60

# =============================================================
# ТУСЛАХ ФУНКЦҮҮД
# =============================================================
def get_mask(img_bgr):
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mg    = cv2.inRange(hsv, GREEN_LOW,  GREEN_HIGH)
    mb    = cv2.inRange(hsv, BROWN_LOW,  BROWN_HIGH)
    mask  = cv2.bitwise_or(mg, mb)
    k5    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k9    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k9)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5)
    return mask

def find_tree_boxes(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA: continue
        x, y, w, h = cv2.boundingRect(c)
        if h < MIN_HEIGHT_PX: continue
        aspect = h / max(w, 1)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT): continue
        boxes.append([x, y, x+w, y+h])
    return boxes

def merge_boxes(boxes, dist=MERGE_DIST):
    """Ойролцоо bbox-уудыг нэгтгэнэ (нэг мод олон хэсэгт хуваагдсан үед)."""
    if not boxes: return []
    merged = True
    while merged:
        merged = False
        out = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]: continue
            x1,y1,x2,y2 = boxes[i]
            for j in range(i+1, len(boxes)):
                if used[j]: continue
                ax1,ay1,ax2,ay2 = boxes[j]
                # Хоёр bbox ойрхон уу?
                if (ax1 - dist < x2 and ax2 + dist > x1 and
                    ay1 - dist < y2 and ay2 + dist > y1):
                    x1 = min(x1,ax1); y1 = min(y1,ay1)
                    x2 = max(x2,ax2); y2 = max(y2,ay2)
                    used[j] = True; merged = True
            out.append([x1,y1,x2,y2])
            used[i] = True
        boxes = out
    # Нэгтгэсний дараа aspect ratio дахин шалгана
    result = []
    for (x1,y1,x2,y2) in boxes:
        w = x2-x1; h = y2-y1
        if h < MIN_HEIGHT_PX: continue
        if h / max(w,1) < MIN_ASPECT * 0.7: continue
        result.append((x1,y1,x2,y2))
    return result

def draw_results(img, boxes, title="Tree Detection"):
    out = img.copy()
    for idx, (x1,y1,x2,y2) in enumerate(boxes):
        w = x2-x1; h = y2-y1
        label = f"MOD #{idx+1}  {w}x{h}px"
        cv2.rectangle(out, (x1,y1),(x2,y2), (0, 220, 80), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1-th-8), (x1+tw+4, y1), (0, 180, 60), -1)
        cv2.putText(out, label, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # Дээд зүүн буланд нийт тоо
    cv2.rectangle(out, (0,0),(260,34),(0,0,0),-1)
    cv2.putText(out, f"{title}  —  {len(boxes)} мод",
                (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 255), 2)
    return out

# =============================================================
# ҮНДЭС
# =============================================================
def run(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Зураг нээгдсэнгүй: {image_path}"); return

    # Stereo зураг бол зөвхөн зүүн хагасыг ав
    h, w = img.shape[:2]
    left = img[:, w//2:] if w > h * 1.5 else img

    print(f"[IN]  {image_path}  left: {left.shape[1]}x{left.shape[0]}")

    mask  = get_mask(left)
    boxes = find_tree_boxes(mask)
    boxes = merge_boxes(boxes)

    print(f"[OUT] {len(boxes)} tree(s) found")
    for i,(x1,y1,x2,y2) in enumerate(boxes):
        print(f"  #{i+1}  bbox=({x1},{y1})-({x2},{y2})  size={x2-x1}x{y2-y1}px")

    result = draw_results(left, boxes)

    # Маск дэлгэц (debugging)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([result, mask_bgr])

    cv2.namedWindow("Tree Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tree Detection", min(combined.shape[1], 1400), 600)
    cv2.imshow("Tree Detection", combined)

    out_path = image_path.replace(".png", "_result.png").replace(".jpg", "_result.jpg")
    cv2.imwrite(out_path, result)
    print(f"[SAVE] {out_path}")
    print("ESC / Q to exit")
    while True:
        k = cv2.waitKey(30)
        if k in (27, ord('q'), ord('Q')): break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    run(path)
