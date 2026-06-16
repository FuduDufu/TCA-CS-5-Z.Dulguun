import cv2
import numpy as np
import math

# ============ Камерын параметрүүд ============
FOV_DEG = 96       # Камерын харагдах өнцөг (градус)
BASELINE_M = 0.065 # Хоёр линзний хоорондох зай: 6.5 см
# =============================================

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camR = cv2.VideoCapture(1)
# Камерын resolution аль болох өндөр болгох
camR.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  # хамгийн өндөр өргөн
camR.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Эхний frame-ээс focal length тооцоолох
ret_init, frame_init = camR.read()
if not ret_init:
    print("Камер нээгдсэнгүй!")
    exit()
half_w = frame_init.shape[1] // 2
FOCAL_LENGTH_PX = (half_w / 2) / math.tan(math.radians(FOV_DEG / 2))

# StereoSGBM
window_size = 5
stereo_left = cv2.StereoSGBM.create(
    minDisparity=0,
    numDisparities=128,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# WLS filter
stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.5)

print("=" * 50)
print("FACE DISTANCE - Нүүр таних + зай хэмжих")
print(f"  Resolution:   {half_w}x{frame_init.shape[0]}")
print(f"  Focal length: {FOCAL_LENGTH_PX:.1f} px")
print(f"  Baseline:     {BASELINE_M * 100:.1f} cm")
print(f"  FOV:          {FOV_DEG} deg")
print("  ESC = гарах")
print("=" * 50)

while True:
    ret, frame = camR.read()
    if not ret:
        break

    frame = cv2.flip(frame, -1)
    h, w, _ = frame.shape
    left = frame[:, :w // 2]
    right = frame[:, w // 2:]

    # Grayscale + CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_left = clahe.apply(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
    gray_right = clahe.apply(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))

    # Stereo disparity тооцоолох
    disp_left = stereo_left.compute(gray_left, gray_right)
    disp_right = stereo_right.compute(gray_right, gray_left)
    disparity = wls_filter.filter(disp_left, gray_left, None, disp_right)
    disparity = disparity.astype(np.float32) / 16.0

    # Disparity map харуулахад
    valid_mask = disparity > 0
    disp_vis = np.zeros_like(disparity)
    if valid_mask.any():
        disp_vis[valid_mask] = disparity[valid_mask]
    disp_u8 = np.uint8(cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX))
    disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)
    disp_color[~valid_mask] = 0

    # --- Нүүр илрүүлэх (зүүн зураг дээр) ---
    faces = face_cascade.detectMultiScale(
        gray_left,
        scaleFactor=1.05,  # илүү нарийн хайлт (1.1 -> 1.05)
        minNeighbors=4,
        minSize=(15, 15)   # жижиг нүүр илрүүлэх (30 -> 15)
    )

    left_draw = left.copy()
    for (fx, fy, fw, fh) in faces:
        # Нүүрний хэсгийн disparity-ийн median авах
        roi = disparity[fy:fy + fh, fx:fx + fw]
        valid_roi = roi[roi > 0]

        if len(valid_roi) > 10:
            med_disp = np.median(valid_roi)
            if med_disp > 0:
                distance_m = (FOCAL_LENGTH_PX * BASELINE_M) / med_disp

                if 0.1 < distance_m < 20.0:
                    # Ойр=улаан, хол=ногоон
                    ratio = min(distance_m / 5.0, 1.0)
                    color = (0, int(255 * ratio), int(255 * (1 - ratio)))

                    cv2.rectangle(left_draw, (fx, fy), (fx + fw, fy + fh), color, 2)
                    cv2.putText(left_draw, f"{distance_m:.2f} m",
                                (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)
                    continue

        # Disparity олдоогүй бол зүгээр хүрээ зурна
        cv2.rectangle(left_draw, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
        cv2.putText(left_draw, "face", (fx, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Дэлгэцэнд гаргах
    top_row = np.hstack([left_draw, right])
    disp_resized = cv2.resize(disp_color, (top_row.shape[1], h))
    combined = np.vstack([top_row, disp_resized])

    scale = min(1280 / combined.shape[1], 720 / combined.shape[0])
    display = cv2.resize(combined, None, fx=scale, fy=scale)

    cv2.imshow("Face Distance", display)

    if cv2.waitKey(1) == 27:
        break

camR.release()
cv2.destroyAllWindows()
