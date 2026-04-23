import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from collections import deque
import urllib.request
import os

# ─────────────────────────────────────────────
#  Download model
# ─────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task (~9 MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH)
    print("Done.")

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
CAM_W, CAM_H   = 1280, 720
BRUSH_BASE      = 4
NEON_BLUR       = 23
NEON_LAYERS     = 4
SMOOTHING       = 0.55          # 0=raw, 1=frozen  — lower = more responsive
MAX_PTS         = 2048
GAP_THRESHOLD   = 40            # px — if jump > this, insert pen-up

# Neon palette  (BGR, display name, hex label)
PALETTE = [
    ((255, 60,  180), "PINK",    "#FF3CB4"),
    ((  0, 255, 255), "YELLOW",  "#FFFF00"),
    ((255, 255,   0), "CYAN",    "#00FFFF"),
    ((  0, 255, 128), "GREEN",   "#00FF80"),
    ((  0, 128, 255), "ORANGE",  "#FF8000"),
    ((255, 255, 255), "WHITE",   "#FFFFFF"),
]

# UI
BAR_H    = 72
BTN_R    = 22          # colour circle radius
BTN_GAP  = 64          # spacing between circles
BAR_PAD  = 20

# ─────────────────────────────────────────────
#  MediaPipe
# ─────────────────────────────────────────────
options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.55,
    min_hand_presence_confidence=0.55,
    min_tracking_confidence=0.55,
)
landmarker = HandLandmarker.create_from_options(options)

TIP = [4, 8, 12, 16, 20]
PIP = [3, 6, 10, 14, 18]

# ─────────────────────────────────────────────
#  State
# ─────────────────────────────────────────────
queues       = [deque(maxlen=MAX_PTS) for _ in PALETTE]
color_idx    = 0
drawing      = False
sx, sy       = 0, 0       # smoothed finger position
frame_ts     = 0

# ─────────────────────────────────────────────
#  Gesture helpers
# ─────────────────────────────────────────────
def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def gestures(lm):
    idx  = finger_up(lm, 8,  6)
    mid  = finger_up(lm, 12, 10)
    ring = finger_up(lm, 16, 14)
    pin  = finger_up(lm, 20, 18)
    return idx, mid, ring, pin

def is_draw(g):   return g[0] and not g[1] and not g[2] and not g[3]
def is_peace(g):  return g[0] and g[1] and not g[2] and not g[3]
def is_fist(g):   return not any(g)

# ─────────────────────────────────────────────
#  Neon render
# ─────────────────────────────────────────────
def neon(ink):
    glow = ink.copy()
    for _ in range(NEON_LAYERS):
        blur = cv2.GaussianBlur(glow, (NEON_BLUR, NEON_BLUR), 0)
        glow = cv2.addWeighted(glow, 1.0, blur, 0.7, 0)
    return glow

def render_strokes(target, brush):
    for i, q in enumerate(queues):
        pts   = list(q)
        color = PALETTE[i][0]
        for j in range(1, len(pts)):
            if pts[j-1] is None or pts[j] is None:
                continue
            cv2.line(target, pts[j-1], pts[j], color, brush)

# ─────────────────────────────────────────────
#  Beautiful HUD bar
# ─────────────────────────────────────────────
def btn_center(i):
    """Return (x, y) of the i-th colour circle."""
    return (BAR_PAD + BTN_R + i * BTN_GAP, BAR_H // 2)

CLEAR_X1 = BAR_PAD + BTN_R + len(PALETTE) * BTN_GAP + 10
CLEAR_X2 = CLEAR_X1 + 90
CLEAR_Y1 = BAR_H // 2 - 18
CLEAR_Y2 = BAR_H // 2 + 18

def draw_hud(frame, drawing_state, cur_idx):
    h, w = frame.shape[:2]

    # Frosted glass bar background
    bar = np.zeros((BAR_H, w, 3), dtype=np.uint8)
    bar[:] = (18, 18, 28)
    # subtle gradient tint
    for row in range(BAR_H):
        alpha = 1.0 - row / BAR_H * 0.4
        bar[row] = (np.array([18, 18, 28]) * alpha).astype(np.uint8)

    # Blend bar onto frame
    roi = frame[:BAR_H, :w]
    cv2.addWeighted(bar, 0.82, roi, 0.18, 0, roi)

    # Divider line (neon accent)
    accent = PALETTE[cur_idx][0]
    cv2.line(frame, (0, BAR_H), (w, BAR_H), accent, 2)
    # subtle glow on divider
    glow_line = np.zeros_like(frame)
    cv2.line(glow_line, (0, BAR_H-1), (w, BAR_H-1), accent, 6)
    glow_line = cv2.GaussianBlur(glow_line, (0, 0), 4)
    cv2.add(frame, glow_line, frame)

    # Colour circles
    for i, (bgr, name, _) in enumerate(PALETTE):
        cx, cy = btn_center(i)
        if i == cur_idx:
            # Outer glow ring
            glow_buf = np.zeros_like(frame)
            cv2.circle(glow_buf, (cx, cy), BTN_R + 10, bgr, 3)
            glow_buf = cv2.GaussianBlur(glow_buf, (0,0), 6)
            cv2.add(frame, glow_buf, frame)
            # White selection ring
            cv2.circle(frame, (cx, cy), BTN_R + 5, (240, 240, 255), 2)

        # Filled circle
        cv2.circle(frame, (cx, cy), BTN_R, bgr, -1)
        # Inner shine
        cv2.circle(frame, (cx - BTN_R//4, cy - BTN_R//4), BTN_R//4,
                   tuple(min(255, c+80) for c in bgr), -1)

    # CLEAR button — sleek pill
    radius = 10
    cv2.rectangle(frame, (CLEAR_X1 + radius, CLEAR_Y1),
                  (CLEAR_X2 - radius, CLEAR_Y2), (38, 28, 48), -1)
    cv2.rectangle(frame, (CLEAR_X1, CLEAR_Y1 + radius),
                  (CLEAR_X2, CLEAR_Y2 - radius), (38, 28, 48), -1)
    cv2.circle(frame, (CLEAR_X1 + radius, CLEAR_Y1 + radius), radius, (38, 28, 48), -1)
    cv2.circle(frame, (CLEAR_X2 - radius, CLEAR_Y1 + radius), radius, (38, 28, 48), -1)
    cv2.circle(frame, (CLEAR_X1 + radius, CLEAR_Y2 - radius), radius, (38, 28, 48), -1)
    cv2.circle(frame, (CLEAR_X2 - radius, CLEAR_Y2 - radius), radius, (38, 28, 48), -1)
    # Border
    cv2.rectangle(frame, (CLEAR_X1 + radius, CLEAR_Y1),
                  (CLEAR_X2 - radius, CLEAR_Y2), (90, 70, 120), 1)
    cv2.putText(frame, "CLEAR", (CLEAR_X1 + 14, CLEAR_Y1 + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 180, 230), 1, cv2.LINE_AA)

    # Status pill (right side)
    status_txt = "● DRAWING" if drawing_state else "○ HOVER"
    status_col = (80, 255, 160) if drawing_state else (120, 120, 160)
    tx = w - 200
    ty = BAR_H // 2 + 6
    cv2.putText(frame, status_txt, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, status_col, 1, cv2.LINE_AA)

    return frame

def hud_hit(x, y):
    """Returns ('color', idx) | ('clear',) | None"""
    if y > BAR_H:
        return None
    for i in range(len(PALETTE)):
        cx, cy = btn_center(i)
        if (x - cx)**2 + (y - cy)**2 <= (BTN_R + 6)**2:
            return ('color', i)
    if CLEAR_X1 <= x <= CLEAR_X2 and CLEAR_Y1 <= y <= CLEAR_Y2:
        return ('clear',)
    return None

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce lag

WIN = "Air Canvas"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("AIR CANVAS  |  ☝ draw  ✌ lift  ✊ clear  Q quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame    = cv2.flip(frame, 1)
    frame    = cv2.resize(frame, (CAM_W, CAM_H))
    frame_ts += 16   # ~60 fps tick

    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    det       = landmarker.detect_for_video(mp_img, frame_ts)

    if det.hand_landmarks:
        lm  = det.hand_landmarks[0]
        g   = gestures(lm)

        # Raw fingertip pos
        rx = int(lm[8].x * CAM_W)
        ry = int(lm[8].y * CAM_H)

        # Exponential smoothing — more responsive than before
        sx = int(sx * SMOOTHING + rx * (1 - SMOOTHING))
        sy = int(sy * SMOOTHING + ry * (1 - SMOOTHING))

        if is_fist(g):
            for q in queues: q.clear()
            drawing = False

        elif is_peace(g):
            drawing = False
            queues[color_idx].append(None)

        elif is_draw(g):
            hit = hud_hit(sx, sy)
            if hit and hit[0] == 'clear':
                for q in queues: q.clear()
                drawing = False
            elif hit and hit[0] == 'color':
                color_idx = hit[1]
                drawing = False
            else:
                # Gap detection — prevents lines jumping across canvas
                q = queues[color_idx]
                if q and q[-1] is not None:
                    px, py = q[-1]
                    if abs(sx - px) + abs(sy - py) > GAP_THRESHOLD:
                        q.append(None)
                drawing = True
                queues[color_idx].append((sx, sy))
        else:
            drawing = False

        # Draw skeleton
        conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                 (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                 (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
        pts_sk = [(int(l.x*CAM_W), int(l.y*CAM_H)) for l in lm]
        for a,b in conns:
            cv2.line(frame, pts_sk[a], pts_sk[b], (60,60,80), 1)

        # Cursor ring
        col = PALETTE[color_idx][0]
        cv2.circle(frame, (sx, sy), 12, col, 2)
        cv2.circle(frame, (sx, sy),  4, col, -1)
        # Glow on cursor
        gcur = np.zeros_like(frame)
        cv2.circle(gcur, (sx, sy), 14, col, 3)
        gcur = cv2.GaussianBlur(gcur, (0,0), 5)
        cv2.add(frame, gcur, frame)

    else:
        drawing = False
        sx, sy = CAM_W//2, CAM_H//2

    # Render strokes
    ink = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    render_strokes(ink, BRUSH_BASE)
    glow_ink = neon(ink)

    mask     = cv2.cvtColor(glow_ink, cv2.COLOR_BGR2GRAY)
    _, mask  = cv2.threshold(mask, 8, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg  = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg  = cv2.bitwise_and(glow_ink, glow_ink, mask=mask)
    out = cv2.add(bg, fg)

    out = draw_hud(out, drawing, color_idx)

    cv2.imshow(WIN, out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()