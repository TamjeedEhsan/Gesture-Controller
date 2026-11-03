import cv2
import numpy as np
import os
import time
from datetime import datetime
from hand_tracking import HandTracker
from color_picker import ColorPicker
from virtual_keyboard import VirtualKeyboard
from mouse_control import MouseController
from volume_control import VolumeController
from zoom_control import ZoomController
from media_control import MediaController
from instructions import InstructionsPanel

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap.set(cv2.CAP_PROP_FPS, 30)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera Resolution: {W}x{H}")
cv2.namedWindow("Gesture Controller", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Gesture Controller", 1280, 800)
tracker = HandTracker()
picker = ColorPicker(W)
keyboard = VirtualKeyboard(W)
mouse = MouseController(W, H)
volume = VolumeController()
zoom = ZoomController()
media = MediaController()
instructions = InstructionsPanel(W, H)
cv2.namedWindow("Gesture Controller")
cv2.setMouseCallback("Gesture Controller", instructions.on_mouse)

mode = "draw"   
canvas = None
prev = None

save_msg = ""
save_time = 0
os.makedirs("drawings", exist_ok=True)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    frame = tracker.find_hands(frame)
    lm = tracker.get_landmarks(frame)

    if mode == "draw":
        if lm:
            fingers = tracker.fingers_up(lm)
            x, y = lm[8][1], lm[8][2]

            if fingers[1] == 1 and fingers[2] == 1:
                prev = None
                picker.select(x, y)
                cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
            elif fingers[1] == 1 and fingers[2] == 0:
                if y > picker.height:
                    if prev is None:
                        prev = (x, y)
                    cv2.line(canvas, prev, (x, y), picker.color, picker.thickness)
                    prev = (x, y)
                    cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
                else:
                    prev = None
            else:
                prev = None
        else:
            prev = None

        mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
        frame[mask] = canvas[mask]
        picker.draw(frame)

    elif mode == "keyboard":
        if lm:
            x, y = lm[8][1], lm[8][2]
            palm = tracker.distance(lm, 0, 9)
            pinching = tracker.distance(lm, 4, 8) < 0.3 * palm
            keyboard.update(x, y, pinching)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), cv2.FILLED)
        else:
            keyboard.update(-1, -1, False)
        keyboard.draw(frame)
    elif mode == "mouse":
        mouse.draw(frame)
        if lm:
            x, y = lm[8][1], lm[8][2]
            palm = tracker.distance(lm, 0, 9)
            pinching = tracker.distance(lm, 4, 8) < 0.3 * palm
            mouse.update(x, y, pinching)
            cv2.circle(frame, (x, y), 8, (255, 0, 255), cv2.FILLED)

    elif mode == "volume":
            if lm:
                fingers = tracker.fingers_up(lm)
                palm = tracker.distance(lm, 0, 9)
                ratio = tracker.distance(lm, 4, 8) / palm
                locked = fingers[4] == 1          # pinky up = lock the volume
    
                volume.update(ratio, locked)
    
                p1, p2 = (lm[4][1], lm[4][2]), (lm[8][1], lm[8][2])
                color = (0, 0, 255) if locked else (0, 255, 0)
                cv2.line(frame, p1, p2, color, 3)
                cv2.circle(frame, p1, 8, color, cv2.FILLED)
                cv2.circle(frame, p2, 8, color, cv2.FILLED)
            volume.draw(frame)

    elif mode == "zoom":
        if lm:
            fingers = tracker.fingers_up(lm)
            palm = tracker.distance(lm, 0, 9)
            ratio = tracker.distance(lm, 4, 8) / palm
            locked = fingers[4] == 1         

            zoom.update(ratio, locked)

            p1, p2 = (lm[4][1], lm[4][2]), (lm[8][1], lm[8][2])
            color = (0, 0, 255) if locked else (0, 255, 0)
            cv2.line(frame, p1, p2, color, 3)
            cv2.circle(frame, p1, 8, color, cv2.FILLED)
            cv2.circle(frame, p2, 8, color, cv2.FILLED)

        frame = zoom.apply(frame)   
        zoom.draw(frame)

    elif mode == "media":
        if lm:
            fingers = tracker.fingers_up(lm)
            media.update(fingers)
        else:
            media.update([0, 0, 0, 0, 0])   
        media.draw(frame)

    cv2.putText(frame, f"Mode: {mode}", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if time.time() - save_time < 2:
        cv2.putText(frame, save_msg, (10, frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    instructions.draw(frame)
    
    cv2.imshow("Gesture Controller", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('d'):
        mode, prev = "draw", None
    elif key == ord('k'):
        mode, prev = "keyboard", None
    elif key == ord('m'):
        mode, prev = "mouse", None
    elif key == ord('v'):
        mode, prev = "volume", None
    elif key == ord('z'):
        mode, prev = "zoom", None
    elif key == ord('p'):
        mode, prev = "media", None
    elif key == ord('i'):
        instructions.toggle()
    elif key == ord('s'):
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
            if mask.any():
                out = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)        
                out[:, :, 3] = mask.astype(np.uint8) * 255           
                name = datetime.now().strftime("drawings/drawing_%Y%m%d_%H%M%S.png")
                cv2.imwrite(name, out)
                save_msg, save_time = f"Saved {name}", time.time()
            else:
                save_msg, save_time = "Nothing to save", time.time()
    
    

cap.release()
cv2.destroyAllWindows()