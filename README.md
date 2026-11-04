# Gesture Controller

Control your computer with hand gestures using just a webcam. Built with Python, OpenCV and MediaPipe as my first computer vision project.

![Demo](demo.gif)

## Features

- **Draw:** draw in the air with your fingertip, with a color palette and eraser
- **Virtual keyboard:** hover over keys and pinch to type
- **Mouse:** move the real cursor with your index finger and pinch to click
- **Volume:** control system volume with finger distance (Windows)
- **Zoom:** zoom the camera view in and out
- **Media:** play/pause and skip tracks with held gestures
- **Save:** export your drawing as a transparent PNG

## How it works

```
Webcam frame → MediaPipe finds 21 hand landmarks → interpret gesture → trigger action
```

Fingers are classified as up or down by comparing landmark positions. Pinch and slider gestures use the distance between fingertips, scaled by palm size so they work at any distance from the camera.

## Controls

Click the video window first, then use these hotkeys:

| Key | Action |
|---|---|
| `D` | Draw mode |
| `R` | Redo |
| `U` | Undo |
| `K` | Keyboard mode |
| `M` | Mouse mode |
| `V` | Volume mode |
| `Z` | Zoom mode |
| `P` | Media mode |
| `C` | Clear canvas |
| `S` | Save drawing |
| `Q` | Quit |

## Gestures

| Mode | Gesture |
|---|---|
| Draw | Index finger up = draw, index + middle up = pick a color |
| Keyboard | Hover a key with your index finger, pinch thumb and index to type |
| Mouse | Move your index finger to move the cursor, pinch to click |
| Volume / Zoom | Spread or pinch thumb and index finger, raise your pinky to lock |
| Media | Hold open palm = play/pause, index only = next, peace sign = previous |

## Setup

Requires **Python 3.11** (MediaPipe 0.10.14 does not support newer versions) and a webcam.

```bash
git clone https://github.com/TamjeedEhsan/AI-Gesture-Controller.git
cd AI-Gesture-Controller
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Volume control uses `pycaw` and works on Windows only.

## Project structure

```
├── main.py               # main loop and mode switching
├── hand_tracking.py      # HandTracker: MediaPipe wrapper, finger detection
├── color_picker.py       # draw-mode palette and eraser
├── virtual_keyboard.py   # on-screen keyboard
├── mouse_control.py      # cursor control
├── volume_control.py     # system volume
├── zoom_control.py       # view zoom
├── media_control.py      # media keys
└── requirements.txt
```

## Tech stack

Python, OpenCV, MediaPipe, NumPy, PyAutoGUI, pycaw
