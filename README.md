# Portal

Hand-gesture webcam effect. Hold up both hands to open a "portal" between your fingertips with a color filter inside, while a particle cloud follows the shape of your hands.

Built with OpenCV and MediaPipe Hands.

## Features

- Portal frame between the thumbs and index fingers of both hands
- Pinch (thumb + index finger) to switch between filters
- Particle cloud that takes the shape of your hands and drifts away when they leave
- On-screen button to toggle rainbow / green particles (point at it with your index finger)

## Run

Requires Python 3.12 (MediaPipe 0.10.21 doesn't work with newer versions).

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python portal.py
```

Quit with `q`.