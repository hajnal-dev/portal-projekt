# Portal

Kézgesztussal vezérelt webkamera-effekt. Két kézzel keretet lehet "rajzolni" a levegőbe, és a kereten belül színszűrő jelenik meg.

OpenCV + MediaPipe Hands.

## Futtatás

Python 3.12 kell (a MediaPipe 0.10.21 újabbal nem megy).

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python portal_projekt.py
```

Kilépés: `q`