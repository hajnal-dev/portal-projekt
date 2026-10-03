import cv2
import mediapipe as mp
import numpy as np
from particles import ParticleCloud

mp_hands = mp.solutions.hands          # hand detection module
mp_draw = mp.solutions.drawing_utils   # helper for drawing landmarks

# - - - Constants - - -
THUMB_TIP = 4         # index of the thumb tip among the 21 landmarks
INDEX_TIP = 8         # index of the index finger tip
TOUCH_THRESHOLD = 0.06  # below this distance = touch (normalized units)
FRAME_COLOR = (0, 255, 0)  # BGR: green
WHITE = 255           # in the mask: "inside"
SHOW_SKELETON = False  # set to True to see the hand landmarks (useful for debugging)

# Available filters: (name, OpenCV colormap)
FILTERS = [
    ("CYBER_NEON", cv2.COLORMAP_PLASMA),
    ("LAVA", cv2.COLORMAP_HOT),
    ("OCEAN", cv2.COLORMAP_OCEAN),
    ("RAINBOW", cv2.COLORMAP_JET),
]


def to_pixel(landmark, width, height):
    """Convert a normalized (0-1) landmark to pixel coordinates as an (x, y) tuple."""
    return (int(landmark.x * width), int(landmark.y * height))


class PortalApp:
    """Webcam portal effect: draws a frame between two hands with a color filter inside."""

    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

        # State that persists from frame to frame
        self.current_filter = 0
        self.was_touching = False
        self.particles = ParticleCloud(1000)

    def _is_touching(self, thumb, index):
        """Return True if the thumb tip and index finger tip are close enough."""
        distance = np.linalg.norm(np.array([thumb.x, thumb.y]) - np.array([index.x, index.y]))
        return distance <= TOUCH_THRESHOLD

    def _process_hands(self, frame, result):
        """Draw the hands and return the corner points and whether there was a touch."""
        height, width, _ = frame.shape
        corners = []
        touching = False

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                if SHOW_SKELETON:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                thumb = hand.landmark[THUMB_TIP]
                index = hand.landmark[INDEX_TIP]

                if self._is_touching(thumb, index):   # <- 1.
                    touching = True

                corners.append(to_pixel(thumb, width, height))
                corners.append(to_pixel(index, width, height))

        return corners, touching

    def _update_filter(self, touching):
        """Switch to the next filter at the moment a touch begins."""
        if touching and not self.was_touching:   # <- 2.
            self.current_filter = (self.current_filter + 1) % len(FILTERS)
        self.was_touching = touching   # <- 3.

    def _draw_portal(self, frame, corners):
        """If all 4 corners are present, draw the portal with filter and label."""
        if len(corners) != 4:
            return   # no two hands → nothing to draw

        filter_name, colormap = FILTERS[self.current_filter]
        height, width, _ = frame.shape

        points = np.array(corners, dtype=np.int32)
        hull = cv2.convexHull(points)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], WHITE)

        effect = cv2.applyColorMap(frame, colormap)
        frame[mask == WHITE] = effect[mask == WHITE]

        cv2.polylines(frame, [hull], True, FRAME_COLOR, 3)

        x, y, _, _ = cv2.boundingRect(hull)
        cv2.putText(frame, f"PORTAL: {filter_name}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, FRAME_COLOR, 2)

    def run(self):
        """Main loop: process and display each frame, quit on q."""
        try:
            while True:
                ok, frame = self.camera.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)

                corners, touching = self._process_hands(frame, result)   # <- 4.
                self._update_filter(touching)
                self._draw_portal(frame, corners)
                if result.multi_hand_landmarks:
                    self.particles.update(result.multi_hand_landmarks)
                else:
                     self.particles.scatter()
                self.particles.draw(frame)

                cv2.imshow("Portal", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            # Runs even if an error occurs
            self.camera.release()
            self.hands.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    PortalApp().run()