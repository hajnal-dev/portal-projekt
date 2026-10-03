"""Webcam hand-gesture portal: a filtered frame between two hands and a particle cloud."""

import time

import cv2
import mediapipe as mp
import numpy as np

from particles import ParticleCloud

mp_hands = mp.solutions.hands  # hand detection module
mp_draw = mp.solutions.drawing_utils  # helper for drawing landmarks

# - - - Constants - - -
SHOW_SKELETON = False  # set to True to see the hand landmarks (useful for debugging)
THUMB_TIP = 4  # index of the thumb tip among the 21 landmarks
INDEX_TIP = 8  # index of the index finger tip
TOUCH_THRESHOLD = 0.06  # below this distance = touch (normalized units)
WHITE = 255  # in the mask: "inside"
PARTICLE_COUNT = 1500  # number of particles in the cloud

# - - - Look & feel - - -
# x1, y1, x2, y2 in normalized (0-1) coordinates, top-right
BUTTON_AREA = (0.72, 0.03, 0.97, 0.11)
FRAME_COLOR = (0, 150, 0)  # BGR: darker green
FRAME_THICKNESS = 2  # line width of the portal frame in pixels
FONT = cv2.FONT_HERSHEY_PLAIN  # thin, terminal-like font
TEXT_THICKNESS = 1  # line width of the text in pixels
SCANLINES = False  # darken every few rows, like an old CRT monitor
SCANLINE_GAP = 3  # every 3rd row is darkened
SCANLINE_DARKNESS = 0.6  # darkened rows keep 60% of their brightness

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


def blinking_cursor():
    """Return '_' or ' ', switching twice per second, like a terminal cursor."""
    return "_" if int(time.time() * 2) % 2 == 0 else " "


def add_scanlines(frame):
    """Darken every SCANLINE_GAP-th row of the frame."""
    frame[::SCANLINE_GAP] = (frame[::SCANLINE_GAP] * SCANLINE_DARKNESS).astype(np.uint8)


class PortalApp:
    """Webcam portal effect: draws a frame between two hands with a color filter inside."""

    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

        # State that persists from frame to frame
        self.current_filter = 0
        self.was_touching = False
        self.was_on_button = False
        self.particles = ParticleCloud(PARTICLE_COUNT)

    def _is_touching(self, thumb, index):
        """Return True if the thumb tip and index finger tip are close enough."""
        distance = np.linalg.norm(
            np.array([thumb.x, thumb.y]) - np.array([index.x, index.y])
        )
        return distance <= TOUCH_THRESHOLD

    def _process_hands(self, frame, result):
        """Optionally draw the hands; return the corner points and whether there was a touch."""
        height, width, _ = frame.shape
        corners = []
        touching = False

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                if SHOW_SKELETON:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                thumb = hand.landmark[THUMB_TIP]
                index = hand.landmark[INDEX_TIP]

                if self._is_touching(thumb, index):
                    touching = True

                corners.append(to_pixel(thumb, width, height))
                corners.append(to_pixel(index, width, height))

        return corners, touching

    def _update_filter(self, touching):
        """Switch to the next filter at the moment a touch begins."""
        if touching and not self.was_touching:
            self.current_filter = (self.current_filter + 1) % len(FILTERS)
            self.particles.randomize_colors()  # new random colors for the particles
        self.was_touching = touching

    def _draw_portal(self, frame, corners):
        """If all 4 corners are present, draw the portal with filter and label."""
        if len(corners) != 4:
            return  # no two hands → nothing to draw

        filter_name, colormap = FILTERS[self.current_filter]
        height, width, _ = frame.shape

        points = np.array(corners, dtype=np.int32)
        hull = cv2.convexHull(points)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], WHITE)

        effect = cv2.applyColorMap(frame, colormap)
        frame[mask == WHITE] = effect[mask == WHITE]

        cv2.polylines(frame, [hull], True, FRAME_COLOR, FRAME_THICKNESS)

        x, y, _, _ = cv2.boundingRect(hull)
        cv2.putText(
            frame,
            f"> PORTAL: {filter_name}{blinking_cursor()}",
            (x, y - 10),
            FONT,
            1.2,
            FRAME_COLOR,
            TEXT_THICKNESS,
        )

    def _draw_hud(self, frame, hand_count):
        """Terminal-style status lines in the top-left corner."""
        filter_name, _ = FILTERS[self.current_filter]
        lines = [
            "> HAND_TRACKING: ONLINE",
            f"> HANDS_DETECTED: {hand_count}",
            f"> FILTER: {filter_name}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                frame, line, (10, 25 + i * 22), FONT, 1.1, FRAME_COLOR, TEXT_THICKNESS
            )

    def _finger_on_button(self, result):
        """Return True if any index fingertip is inside the button area."""
        if not result.multi_hand_landmarks:
            return False

        x1, y1, x2, y2 = BUTTON_AREA
        for hand in result.multi_hand_landmarks:
            tip = hand.landmark[INDEX_TIP]
            if x1 <= tip.x <= x2 and y1 <= tip.y <= y2:
                return True
        return False

    def _update_button(self, on_button):
        """Toggle the particle colors at the moment a finger enters the button."""
        if on_button and not self.was_on_button:
            self.particles.toggle_rainbow()
        self.was_on_button = on_button

    def _draw_button(self, frame, on_button):
        """Draw the button: filled while a finger is on it, outlined otherwise."""
        height, width, _ = frame.shape
        x1, y1, x2, y2 = BUTTON_AREA
        top_left = (int(x1 * width), int(y1 * height))
        bottom_right = (int(x2 * width), int(y2 * height))

        thickness = -1 if on_button else 1  # -1 = filled rectangle
        # black text on the filled button
        text_color = (0, 0, 0) if on_button else FRAME_COLOR
        state = "ON" if self.particles.rainbow else "OFF"

        cv2.rectangle(frame, top_left, bottom_right, FRAME_COLOR, thickness)
        cv2.putText(
            frame,
            f"RAINBOW: {state}",
            (top_left[0] + 8, bottom_right[1] - 10),
            FONT,
            1.1,
            text_color,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

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

                corners, touching = self._process_hands(frame, result)
                self._update_filter(touching)
                on_button = self._finger_on_button(result)
                self._update_button(on_button)
                self._draw_portal(frame, corners)

                if result.multi_hand_landmarks:
                    self.particles.update(result.multi_hand_landmarks)
                else:
                    self.particles.scatter()
                self.particles.draw(frame)

                self._draw_hud(frame, len(corners) // 2)
                self._draw_button(frame, on_button)

                if SCANLINES:
                    add_scanlines(frame)

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