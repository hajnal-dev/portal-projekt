import cv2
import numpy as np

PARTICLE_COLOR = (255, 255, 0)  # BGR: cyan
PARTICLE_RADIUS = 2


class ParticleCloud:
    """A cloud of particles stored in normalized (0-1) coordinates."""

    def __init__(self, count=300):
        # One row per particle: [x, y], both random numbers between 0 and 1
        self.positions = np.random.rand(count, 2)

    def draw(self, frame):
        """Draw every particle onto the frame as a small filled circle."""
        height, width, _ = frame.shape

        # Convert all positions to pixels at once, without a loop
        pixels = (self.positions * [width, height]).astype(int)

        for x, y in pixels:
            cv2.circle(frame, (x, y), PARTICLE_RADIUS, PARTICLE_COLOR, -1)