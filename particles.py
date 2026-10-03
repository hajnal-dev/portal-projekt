import cv2
import mediapipe as mp
import numpy as np

PARTICLE_COLOR = (255, 255, 0)  # BGR: cyan
PARTICLE_RADIUS = 2
FOLLOW_SPEED = 0.1     # fraction of the remaining distance covered per frame
JITTER = 0.002         # strength of the random trembling (normalized units)

# Pairs of landmark ids that are connected on the hand, e.g. (5, 6)
BONES = np.array(list(mp.solutions.hands.HAND_CONNECTIONS))


class ParticleCloud:
    """A cloud of particles stored in normalized (0-1) coordinates."""

    def __init__(self, count=300):
        # One row per particle: [x, y], both random numbers between 0 and 1
        self.positions = np.random.rand(count, 2)

        # Each particle picks a random bone...
        bone_choice = np.random.randint(0, len(BONES), count)
        self.start_ids = BONES[bone_choice, 0]   # first landmark of the bone
        self.end_ids = BONES[bone_choice, 1]  # second landmark of the bone

        # ...and a random spot along it: 0 = start, 1 = end
        self.ratios = np.random.rand(count, 1)

    def update(self, hand):
        """Move every particle a step closer to its target spot on the hand."""
        # The 21 landmarks as a (21, 2) table of normalized [x, y] values
        points = np.array([[lm.x, lm.y] for lm in hand.landmark])

        # Start and end point of each particle's bone
        starts = points[self.start_ids]
        ends = points[self.end_ids]

        # Target = a spot between start and end (linear interpolation)
        targets = starts + (ends- starts) * self.ratios

        # Small random offset every frame, so the cloud looks alive
        jitter = np.random.normal(0, JITTER, self.positions.shape)

        # Cover a fraction of the remaining distance, plus the trembling
        self.positions += (targets - self.positions) * FOLLOW_SPEED + jitter

    def draw(self, frame):
        """Draw every particle onto the frame as a small filled circle."""
        height, width, _ = frame.shape

        # Convert all positions to pixels at once, without a loop
        pixels = (self.positions * [width, height]).astype(int)

        for x, y in pixels:
            cv2.circle(frame, (x, y), PARTICLE_RADIUS, PARTICLE_COLOR, -1)