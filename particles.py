import cv2
import mediapipe as mp
import numpy as np

# - - - Constants - - -
PARTICLE_COLOR = (255, 255, 0)  # BGR: cyan
PARTICLE_RADIUS = 2
FOLLOW_SPEED = 0.5     # fraction of the remaining distance covered per frame
JITTER = 0.002         # strength of the random trembling (normalized units)
MAX_HANDS = 2
SPREAD = 0.015         # how far particles sit from the bone line (finger thickness)
SCATTER_PUSH = 0.0005  # strength of the random push per frame when there is no hand
DRAG = 0.95            # fraction of the velocity kept each frame (slows particles down)

# Pairs of landmark ids that are connected on the hand, e.g. (5, 6)
BONES = np.array(list(mp.solutions.hands.HAND_CONNECTIONS))


def hand_to_array(hand):
    """The 21 landmarks of one hand as a (21, 2) table of normalized [x, y] values."""
    return np.array([[lm.x, lm.y] for lm in hand.landmark])


class ParticleCloud:
    """A cloud of particles stored in normalized (0-1) coordinates."""

    def __init__(self, count=300):
        # One row per particle: [x, y], both random numbers between 0 and 1
        self.positions = np.random.rand(count, 2)

        # Each particle belongs to one of the hands (0 or 1)...
        self.hand_ids = np.random.randint(0, MAX_HANDS, count)

        # ...picks a random bone...
        bone_choice = np.random.randint(0, len(BONES), count)
        self.start_ids = BONES[bone_choice, 0]   # first landmark of the bone
        self.end_ids = BONES[bone_choice, 1]     # second landmark of the bone

        # ...and a random spot along it: 0 = start, 1 = end
        self.ratios = np.random.rand(count, 1)
        # A fixed random offset from the bone line, so the cloud has thickness
        self.offsets = np.random.normal(0, SPREAD, (count, 2))
        # Movement per frame, used only while drifting without a hand
        self.velocities = np.zeros((count, 2))

    def update(self, hands):
        """Move every particle a step closer to its target spot on its hand."""
        # All detected hands as a (number_of_hands, 21, 2) table
        points = np.array([hand_to_array(hand) for hand in hands])

        # If only one hand is visible, every particle goes to it
        hand_idx = self.hand_ids % len(hands)

        # Start and end point of each particle's bone, on its own hand
        starts = points[hand_idx, self.start_ids]
        ends = points[hand_idx, self.end_ids]

        # Target = a spot between start and end (linear interpolation)
        targets = starts + (ends - starts) * self.ratios + self.offsets

        # Small random offset every frame, so the cloud looks alive
        jitter = np.random.normal(0, JITTER, self.positions.shape)

        # Cover a fraction of the remaining distance, plus the trembling
        self.positions += (targets - self.positions) * FOLLOW_SPEED + jitter

        # While following a hand, there is no drifting momentum
        self.velocities = np.zeros_like(self.positions)


    def scatter(self):
        """Let the particles drift away like smoke when no hand is visible."""
        # A small random push every frame, so each particle wanders around
        self.velocities += np.random.normal(0, SCATTER_PUSH, self.velocities.shape)

        # Slow down a little each frame, so they don't fly away forever
        self.velocities *= DRAG

        # Move each particle by its velocity
        self.positions += self.velocities


    def draw(self, frame, color=PARTICLE_COLOR):
        """Draw every particle onto the frame as a small filled circle."""
        height, width, _ = frame.shape

        # Convert all positions to pixels at once, without a loop
        pixels = (self.positions * [width, height]).astype(int)

        for x, y in pixels:
            cv2.circle(frame, (x, y), PARTICLE_RADIUS, color, -1)