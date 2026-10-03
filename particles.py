"""Particle cloud that follows the shape of the hands and drifts away when they disappear."""

import cv2
import mediapipe as mp
import numpy as np

# - - - Constants - - -
PARTICLE_COLOR = (0, 220, 0)  # BGR: green (used when rainbow is off)
PARTICLE_SIZE = 2  # side length of the square particles in pixels
FOLLOW_SPEED = 0.7  # fraction of the remaining distance covered per frame
JITTER = 0.002  # strength of the random trembling (normalized units)
MAX_HANDS = 2  # particles are split between at most this many hands
SPREAD = 0.015  # how far particles sit from the bone line (finger thickness)
SCATTER_PUSH = 0.0005  # strength of the random push per frame when there is no hand
DRAG = 0.95  # fraction of the velocity kept each frame (slows particles down)

# Pairs of landmark ids that are connected on the hand, e.g. (5, 6)
BONES = np.array(list(mp.solutions.hands.HAND_CONNECTIONS))


def hand_to_array(hand):
    """The 21 landmarks of one hand as a (21, 2) table of normalized [x, y] values."""
    return np.array([[lm.x, lm.y] for lm in hand.landmark])


class ParticleCloud:
    """A cloud of particles stored in normalized (0-1) coordinates."""

    def __init__(self, count=300, rainbow=False):
        # One row per particle: [x, y], both random numbers between 0 and 1
        self.positions = np.random.rand(count, 2)

        # Each particle belongs to one of the hands (0 or 1)...
        self.hand_ids = np.random.randint(0, MAX_HANDS, count)

        # ...picks a random bone...
        bone_choice = np.random.randint(0, len(BONES), count)
        self.start_ids = BONES[bone_choice, 0]  # first landmark of the bone
        self.end_ids = BONES[bone_choice, 1]  # second landmark of the bone

        # ...and a random spot along it: 0 = start, 1 = end
        self.ratios = np.random.rand(count, 1)

        # A fixed random offset from the bone line, so the cloud has thickness
        self.offsets = np.random.normal(0, SPREAD, (count, 2))

        # Movement per frame, used only while drifting without a hand
        self.velocities = np.zeros((count, 2))

        # Whether the particles are rainbow colored or all green (toggled at runtime)
        self.rainbow = rainbow

        # Every particle gets its color
        self.randomize_colors()

    def randomize_colors(self):
        """Give every particle a color: random vivid ones, or all the same green."""
        count = len(self.positions)

        if not self.rainbow:
            self.colors = [PARTICLE_COLOR] * count
            return

        # cvtColor works on images, so we build a tiny "image":
        # count rows, 1 column, 3 channels
        hsv = np.zeros((count, 1, 3), dtype=np.uint8)
        # hue: any color of the wheel (0-179 in OpenCV)
        hsv[:, 0, 0] = np.random.randint(0, 180, count)
        hsv[:, 0, 1] = 255  # saturation: fully vivid, no grayness
        hsv[:, 0, 2] = 255  # value: full brightness

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.colors = bgr[:, 0, :].tolist()

    def toggle_rainbow(self):
        """Switch between rainbow and plain green particles."""
        self.rainbow = not self.rainbow
        self.randomize_colors()

    def update(self, hands):
        """Move every particle a step closer to its target spot on its hand."""
        # All detected hands as a (number_of_hands, 21, 2) table
        points = np.array([hand_to_array(hand) for hand in hands])

        # If only one hand is visible, every particle goes to it
        hand_idx = self.hand_ids % len(hands)

        # Start and end point of each particle's bone, on its own hand
        starts = points[hand_idx, self.start_ids]
        ends = points[hand_idx, self.end_ids]

        # Target = a spot between start and end (linear interpolation), plus thickness
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

    def draw(self, frame):
        """Draw every particle onto the frame as a small filled square."""
        height, width, _ = frame.shape

        # Convert all positions to pixels at once, without a loop
        pixels = (self.positions * [width, height]).astype(int)

        # Walk through the positions and the colors side by side
        for (x, y), color in zip(pixels, self.colors):
            cv2.rectangle(
                frame, (x, y), (x + PARTICLE_SIZE, y + PARTICLE_SIZE), color, -1
            )