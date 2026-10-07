"""ASCII art filter: redraws an image with characters, terminal style."""

import cv2
import numpy as np

# - - - Constants - - -
CHARS = " .:-=+*#%@"  # from dark to bright: brighter pixel → "denser" character
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 0.5  # smaller = more, tinier characters (more detail, harder to read)
TEXT_COLOR = (0, 220, 0)  # BGR: green, matches the terminal look
BOOST_CONTRAST = True  # stretch the brightness range so the picture is clearer
SHADING = True  # dark areas get dimmer characters, not only "thinner" ones
COLORED = False  # True: each character takes the color of the image under it


def _make_glyphs():
    """Draw every character once into its own small black-and-white tile.

    Returns a (number of chars, cell height, cell width) array, where 255 = ink.
    """
    # The widest/tallest character decides the cell size, so nothing gets cut off
    sizes = [cv2.getTextSize(char, FONT, FONT_SCALE, 1) for char in CHARS]
    cell_w = max(w for (w, h), baseline in sizes)
    cell_h = max(h + baseline for (w, h), baseline in sizes)

    glyphs = []
    for char, ((w, h), baseline) in zip(CHARS, sizes):
        tile = np.zeros((cell_h, cell_w), dtype=np.uint8)
        x = (cell_w - w) // 2  # center the character horizontally
        cv2.putText(tile, char, (x, cell_h - baseline), FONT, FONT_SCALE, 255, 1, cv2.LINE_AA)
        glyphs.append(tile)
    return np.array(glyphs)


# Built only once, when the module is imported, not on every frame
GLYPHS = _make_glyphs()
CELL_H, CELL_W = GLYPHS.shape[1:]


def _cells_to_pixels(cells, size):
    """Blow up a per-cell image so every cell covers its CELL_W x CELL_H block."""
    return cv2.resize(cells, size, interpolation=cv2.INTER_NEAREST)


def _vivid_colors(small):
    """The image's colors at full brightness (brightness is shown by the characters)."""
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = 255  # V channel = brightness → max
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def to_ascii(image):
    """Return a copy of the image redrawn as ASCII characters on black."""
    height, width = image.shape[:2]
    rows, cols = height // CELL_H, width // CELL_W
    grid_size = (cols * CELL_W, rows * CELL_H)  # (width, height), as OpenCV wants it

    # 1) One brightness value per character cell
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    if BOOST_CONTRAST:
        small = cv2.equalizeHist(small)

    # 2) Brightness (0-255) → character index (0 ... len(CHARS)-1)
    indices = small.astype(np.int32) * len(CHARS) // 256

    # 3) Look up the tile of every cell at once: shape (rows, cols, CELL_H, CELL_W)
    tiles = GLYPHS[indices]

    # 4) Lay the tiles side by side into one big image: (rows*CELL_H, cols*CELL_W)
    ink = tiles.transpose(0, 2, 1, 3).reshape(rows * CELL_H, cols * CELL_W)

    # 5) Shading: scale the ink by the cell's brightness (dark cell → dim character)
    if SHADING:
        ink = cv2.multiply(ink, _cells_to_pixels(small, grid_size), scale=1 / 255)

    # 6) Color it
    result = np.zeros_like(image)
    if COLORED:
        small_color = cv2.resize(image, (cols, rows), interpolation=cv2.INTER_AREA)
        colors = _cells_to_pixels(_vivid_colors(small_color), grid_size)
        ink_3ch = cv2.merge([ink, ink, ink])
        result[: grid_size[1], : grid_size[0]] = cv2.multiply(colors, ink_3ch, scale=1 / 255)
    else:
        # each BGR channel = ink scaled by that channel of TEXT_COLOR
        channels = [cv2.convertScaleAbs(ink, alpha=c / 255) for c in TEXT_COLOR]
        result[: grid_size[1], : grid_size[0]] = cv2.merge(channels)
    return result


if __name__ == "__main__":
    # Quick test without the portal: the whole webcam image as ASCII
    camera = cv2.VideoCapture(0)
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        cv2.imshow("ASCII test", to_ascii(cv2.flip(frame, 1)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()