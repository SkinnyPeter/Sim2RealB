from isaacsim.util.debug_draw import _debug_draw


# Colors as RGBA tuples in [0, 1]
COLOR_RIGHT = (1.0, 0.0, 0.0, 1.0)    # red
COLOR_LEFT  = (0.0, 0.0, 0.55, 1.0)   # dark blue

POINT_SIZE = 10  # pixels


class EEFVisualizer:
    def __init__(self):
        self._draw = _debug_draw.acquire_debug_draw_interface()

    def draw(self, pos_right, pos_left):
        """Draw one point per arm at the given EEF positions.

        Args:
            pos_right: array-like [x, y, z] for the right arm EEF
            pos_left:  array-like [x, y, z] for the left  arm EEF
        """
        r = (float(pos_right[0]), float(pos_right[1]), float(pos_right[2]))
        l = (float(pos_left[0]),  float(pos_left[1]),  float(pos_left[2]))

        self._draw.draw_points([r], [COLOR_RIGHT], [POINT_SIZE])
        self._draw.draw_points([l], [COLOR_LEFT],  [POINT_SIZE])
