from omni.debugdraw import get_debug_draw_interface
import carb


# Colors as RGBA floats in [0, 1]
COLOR_RIGHT = carb.Float4(1.0, 0.0, 0.0, 1.0)       # red
COLOR_LEFT  = carb.Float4(0.0, 0.0, 0.55, 1.0)      # dark blue

SPHERE_RADIUS = 0.02  # metres


class EEFVisualizer:
    def __init__(self):
        self._draw = get_debug_draw_interface()

    def draw(self, pos_right, pos_left):
        """Draw one sphere per arm at the given EEF positions.

        Args:
            pos_right: array-like [x, y, z] for the right arm EEF
            pos_left:  array-like [x, y, z] for the left  arm EEF
        """
        r = carb.Float3(float(pos_right[0]), float(pos_right[1]), float(pos_right[2]))
        l = carb.Float3(float(pos_left[0]),  float(pos_left[1]),  float(pos_left[2]))

        self._draw.draw_sphere(r, SPHERE_RADIUS, COLOR_RIGHT)
        self._draw.draw_sphere(l, SPHERE_RADIUS, COLOR_LEFT)
