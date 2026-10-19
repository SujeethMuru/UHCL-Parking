"""Geometry checks independent of detector weights and private footage."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'parking_cv'))
from perspective import projection_maps


class ProjectionTests(unittest.TestCase):
    def test_cardinal_directions(self):
        for yaw, expected_x in [(0, 200), (90, 300), (-90, 100)]:
            x, y = projection_maps(400, 200, yaw, pitch=0, width=3, height=3)
            self.assertAlmostEqual(float(x[1,1]), expected_x + 0.5, places=4)
            self.assertAlmostEqual(float(y[1,1]), 99.5, places=4)

    def test_pitch_and_seam(self):
        x, y = projection_maps(400, 200, 180, pitch=45, width=3, height=3)
        self.assertAlmostEqual(float(y[1,1]), 49.5, places=4)
        # Rays on opposite sides of the seam sample opposite ends of the panorama.
        self.assertGreater(float(x[1,0]), 300)
        self.assertLess(float(x[1,2]), 100)

    def test_poles_are_clamped(self):
        x, y = projection_maps(400, 200, 0, pitch=90)
        self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())
        self.assertGreaterEqual(float(y.min()), 0)
        self.assertLessEqual(float(y.max()), 199)


if __name__ == '__main__':
    unittest.main()
