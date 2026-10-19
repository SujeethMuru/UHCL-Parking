import sys
from pathlib import Path
import unittest

import numpy as np
import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'parking_cv'))
from parking_editor import ViewTransform
from occupancy import validate_spaces
from space_suggestions import split_row,suggest_spaces


class EditorGeometryTests(unittest.TestCase):
    def test_fit_contains_whole_image(self):
        v=ViewTransform(1024,768)
        for cw,ch in [(800,400),(400,800)]:
            v.fit(cw,ch)
            self.assertGreaterEqual(v.x,0)
            self.assertGreaterEqual(v.y,0)
            self.assertLessEqual(v.x+1024*v.scale,cw)
            self.assertLessEqual(v.y+768*v.scale,ch)

    def test_zoom_preserves_cursor_coordinate(self):
        v=ViewTransform(1024,768)
        v.fit(800,400)
        before=v.to_image(220,180)
        v.zoom(2,220,180)
        np.testing.assert_allclose(v.to_image(220,180),before)

    def test_perspective_row_tiles_without_overlap(self):
        row=[[.1,.2],[.9,.3],[.8,.8],[.2,.7]]
        cells=split_row(row,5)
        validate_spaces(cells)
        self.assertEqual(len(cells),5)
        np.testing.assert_allclose(cells[0][0],row[0],atol=1e-6)
        np.testing.assert_allclose(cells[-1][1],row[1],atol=1e-6)
        np.testing.assert_allclose(cells[1][0],cells[0][1],atol=1e-6)

    def test_blank_image_has_no_suggestions(self):
        self.assertEqual(suggest_spaces(np.zeros((480,640,3),np.uint8)),[])

    def test_painted_lines_generate_reviewable_candidates(self):
        frame=np.full((480,640,3),90,np.uint8)
        for x in (100,230,360):
            cv2.line(frame,(x,250),(x,460),(245,245,245),6)
        candidates=suggest_spaces(frame)
        self.assertGreater(len(candidates),0)
        for candidate in candidates:
            validate_spaces([candidate])
        # Masking the complete vehicle region suppresses those bright lines.
        self.assertEqual(suggest_spaces(frame,[{'box_xyxy':[0,0,640,480]}]),[])


if __name__=='__main__':
    unittest.main()
