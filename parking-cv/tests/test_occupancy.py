"""Occupancy geometry: boundary contact, background vehicles and invalid spaces."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'parking_cv'))
from occupancy import classify_spaces, validate_spaces


class OccupancyTests(unittest.TestCase):
    def setUp(self):
        self.spaces = [[[.1,.2],[.4,.2],[.4,.8],[.1,.8]],
                       [[.5,.2],[.8,.2],[.8,.8],[.5,.8]]]

    def test_ground_anchor_and_background(self):
        # First box belongs in P1; second overlaps P2 but sits behind the spaces.
        detections=[{'box_xyxy':[15,30,35,75]}, {'box_xyxy':[45,0,85,15]}]
        states=classify_spaces(self.spaces,detections,100,100)
        self.assertEqual([s['status'] for s in states],['occupied','free'])
        self.assertEqual(states[0]['matched_detection_indices'],[0])

    def test_shared_edge_assigns_once(self):
        spaces=[[[0,0],[.5,0],[.5,1],[0,1]],[[.5,0],[1,0],[1,1],[.5,1]]]
        states=classify_spaces(spaces,[{'box_xyxy':[40,10,60,80]}],100,100)
        self.assertEqual(sum(s['status']=='occupied' for s in states),1)

    def test_bad_polygons_rejected(self):
        for spaces in [[], [[[0,0],[1,1],[1,0],[0,1]]],
                       [self.spaces[0],self.spaces[0]],
                       [[[0,0],[2,0],[2,1],[0,1]]]]:
            with self.assertRaises(ValueError):
                validate_spaces(spaces)

    def test_car_extends_beyond_space(self):
        # Car lower body overlaps bay even though its bottom point lies outside.
        spaces=[[[.1,.2],[.6,.2],[.6,.75],[.1,.75]],
                [[.65,.2],[.95,.2],[.95,.75],[.65,.75]]]
        states=classify_spaces(spaces,[{'box_xyxy':[15,25,55,85]}],100,100)
        self.assertEqual([s['status'] for s in states],['occupied','free'])

    def test_small_edge_overlap_does_not_claim_neighbor(self):
        states=classify_spaces(self.spaces,[{'box_xyxy':[39,10,80,90]}],100,100)
        self.assertEqual([s['status'] for s in states],['free','occupied'])


if __name__ == '__main__':
    unittest.main()
