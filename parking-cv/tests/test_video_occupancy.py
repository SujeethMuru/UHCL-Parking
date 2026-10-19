import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'parking_cv'))
from video_occupancy import sample_indices,source_signature,annotate,make_projector,view_defaults


class VideoOccupancyTests(unittest.TestCase):
    def test_views_have_separate_calibration_and_outputs(self):
        first=view_defaults('first')
        second=view_defaults('second')
        self.assertEqual(first[0],300)
        self.assertEqual(second[0],120)
        self.assertNotEqual(first[1],second[1])
        self.assertNotEqual(first[2],second[2])
        self.assertEqual(first[1],Path('config/video-spaces.local.json'))

    def test_sampling_respects_bounds_and_limit(self):
        self.assertEqual(list(sample_indices(30,90,1)),[0,30,60])
        self.assertEqual(list(sample_indices(30,90,.5,start=.5,end=2)),[15,30,45])
        self.assertEqual(list(sample_indices(30,90,1,limit=2)),[0,30])

    def test_bad_sampling_arguments_rejected(self):
        for kwargs in [{'interval':0},{'interval':.001},{'interval':1,'start':-1},
                       {'interval':1,'end':4},{'interval':1,'limit':0}]:
            with self.assertRaises(ValueError):
                list(sample_indices(30,90,**kwargs))

    def test_vehicle_enters_and_leaves_marked_bay(self):
        frame=np.zeros((768,1024,3),np.uint8)
        spaces=[[[.1,.2],[.4,.2],[.4,.8],[.1,.8]]]
        statuses=[]
        for detections in [[],[{'box_xyxy':[130,220,370,580]}],[]]:
            output,states=annotate(frame,detections,spaces,0,'test')
            statuses.append(states[0]['status'])
            self.assertEqual(output.shape,(846,1024,3))
        self.assertEqual(statuses,['free','occupied','free'])

    def test_uncalibrated_video_has_no_space_counts(self):
        _,states=annotate(np.zeros((768,1024,3),np.uint8),[],[],0,'test')
        self.assertEqual(states,[])

    def test_projector_keeps_view_dimensions(self):
        result=make_projector(400,200,300,-20,90)(np.full((200,400,3),80,np.uint8))
        self.assertEqual(result.shape,(768,1024,3))
        self.assertTrue((result==80).all())

    def test_calibration_signature_detects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.mp4'
            path.write_bytes(b'a'*100)
            first=source_signature(path)
            path.write_bytes(b'b'*100)
            self.assertNotEqual(first,source_signature(path))


if __name__=='__main__':
    unittest.main()
