from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'parking_cv'))
from occupancy_state import global_observations,OccupancyFilter


class GlobalOccupancyTests(unittest.TestCase):
    def setUp(self):
        self.mapping={'views':[{'id':key,'yaw':0,'pitch':-20,'fov':90} for key in ('one','two')],
                      'spaces':[{'id':'B001','owner':'one','corners':[[.2,.4],[.5,.4],[.5,.9],[.2,.9]]},
                                {'id':'B002','owner':'two','corners':[[.5,.4],[.8,.4],[.8,.9],[.5,.9]]}]}

    def test_same_car_across_owners_fills_only_one_bay(self):
        box={'box_xyxy':[350,300,650,600]}
        states,groups=global_observations(self.mapping,{'one':[box],'two':[box]})
        self.assertEqual(len(groups),1)
        self.assertEqual(list(states.values()).count('occupied'),1)

    def test_two_separate_cars_can_fill_two_bays(self):
        boxes=[{'box_xyxy':[240,350,450,620]},{'box_xyxy':[580,350,780,620]}]
        states,groups=global_observations(self.mapping,{'one':boxes,'two':boxes})
        self.assertEqual(len(groups),2)
        self.assertEqual(list(states.values()).count('occupied'),2)

    def test_short_misses_do_not_flip_confirmed_occupancy(self):
        smoother=OccupancyFilter()
        for _ in range(3):
            state=smoother.update({'B001':'occupied'})
        self.assertEqual(state['B001'],'occupied')
        self.assertEqual(smoother.update({'B001':'free'})['B001'],'occupied')
        self.assertEqual(smoother.update({'B001':'occupied'})['B001'],'occupied')
        smoother.update({'B001':'free'})
        self.assertEqual(smoother.update({'B001':'free'})['B001'],'free')

    def test_short_false_positive_is_not_confirmed(self):
        smoother=OccupancyFilter()
        smoother.update({'B001':'free'})
        smoother.update({'B001':'free'})
        self.assertEqual(smoother.update({'B001':'occupied'})['B001'],'free')
        self.assertEqual(smoother.update({'B001':'free'})['B001'],'free')
        self.assertEqual(smoother.update({'B001':'unknown'})['B001'],'unknown')


if __name__=='__main__':
    unittest.main()
