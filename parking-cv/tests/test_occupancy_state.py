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

    def test_ground_contact_wins_over_body_overlap_with_rear_bay(self):
        self.mapping['spaces'][0]['corners']=[[.2,.35],[.8,.35],[.8,.7],[.2,.7]]
        self.mapping['spaces'][1]['corners']=[[.2,.7],[.8,.7],[.8,.95],[.2,.95]]
        box={'box_xyxy':[307,115,717,653]}
        states,_=global_observations(self.mapping,{'one':[box],'two':[box]})
        self.assertEqual(states['B002'],'occupied')
        self.assertNotEqual(states['B001'],'occupied')

    def test_clipped_outlier_cannot_override_intact_view_agreement(self):
        self.mapping['views'].append({'id':'three','yaw':0,'pitch':-20,'fov':90})
        self.mapping['spaces'][0]['corners']=[[.2,.35],[.8,.35],[.8,.7],[.2,.7]]
        self.mapping['spaces'][1]['corners']=[[.2,.7],[.8,.7],[.8,.95],[.2,.95]]
        intact={'box_xyxy':[307,115,717,653],'confidence':.9}
        clipped={'box_xyxy':[307,0,717,520],'confidence':.99}
        states,groups=global_observations(self.mapping,{'one':[intact],'two':[intact],'three':[clipped]})
        self.assertEqual(len(groups),1)
        self.assertEqual(states['B002'],'occupied')
        self.assertNotEqual(states['B001'],'occupied')

    def test_broad_cross_view_box_cannot_merge_two_same_view_cars(self):
        cars=[{'box_xyxy':[240,350,500,620]},{'box_xyxy':[510,350,780,620]}]
        broad={'box_xyxy':[230,330,790,630]}
        _,groups=global_observations(self.mapping,{'one':cars,'two':[broad]})
        self.assertEqual(len(groups),2)
        for group in groups:
            self.assertEqual(len(group['views']),len(set(group['views'])))

    def test_occlusion_holds_occupied_briefly_then_becomes_unknown(self):
        smoother=OccupancyFilter(occlusion_hold_samples=2)
        for _ in range(3):
            smoother.update({'B001':'occupied'})
        self.assertEqual(smoother.update({'B001':'unknown'})['B001'],'occupied')
        self.assertEqual(smoother.update({'B001':'unknown'})['B001'],'occupied')
        self.assertEqual(smoother.update({'B001':'unknown'})['B001'],'unknown')
        smoother.update({'B001':'free'})
        self.assertEqual(smoother.update({'B001':'free'})['B001'],'free')


if __name__=='__main__':
    unittest.main()
