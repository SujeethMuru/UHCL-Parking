from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'parking_cv'))
from parking_map import register_spaces,combined_counts,sphere_mask,reproject,validate_map,cut_off,transfer_space,rays,replace_space
from multi_view import process_sample
from view_gallery import gallery_views,panorama_overlay,editor_references


class SharedMapTests(unittest.TestCase):
    def setUp(self):
        self.view={'id':'one','yaw':0,'pitch':-20,'fov':90}
        self.other={**self.view,'id':'two'}
        self.mapping={'views':[self.view,self.other],'spaces':[]}
        self.bay=[[.35,.4],[.65,.4],[.65,.8],[.35,.8]]

    def test_duplicate_physical_bay_in_other_view_is_not_added(self):
        first,_=register_spaces(self.mapping,'one',[self.bay])
        second,skipped=register_spaces(first,'two',[self.bay])
        self.assertEqual(len(second['spaces']),1)
        self.assertEqual(skipped,[first['spaces'][0]['id']])
        self.assertEqual(second['spaces'][0]['owner'],'one')

    def test_duplicate_with_different_camera_angle(self):
        self.other['yaw']=15
        first,_=register_spaces(self.mapping,'one',[self.bay])
        shifted=reproject(self.bay,self.view,self.other)
        self.assertIsNotNone(shifted)
        second,skipped=register_spaces(first,'two',[shifted])
        self.assertEqual(len(second['spaces']),1)
        self.assertEqual(len(skipped),1)

    def test_edit_retains_bay_identity(self):
        first,_=register_spaces(self.mapping,'one',[self.bay])
        revised=np.asarray(self.bay)+[.002,0]
        second,_=register_spaces(first,'one',[revised.tolist()])
        self.assertEqual(first['spaces'][0]['id'],second['spaces'][0]['id'])

    def test_complete_adjacent_view_replaces_edge_fragment(self):
        self.other['yaw']=-25
        edge=[[0,.4],[.3,.4],[.3,.8],[0,.8]]
        first,_=register_spaces(self.mapping,'one',[edge])
        complete=reproject(edge,self.view,self.other)
        self.assertIsNotNone(complete)
        self.assertFalse(cut_off(complete))
        second,skipped=register_spaces(first,'two',[complete])
        self.assertEqual(skipped,[])
        self.assertEqual(len(second['spaces']),1)
        self.assertEqual(second['spaces'][0]['id'],first['spaces'][0]['id'])
        self.assertEqual(second['spaces'][0]['owner'],'two')

    def test_seam_and_rotation_do_not_expand_duplicate_mask(self):
        a=sphere_mask(self.bay,{**self.view,'yaw':180})
        b=sphere_mask(self.bay,{**self.view,'yaw':-180})
        np.testing.assert_array_equal(a,b)
        self.assertLess(np.count_nonzero(a)/a.size,.1)

    def test_missing_observation_is_unknown_not_free(self):
        mapping,_=register_spaces(self.mapping,'one',[self.bay])
        totals=combined_counts(mapping,{})
        self.assertEqual((totals['mapped'],totals['occupied'],totals['estimated_free'],totals['unknown']),(1,0,0,1))

    def test_duplicate_global_ids_rejected(self):
        mapping,_=register_spaces(self.mapping,'one',[self.bay])
        mapping['spaces'].append(deepcopy(mapping['spaces'][0]))
        with self.assertRaises(ValueError):
            validate_map(mapping)
        with self.assertRaises(ValueError):
            combined_counts(mapping,{})

    def test_combined_sample_counts_bay_once_with_car_in_both_views(self):
        mapping,_=register_spaces(self.mapping,'one',[self.bay])
        blank=np.zeros((768,1024,3),np.uint8)
        car={'box_xyxy':[400,300,600,580]}
        grid,counts,details=process_sample(mapping,{'one':blank,'two':blank},
                                         {'one':[car],'two':[car]},20)
        self.assertEqual(counts['mapped'],1)
        self.assertEqual(counts['occupied'],1)
        self.assertEqual(len(details),2)
        self.assertEqual(grid.shape,(604,1280,3))

    def test_edge_cut_bay_is_unknown(self):
        edge=[[0,.4],[.3,.4],[.3,.8],[0,.8]]
        mapping,_=register_spaces(self.mapping,'one',[edge])
        blank=np.zeros((768,1024,3),np.uint8)
        _,counts,_=process_sample(mapping,{'one':blank},{'one':[]},0)
        self.assertTrue(cut_off(edge))
        self.assertEqual(counts['unknown'],1)
        self.assertEqual(counts['estimated_free'],0)

    def test_partial_reprojection_clips_instead_of_disappearing(self):
        target={**self.view,'yaw':35}
        self.assertIsNone(reproject(self.bay,self.view,target))
        clipped=reproject(self.bay,self.view,target,allow_partial=True)
        self.assertIsNotNone(clipped)
        self.assertGreaterEqual(len(clipped),3)
        self.assertTrue((np.asarray(clipped)>=0).all())
        self.assertTrue((np.asarray(clipped)<=1).all())

    def test_gallery_is_stable_and_preserves_old_markings(self):
        mapping,_=register_spaces(self.mapping,'one',[self.bay])
        saved=deepcopy(mapping['spaces'])
        views=gallery_views(mapping)
        size=len(mapping['views'])
        again=gallery_views(mapping)
        self.assertEqual(len(views),8)
        self.assertEqual(len(mapping['views']),size)
        self.assertEqual(views,again)
        self.assertEqual(mapping['spaces'],saved)
        image=panorama_overlay(np.zeros((256,512,3),np.uint8),mapping)
        self.assertEqual(image.shape,(640,1280,3))
        self.assertGreater(np.count_nonzero(image),0)

    def test_editor_shows_other_view_without_adding_editable_spaces(self):
        mapping,_=register_spaces(self.mapping,'one',[self.bay])
        before=deepcopy(mapping)
        self.other['yaw']=35
        references=editor_references(mapping,self.other)
        self.assertEqual(len(references),1)
        self.assertIn(mapping['spaces'][0]['id'],references[0]['label'])
        self.assertEqual(editor_references(mapping,self.view),[])
        self.assertEqual(mapping,before)

    def test_explicit_transfer_keeps_identity_and_count(self):
        self.other['yaw']=15
        mapping,_=register_spaces(self.mapping,'one',[self.bay])
        before=deepcopy(mapping)
        key=mapping['spaces'][0]['id']
        moved=transfer_space(mapping,key,'two')
        self.assertEqual(mapping,before)
        self.assertEqual(len(moved['spaces']),1)
        self.assertEqual(moved['spaces'][0]['id'],key)
        self.assertEqual(moved['spaces'][0]['owner'],'two')
        np.testing.assert_allclose(rays(self.bay,self.view),
                                   rays(moved['spaces'][0]['corners'],self.other),atol=1e-10)
        self.assertEqual(combined_counts(moved,{key:'occupied'})['occupied'],1)

    def test_transfer_rejects_partial_outline_without_changing_map(self):
        self.other['yaw']=60
        mapping,_=register_spaces(self.mapping,'one',[self.bay])
        before=deepcopy(mapping)
        with self.assertRaises(ValueError):
            transfer_space(mapping,mapping['spaces'][0]['id'],'two')
        self.assertEqual(mapping,before)

    def test_redraw_outside_original_view_keeps_same_id(self):
        self.other['yaw']=180
        mapping,_=register_spaces(self.mapping,'one',[self.bay])
        before=deepcopy(mapping)
        references=editor_references(mapping,self.other)
        self.assertEqual(len(references),1)
        self.assertFalse(references[0]['transferable'])
        self.assertIsNone(references[0]['corners'])
        key=mapping['spaces'][0]['id']
        revised=replace_space(mapping,key,'two',self.bay)
        self.assertEqual(mapping,before)
        self.assertEqual(len(revised['spaces']),1)
        self.assertEqual(revised['spaces'][0]['id'],key)
        self.assertEqual(revised['spaces'][0]['owner'],'two')

    def test_invalid_redraw_retains_original(self):
        mapping,_=register_spaces(self.mapping,'one',[self.bay])
        before=deepcopy(mapping)
        with self.assertRaises(ValueError):
            replace_space(mapping,mapping['spaces'][0]['id'],'two',[[0,0]]*4)
        self.assertEqual(mapping,before)

    def test_playback_shows_eight_tiles_but_counts_old_owner_once(self):
        mapping,_=register_spaces(self.mapping,'one',[self.bay])
        views=gallery_views(mapping)
        blank=np.zeros((768,1024,3),np.uint8)
        frames={v['id']:blank for v in mapping['views']}
        detections={v['id']:[] for v in mapping['views']}
        detections['one']=[{'box_xyxy':[400,300,600,580]}]
        grid,total,_=process_sample(mapping,frames,detections,0,display_views=views)
        self.assertEqual(grid.shape,(1118,2560,3))
        self.assertEqual(total['mapped'],1)
        self.assertEqual(total['occupied'],1)


if __name__=='__main__':
    unittest.main()
