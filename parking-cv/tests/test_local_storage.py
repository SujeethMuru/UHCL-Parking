import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'parking_cv'))
from local_storage import save_calibration,write_artifact


class StorageTests(unittest.TestCase):
    def test_unchanged_reference_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            frame=np.zeros((40,60,3),np.uint8)
            cv2.imwrite(str(root/'reference.png'),frame)
            before=(root/'reference.png').stat().st_mtime_ns
            with patch('local_storage.save_image',side_effect=AssertionError('Must not rewrite reference')):
                result=save_calibration(frame,[[[0,0],[1,0],[1,1],[0,1]]],
                                        root/'spaces.local.json',root)
            self.assertEqual((root/'reference.png').stat().st_mtime_ns,before)
            self.assertEqual(len(json.loads(result.read_text())['spaces']),1)
            self.assertEqual(len(list(root.glob('*.recovery.*.local.json'))),1)

    def test_locked_existing_file_uses_new_name(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'spaces.local.json'
            target.write_bytes(b'previous')
            replace=os.replace
            def locked(source,destination):
                if Path(destination)==target:
                    raise PermissionError('Simulated Windows viewer lock')
                return replace(source,destination)
            with patch('local_storage.os.replace',side_effect=locked):
                saved=write_artifact(target,b'new data')
            self.assertNotEqual(saved,target)
            self.assertTrue(saved.name.endswith('.local.json'))
            self.assertEqual(saved.read_bytes(),b'new data')
            self.assertEqual(target.read_bytes(),b'previous')
            self.assertEqual(list(Path(directory).glob('.parking-*')),[])

    def test_main_config_failure_keeps_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            target=root/'spaces.local.json'
            real_write=write_artifact
            def fail_config(path,data):
                if Path(path)==target:
                    raise OSError('Simulated config failure')
                return real_write(path,data)
            with patch('local_storage.write_artifact',side_effect=fail_config):
                saved=save_calibration(np.zeros((40,60,3),np.uint8),[],target,root)
            self.assertIn('recovery',saved.name)
            payload=json.loads(saved.read_text())
            self.assertTrue(Path(payload['reference_image']).exists())


if __name__=='__main__':
    unittest.main()
