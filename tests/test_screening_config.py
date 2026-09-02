import json
import os
import unittest


class ScreeningConfigTest(unittest.TestCase):
    def setUp(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(root, 'config', 'screening_modalities.json')
        with open(config_path, 'r', encoding='utf-8') as config_file:
            self.config = json.load(config_file)

    def test_modality_order_and_ids(self):
        self.assertEqual(
            [item['id'] for item in self.config['modalities']],
            ['eye', 'skin', 'scalp'],
        )

    def test_only_eye_model_is_ready(self):
        statuses = {
            item['id']: item['model_status']
            for item in self.config['modalities']
        }
        self.assertEqual(statuses['eye'], 'ready')
        self.assertEqual(statuses['skin'], 'not_configured')
        self.assertEqual(statuses['scalp'], 'not_configured')

    def test_eye_classes_follow_model_index_order(self):
        by_id = {item['id']: item for item in self.config['modalities']}
        self.assertEqual(
            by_id['eye']['classes'],
            ['Conjunctivitis', 'Eyelid', 'Cataract', 'Normal', 'Uveitis'],
        )

    def test_placeholder_classes_are_configurable(self):
        by_id = {item['id']: item for item in self.config['modalities']}
        self.assertEqual(by_id['skin']['classes'], ['ex1', 'ex2'])
        self.assertEqual(by_id['scalp']['classes'], ['ex1', 'ex2'])

    def test_camera_roles_match_capture_hardware(self):
        by_id = {item['id']: item for item in self.config['modalities']}
        self.assertEqual(by_id['eye']['camera_role'], 'webcam')
        self.assertEqual(by_id['skin']['camera_role'], 'webcam')
        self.assertEqual(by_id['scalp']['camera_role'], 'usb_microscope')


if __name__ == '__main__':
    unittest.main()
