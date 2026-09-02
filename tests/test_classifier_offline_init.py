import os
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


os.environ.setdefault('TORCH_DEVICE', 'cpu')

from modules import classifier


class TinyEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.ModuleList([nn.Identity()])
        self.classifier = nn.Sequential(nn.Identity(), nn.Linear(3, 3))


class ClassifierOfflineInitTest(unittest.TestCase):
    @mock.patch.object(classifier, 'GradCAM')
    @mock.patch.object(classifier.models, 'efficientnet_b0')
    def test_initialization_disables_external_weights(self, efficientnet_b0, _grad_cam):
        efficientnet_b0.side_effect = lambda **_kwargs: TinyEfficientNet()
        checkpoint_model = TinyEfficientNet()
        checkpoint_model.classifier[1] = nn.Linear(3, 5)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / 'classifier.pth'
            torch.save(checkpoint_model.state_dict(), checkpoint_path)
            classifier.DiseaseClassifier(str(checkpoint_path), num_classes=5)

        efficientnet_b0.assert_called_once_with(weights=None)

    def test_extracts_wrapped_dataparallel_state_dict(self):
        wrapped = {
            'state_dict': OrderedDict({
                'module.layer.weight': torch.ones(1),
                'module.layer.bias': torch.zeros(1),
            })
        }
        extracted = classifier.DiseaseClassifier._extract_state_dict(wrapped)
        self.assertEqual(list(extracted), ['layer.weight', 'layer.bias'])

    def test_rejects_unsupported_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, 'Unsupported classifier checkpoint type'):
            classifier.DiseaseClassifier._extract_state_dict(['not', 'a', 'mapping'])


if __name__ == '__main__':
    unittest.main()
