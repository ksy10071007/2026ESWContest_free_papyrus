import os
import unittest
from unittest import mock


os.environ.setdefault('TORCH_DEVICE', 'cpu')

import config


class TorchDeviceSelectionTest(unittest.TestCase):
    def test_cpu_request_selects_cpu(self):
        self.assertEqual(config.resolve_torch_device('cpu', 0).type, 'cpu')

    @mock.patch.object(config.torch.cuda, 'device_count', return_value=0)
    @mock.patch.object(config.torch.cuda, 'is_available', return_value=False)
    def test_auto_without_cuda_selects_cpu(self, _available, _count):
        self.assertEqual(config.resolve_torch_device('auto', 0).type, 'cpu')

    @mock.patch.object(config, '_probe_cuda_device')
    @mock.patch.object(config.torch.cuda, 'device_count', return_value=1)
    @mock.patch.object(config.torch.cuda, 'is_available', return_value=True)
    def test_auto_with_cuda_selects_cuda(self, _available, _count, _probe):
        self.assertEqual(str(config.resolve_torch_device('auto', 0)), 'cuda:0')

    @mock.patch.object(config.torch.cuda, 'device_count', return_value=0)
    @mock.patch.object(config.torch.cuda, 'is_available', return_value=False)
    def test_cuda_without_cuda_has_diagnostics(self, _available, _count):
        with self.assertRaisesRegex(RuntimeError, 'JetPack/L4T'):
            config.resolve_torch_device('cuda', 0)

    def test_invalid_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid TORCH_DEVICE'):
            config.resolve_torch_device('gpu', 0)

    @mock.patch.object(config.torch.cuda, 'device_count', return_value=1)
    @mock.patch.object(config.torch.cuda, 'is_available', return_value=True)
    def test_invalid_cuda_index_is_rejected(self, _available, _count):
        with self.assertRaisesRegex(RuntimeError, 'GPU index'):
            config.resolve_torch_device('cuda', 2)


if __name__ == '__main__':
    unittest.main()
