import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_model
import launch_instances


class ExportTests(unittest.TestCase):
    def test_default_and_validation(self):
        self.assertEqual(prepare_model.parse_args([]).batch_size, 1)
        for value in ('0', '-1'):
            with self.assertRaises(SystemExit):
                prepare_model.parse_args(['--batch-size', value])

    def test_export_names_preserve_single_model(self):
        for batch in (1, 2, 4, 8, 16):
            with self.subTest(batch=batch), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / 'yolo26n.pt'
                checkpoint.write_bytes(b'weights')
                original = checkpoint.with_suffix('.onnx')
                original.write_bytes(b'original')
                calls = []
                class FakeModel:
                    def __init__(self, path):
                        self.ckpt_path = path
                    def export(self, **kwargs):
                        calls.append(kwargs)
                        result = Path(self.ckpt_path).with_suffix('.onnx')
                        result.write_bytes(b'exported')
                        return result
                args = prepare_model.parse_args(['--model', str(checkpoint), '--batch-size', str(batch)])
                with patch.object(prepare_model, 'validate_model_shape') as validate:
                    result = prepare_model.export_model(args, FakeModel)
                    validate.assert_called_once()
                expected = original if batch == 1 else checkpoint.with_name(f'yolo26n-b{batch}.onnx')
                self.assertEqual(Path(result['onnx']), expected)
                self.assertEqual(expected.read_bytes(), b'exported')
                self.assertEqual(calls[0]['batch'], batch)
                self.assertFalse(calls[0]['dynamic'])
                if batch > 1:
                    self.assertEqual(original.read_bytes(), b'original')
                self.assertFalse(list(Path(directory).glob('.batch-export-*')))

    def test_failed_export_preserves_models(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'yolo26n.pt'
            checkpoint.write_bytes(b'weights')
            original = checkpoint.with_suffix('.onnx')
            original.write_bytes(b'original')
            target = checkpoint.with_name('yolo26n-b8.onnx')
            target.write_bytes(b'existing-batch8')
            class FakeModel:
                def __init__(self, path):
                    self.ckpt_path = path
                def export(self, **kwargs):
                    Path(self.ckpt_path).with_suffix('.onnx').write_bytes(b'partial')
                    raise RuntimeError('export failed')
            args = prepare_model.parse_args(['--model', str(checkpoint), '--batch-size', '8'])
            with self.assertRaisesRegex(RuntimeError, 'export failed'):
                prepare_model.export_model(args, FakeModel)
            self.assertEqual(original.read_bytes(), b'original')
            self.assertEqual(target.read_bytes(), b'existing-batch8')
            self.assertFalse(list(Path(directory).glob('.batch-export-*')))


class LauncherTests(unittest.TestCase):
    def test_batch_forwarded(self):
        for batch in (1, 8):
            with patch('sys.argv', ['launch_instances.py', '--batch-size', str(batch)]):
                args = launch_instances.parse_args()
            command = launch_instances.child_command(args, 'test', None)
            self.assertEqual(command[command.index('--batch-size') + 1], str(batch))

    def test_invalid_batch_before_launch(self):
        for batch in ('0', '-2'):
            with patch('sys.argv', ['launch_instances.py', '--batch-size', batch]):
                with self.assertRaisesRegex(ValueError, '--batch-size'):
                    launch_instances.main()


if __name__ == '__main__':
    unittest.main()
