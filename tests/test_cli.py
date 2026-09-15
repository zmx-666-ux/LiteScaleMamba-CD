import contextlib
import io
import unittest

import train
import test


class CommandLineTests(unittest.TestCase):
    def test_training_requires_separate_train_and_validation_splits(self):
        parser = train.build_parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args([
            "--train-dir", "dataset/train", "--train-list", "train.txt",
            "--val-dir", "dataset/val", "--val-list", "val.txt",
        ])
        self.assertEqual(args.epochs, 150)

    def test_testing_requires_a_checkpoint_and_test_split(self):
        parser = test.build_parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args([
            "--test-dir", "dataset/test", "--test-list", "test.txt",
            "--checkpoint", "best.pth",
        ])
        self.assertEqual(args.checkpoint, "best.pth")


if __name__ == "__main__":
    unittest.main()
