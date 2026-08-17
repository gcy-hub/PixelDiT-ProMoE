from __future__ import annotations

import argparse
import unittest

from c2i.infer_c2i import (
    parse_class_counts,
    parse_gpu_ids,
    partition_tasks,
    select_denoiser_state_dict,
)


class InferenceArgumentTests(unittest.TestCase):
    def test_parse_class_counts(self):
        self.assertEqual(parse_class_counts('{"207": 2, "0": 4}'), {0: 4, 207: 2})

    def test_reject_invalid_class_counts(self):
        for value in ('{}', '{"1000": 1}', '{"0": 0}', '{"cat": 1}'):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_class_counts(value)

    def test_parse_gpu_ids(self):
        self.assertEqual(parse_gpu_ids("0, 2,3"), [0, 2, 3])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_gpu_ids("0,0")

    def test_partition_tasks_is_complete_and_disjoint(self):
        assignments = partition_tasks({0: 4, 7: 3}, num_workers=3)
        assigned = [
            (class_id, image_index)
            for assignment in assignments
            for class_id, indices in assignment.items()
            for image_index in indices
        ]
        self.assertEqual(
            sorted(assigned),
            [(0, 0), (0, 1), (0, 2), (0, 3), (7, 0), (7, 1), (7, 2)],
        )


class CheckpointSelectionTests(unittest.TestCase):
    def test_prefers_ema_and_can_select_raw_denoiser(self):
        checkpoint = {
            "state_dict": {
                "denoiser.weight": 1.0,
                "ema_denoiser.weight": 2.0,
            }
        }
        state_dict, source = select_denoiser_state_dict(checkpoint, use_ema=True)
        self.assertEqual(source, "EMA")
        self.assertEqual(state_dict["weight"], 2.0)

        state_dict, source = select_denoiser_state_dict(checkpoint, use_ema=False)
        self.assertEqual(source, "denoiser")
        self.assertEqual(state_dict["weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
