from __future__ import annotations

import unittest

from RecursiveMAS.inference_utils.linkradius import replay_operations, replay_schedule, valid_edges


class ReplayScheduleTests(unittest.TestCase):
    def test_every_r2_edge_has_exact_descendants(self) -> None:
        expected = {
            "p2c@0": ("critic", "solver_feedback", "planner_feedback", "critic", "score_final"),
            "c2s@0": ("solver_feedback", "planner_feedback", "critic", "score_final"),
            "s2p@0": ("planner_feedback", "critic", "score_final"),
            "p2c@1": ("critic", "score_final"),
            "c2s@1": ("score_final",),
        }
        self.assertEqual({edge.edge_id for edge in valid_edges(2)}, set(expected))
        for edge_id, operations in expected.items():
            with self.subTest(edge=edge_id):
                self.assertEqual(replay_operations(edge_id, 2), operations)

    def test_steps_name_exact_consumed_and_produced_edges(self) -> None:
        steps = replay_schedule("c2s@0", 3)
        self.assertEqual(steps[0].consumes_edge.edge_id, "c2s@0")
        self.assertEqual(steps[0].produces_edge.edge_id, "s2p@0")
        self.assertEqual(steps[-1].operation, "score_final")
        self.assertEqual(steps[-1].consumes_edge.edge_id, "c2s@2")
        # Nothing from round zero before the selected c2s receiver is recomputed.
        self.assertNotIn("planner", replay_operations("c2s@0", 3))
        self.assertNotIn("critic", replay_operations("c2s@0", 3)[:1])

    def test_invalid_edge_rejected_before_scheduling(self) -> None:
        with self.assertRaises(ValueError):
            replay_schedule("s2p@3", 4)


if __name__ == "__main__":
    unittest.main()

