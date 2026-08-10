from __future__ import annotations

import unittest

from RecursiveMAS.inference_utils.linkradius import Edge, parse_edge, valid_edges, validate_edge


class EdgeTests(unittest.TestCase):
    def test_exact_chronology_and_count(self) -> None:
        self.assertEqual([edge.edge_id for edge in valid_edges(1)], ["p2c@0", "c2s@0"])
        self.assertEqual(
            [edge.edge_id for edge in valid_edges(2)],
            ["p2c@0", "c2s@0", "s2p@0", "p2c@1", "c2s@1"],
        )
        for horizon in range(1, 8):
            self.assertEqual(len(valid_edges(horizon)), 3 * horizon - 1)

    def test_terminal_s2p_is_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid intervention edge"):
            validate_edge(Edge("s2p", 1), 2)
        with self.assertRaises(ValueError):
            validate_edge("s2p@0", 1)

    def test_parse_and_round_fields(self) -> None:
        for value in ("c2s@3", "c2s:3", "c2s_r3", {"site": "c2s", "code_round": 3}):
            edge = parse_edge(value)
            self.assertEqual(edge, Edge("c2s", 3))
            self.assertEqual(edge.paper_round, 4)
            self.assertEqual(edge.token, "c2s_r3")

    def test_invalid_horizon_and_site_fail_closed(self) -> None:
        for horizon in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                valid_edges(horizon)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Edge("solver", 0)


if __name__ == "__main__":
    unittest.main()

