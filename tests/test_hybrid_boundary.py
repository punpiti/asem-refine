"""Focused regression tests for boundary-recruited ASEM-Hybrid."""

from __future__ import annotations

import unittest

import numpy as np

from asem_refine.align import LocalAlignment
from asem_refine.hybrid import (
    alignment_ref_interval,
    recruit_boundary_reads,
    uncovered_gaps,
)


def aln(start: int, length: int) -> LocalAlignment:
    return LocalAlignment(2.0 * length, start, 0, "A" * length, "A" * length)


class BoundaryRecruitmentTests(unittest.TestCase):
    def test_uncovered_gaps_are_half_open_and_filtered(self) -> None:
        self.assertEqual(uncovered_gaps(np.array([1, 0, 0, 1, 0, 0, 0]), 3), [(4, 7)])

    def test_alignment_interval_counts_reference_bases_not_insertions(self) -> None:
        gapped = LocalAlignment(5.0, 10, 0, "AC-GT", "ACTGT")
        self.assertEqual(alignment_ref_interval(gapped), (10, 14))

    def test_recruitment_uses_only_flanks_and_deduplicates_per_read(self) -> None:
        placed = [
            ("left", [aln(80, 20), aln(100, 10)]),
            ("right", [aln(120, 20)]),
            ("far", [aln(20, 20)]),
            ("inside-gap", [aln(105, 10)]),
        ]
        self.assertEqual(recruit_boundary_reads(placed, [(100, 120)], 20), ["left", "right"])

    def test_no_gap_or_zero_flank_recruits_nothing(self) -> None:
        placed = [("left", [aln(80, 20)])]
        self.assertEqual(recruit_boundary_reads(placed, [], 20), [])
        self.assertEqual(recruit_boundary_reads(placed, [(100, 120)], 0), [])


if __name__ == "__main__":
    unittest.main()
