"""Regression tests for reverse-complement handling in align_local/estep.

Raw FASTQ (unlike this package's own forward-only simulated benchmark
reads, or a pre-oriented BAM SEQ field) mixes reads from both strands.
"""

from __future__ import annotations

import random
import unittest

from asem_refine.align import align_local, reverse_complement
from asem_refine.estep import build_align_read_fn


def _rand_seq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


class ReverseComplementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(7)
        self.ref = _rand_seq(2000, self.rng)

    def test_reverse_complement_is_involution(self) -> None:
        seq = "ACGTACGGT"
        self.assertEqual(reverse_complement(reverse_complement(seq)), seq)
        self.assertEqual(reverse_complement("ACGT"), "ACGT")

    def test_align_local_finds_reverse_strand_read_by_default(self) -> None:
        rev_read = reverse_complement(self.ref[900:1050])
        aln = align_local(rev_read, self.ref)
        self.assertIsNotNone(aln)
        assert aln is not None
        self.assertEqual(aln.ref_start, 900)
        self.assertGreaterEqual(aln.score, 290.0)

    def test_align_local_can_skip_reverse_complement_search(self) -> None:
        rev_read = reverse_complement(self.ref[900:1050])
        aln = align_local(rev_read, self.ref, try_reverse_complement=False)
        # a true reverse-strand read aligned only in the forward direction
        # should score far below a real match (near-random noise level)
        self.assertTrue(aln is None or aln.score < 50.0)

    def test_plain_e_step_places_mixed_orientation_reads(self) -> None:
        align_fn = build_align_read_fn(tau=0.5, recursive=False, min_leftover_len=30)
        reads = []
        for i in range(0, len(self.ref) - 150, 40):
            frag = self.ref[i : i + 150]
            if (i // 40) % 2 == 0:
                frag = reverse_complement(frag)
            reads.append(frag)
        placed = sum(1 for r in reads if align_fn(r, self.ref))
        self.assertEqual(placed, len(reads))

    def test_recursive_e_step_handles_reverse_complemented_straddling_read(self) -> None:
        divergent_middle = _rand_seq(60, self.rng)
        straddle_fwd = self.ref[770:830] + divergent_middle + self.ref[830:890]
        straddle_rev = reverse_complement(straddle_fwd)
        align_fn = build_align_read_fn(tau=0.5, recursive=True, min_leftover_len=20)
        alns = align_fn(straddle_rev, self.ref)
        self.assertGreaterEqual(len(alns), 1)
        # every accepted fragment must land near the true reference window,
        # not at a spurious position from a forward/reverse coordinate mixup
        for aln in alns:
            self.assertTrue(750 <= aln.ref_start <= 910)


if __name__ == "__main__":
    unittest.main()
