"""ASEM-Hybrid: reference-guided EM plus a local de novo gap-fill step.

Reference-guided placement (core.py) leaves a genome's most divergent
regions untouched -- no read ever clears the alignment threshold there, so
those positions keep whatever base theta started with, forever. But the
reads that *originate* from those regions still overlap each other exactly
even when none of them overlaps theta, because they all come from the same
real target sequence.

This module adds one step to every EM iteration: assemble the pool of
reads that fail placement this round via read-read overlap alone (no
reference involved), then try to anchor each resulting contig back onto
theta with the same score-threshold rule used for reads. An anchored
contig is fed into the same position/symbol vote-counting machinery as an
ordinary read (core.accumulate_alignment), so a well-supported contig can
pull a previously-untouched span of theta straight to the true sequence in
one shot. Because next iteration's E-step re-aligns every read --
including previously-unplaced ones -- against the patched theta, reads
that couldn't place before may now clear the threshold near a contig's
edges, shrinking the unplaced pool and letting contigs grow further each
round.

See the accompanying paper for the empirical validation of this mechanism.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .align import LocalAlignment, align_local
from .align import MATCH_SCORE as MATCH_SCORE  # noqa: PLC0414 (re-exported for callers)
from .core import SYMBOLS, accumulate_alignment, build_theta

AlignReadFn = Callable[[str, str], list[LocalAlignment]]


@dataclass
class HybridIterationStats:
    iteration: int
    n_placed: int
    n_unplaced: int
    theta_changed: bool
    n_contigs: int
    n_contigs_anchored: int
    contig_lengths_anchored: list[int] = field(default_factory=list)


@dataclass
class AsemHybridResult:
    theta: str
    history: list[HybridIterationStats] = field(default_factory=list)
    theta_by_iteration: list[str] = field(default_factory=list)


def suffix_prefix_overlap(a: str, b: str, min_overlap: int) -> int:
    """Longest suffix of `a` that exactly matches a prefix of `b`
    (>= min_overlap), or 0 if none reaches that length."""
    max_ov = min(len(a), len(b))
    for ov in range(max_ov, min_overlap - 1, -1):
        if a[-ov:] == b[:ov]:
            return ov
    return 0


def assemble_overlap_contigs(
    reads: list[str], min_overlap: int = 20, seed_k: int = 20
) -> list[str]:
    """Greedy overlap-layout assembly of `reads` by exact suffix-prefix
    overlap. No reference is involved -- this only uses read-read overlap,
    which is why it can recover sequence theta can't reach.

    Candidate pairs are pruned with a k-mer seed index (share a `seed_k`-mer
    at the suffix/prefix boundary) instead of checked exhaustively, since
    reads from unrelated regions of the genome essentially never share one
    by chance. This keeps assembly near-linear in read count rather than
    the O(n^3) cost of an all-pairs brute-force greedy merge.
    """
    n = len(reads)
    if n == 0:
        return []

    # Anchor each read j by its first seed_k bases (offset 0 -- the position
    # its prefix, by definition, always starts at). To find it from the i
    # side, we don't know in advance how long i's overlapping suffix is, so
    # we scan every offset of i (not just its last seed_k bases) for a
    # matching j-anchor -- a hit at offset p in i implies a candidate
    # overlap of length len(i) - p, verified below.
    prefix_index: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(reads):
        if len(r) >= seed_k:
            prefix_index[r[:seed_k]].append(i)

    # (overlap_len, i, j): reads[i]'s suffix overlaps reads[j]'s prefix.
    seen_pairs: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int, int]] = []
    for i, r in enumerate(reads):
        for p in range(0, len(r) - seed_k + 1):
            for j in prefix_index.get(r[p : p + seed_k], ()):
                if i == j or (i, j) in seen_pairs:
                    continue
                seen_pairs.add((i, j))
                ov = suffix_prefix_overlap(r, reads[j], min_overlap)
                if ov > 0:
                    candidates.append((ov, i, j))
    candidates.sort(reverse=True)

    # Greedily chain reads into non-branching paths: each read is the left
    # side of at most one merge and the right side of at most one merge,
    # and a merge is rejected if it would close a cycle. chain_start/
    # chain_end track, for the two free ends of each growing chain, the
    # node at its opposite end -- updated only at those two ends per merge,
    # which is enough to keep both lookups correct (see reasoning in the
    # module's development notes: any node still eligible as a merge side
    # is always a current chain end, so its entry is always live).
    next_of: dict[int, int] = {}
    prev_of: dict[int, int] = {}
    chain_start: dict[int, int] = {i: i for i in range(n)}
    chain_end: dict[int, int] = {i: i for i in range(n)}

    for _, i, j in candidates:
        if i in next_of or j in prev_of:
            continue
        if chain_start[i] == chain_start[j]:
            continue  # would close a cycle
        next_of[i] = j
        prev_of[j] = i
        start_i, end_j = chain_start[i], chain_end[j]
        chain_start[end_j] = start_i
        chain_end[start_i] = end_j

    contigs: list[str] = []
    for i in range(n):
        if i in prev_of:
            continue  # not a chain start
        seq = reads[i]
        cur = i
        while cur in next_of:
            nxt = next_of[cur]
            ov = suffix_prefix_overlap(seq, reads[nxt], min_overlap)
            seq += reads[nxt][ov:]
            cur = nxt
        contigs.append(seq)
    return contigs


def anchor_contigs(
    contigs: list[str], theta: str, contig_tau: float, min_anchor_len: int
) -> list[tuple[str, LocalAlignment]]:
    """Locally align each contig back onto theta and keep the ones with a
    long, reasonably-identical anchor window.

    This deliberately does NOT reuse the read-level tau/length rule (score
    >= tau * MATCH_SCORE * len(contig)). Empirically, contigs assembled
    from an unplaced-read pool anchor to theta at ~73-80% identity over
    their *aligned span* -- exactly the divergence level that made the
    underlying 150bp reads fail the read-level threshold (tau=0.5 is
    roughly an 80%-identity cutoff at that length) in the first place, and
    parasail's local alignment often can't sustain that rate across a
    contig's full length, so it returns the best-scoring *sub*-window
    rather than an end-to-end match. Judging that against a bar sized for
    the whole contig punishes exactly the long, high-value contigs this
    step exists to anchor.

    What actually changes with length is statistical confidence, not
    identity: a coincidental ~75%-identity local match spanning even a few
    hundred bp of otherwise-unrelated sequence is effectively impossible,
    while it happens often at 150bp. So this checks the aligned window's
    own identity against a lower, length-appropriate bar (contig_tau,
    default well below the read-level tau) but requires that window to be
    at least `min_anchor_len` -- comfortably longer than one read -- as
    the actual safeguard against spurious short matches.
    """
    anchored = []
    for contig in contigs:
        aln = align_local(contig, theta)
        if aln is None:
            continue
        aligned_len = len(aln.read_aligned) - aln.read_aligned.count("-")
        if aligned_len < min_anchor_len:
            continue
        if aln.score >= contig_tau * MATCH_SCORE * aligned_len:
            anchored.append((contig, aln))
    return anchored


def has_uncovered_gap(depths: np.ndarray, min_len: int) -> bool:
    """True if `depths` (per-position placed-read support count) contains a
    contiguous run of zero-depth positions at least `min_len` long.

    Used to gate the de novo assembly/anchoring step: it is only worth
    attempting when theta actually has a hole big enough for a contig to
    plausibly fill (anchor_contigs requires an aligned span >= min_anchor_len
    anyway, so a hole shorter than that could never be usefully filled).
    Skipping the gate on iterations with no such hole is not just an
    optimization -- it makes the "only intervenes where genuinely needed"
    claim (Section~sec:hybrid-discussion) true of the implementation, not
    just the outcome."""
    if min_len <= 0:
        return True
    zero = depths == 0
    if not zero.any():
        return False
    padded = np.concatenate(([False], zero, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return bool(((ends - starts) >= min_len).any())


def run_asem_hybrid_loop(
    theta_init: str,
    reads: list[str],
    align_read_fn: AlignReadFn,
    contig_tau: float = 0.25,
    min_anchor_len: int = 300,
    w: float = 0.1,
    max_iterations: int = 6,
    min_overlap: int = 20,
    n_workers: int = 1,
) -> AsemHybridResult:
    theta = theta_init
    history: list[HybridIterationStats] = []
    theta_by_iteration: list[str] = []
    prev_unplaced_count: int | None = None

    pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    try:
        for it in range(1, max_iterations + 1):
            theta_prev = theta
            n = len(theta)
            counts = np.zeros((len(SYMBOLS), n), dtype=np.int64)
            n_placed = 0
            unplaced_reads: list[str] = []

            if pool is not None:
                results = pool.map(align_read_fn, reads, [theta] * len(reads), chunksize=32)
            else:
                results = (align_read_fn(r, theta) for r in reads)
            for read, alns in zip(reads, results):
                if alns:
                    n_placed += 1
                    for aln in alns:
                        accumulate_alignment(counts, aln, n)
                else:
                    unplaced_reads.append(read)

            depths = counts.sum(axis=0)
            if has_uncovered_gap(depths, min_anchor_len):
                contigs = assemble_overlap_contigs(unplaced_reads, min_overlap=min_overlap)
                anchored = anchor_contigs(contigs, theta, contig_tau, min_anchor_len)
                for contig, aln in anchored:
                    accumulate_alignment(counts, aln, n)
            else:
                contigs, anchored = [], []

            theta = build_theta(theta_prev, counts, w)
            theta_by_iteration.append(theta)

            changed = theta != theta_prev
            n_unplaced = len(unplaced_reads)
            history.append(
                HybridIterationStats(
                    iteration=it,
                    n_placed=n_placed,
                    n_unplaced=n_unplaced,
                    theta_changed=changed,
                    n_contigs=len(contigs),
                    n_contigs_anchored=len(anchored),
                    contig_lengths_anchored=[len(contig) for contig, _ in anchored],
                )
            )

            if not changed and (
                prev_unplaced_count is None or n_unplaced >= prev_unplaced_count
            ):
                break
            prev_unplaced_count = n_unplaced
    finally:
        if pool is not None:
            pool.shutdown()

    return AsemHybridResult(theta=theta, history=history, theta_by_iteration=theta_by_iteration)
