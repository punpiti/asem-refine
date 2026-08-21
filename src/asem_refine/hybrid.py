"""ASEM-Hybrid: reference-guided EM plus a local de novo gap-fill step.

Reference-guided placement (core.py) leaves a genome's most divergent
regions untouched -- no read ever clears the alignment threshold there, so
those positions keep whatever base theta started with, forever. But the
reads that *originate* from those regions still overlap each other exactly
even when none of them overlaps theta, because they all come from the same
real target sequence.

This module adds one step to every EM iteration: whenever a zero-depth
reference interval at least `min_anchor_len` long exists, assemble the
pool of reads that fail placement this round, plus any placed reads whose
accepted reference-coordinate alignment overlaps a `boundary_flank`-base
window immediately beside the gap, via read-read overlap alone (the
overlap assembly itself uses no reference information; the flank
coordinates only select which reads join the pool). Each resulting contig
is then anchored back onto theta with a length/identity rule sized for a
partial-contig match. An anchored contig is fed into the same
position/symbol vote-counting machinery as an ordinary read
(core.accumulate_alignment), so a well-supported contig can pull a
previously-untouched span of theta straight to the true sequence in one
shot. Because next iteration's E-step re-aligns every read -- including
previously-unplaced ones -- against the patched theta, reads that
couldn't place before may now clear the threshold near a contig's edges,
shrinking the unplaced pool and letting contigs grow further each round.

Passing `boundary_flank=0` (or `None`) reproduces the earlier unplaced-only
variant, whose assembly pool was strictly the reads that failed placement
that round, with no boundary-read recruitment.

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
    n_boundary_reads: int
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
    underlying 150bp reads fail the read-level threshold in the first
    place -- and parasail's local alignment often can't sustain that rate
    across a contig's full length, so it returns the best-scoring
    *sub*-window rather than an end-to-end match. Judging that against a
    bar sized for the whole contig punishes exactly the long, high-value
    contigs this step exists to anchor.

    contig_tau=0.25 is a score-normalized threshold, not identity
    directly: with this scoring scheme (match=+2, mismatch=-3, no gaps),
    score >= contig_tau * MATCH_SCORE * L works out to an identity floor
    of exactly 70% (solve 2p - 3(1-p) >= 0.5 for p). min_anchor_len=300
    is an additional, independent floor on top of that -- NOT because
    70%-identity matches are common at 150bp against random sequence
    (they aren't: a purely-random-DNA negative control produced 0 false
    anchors in 2000 trials at each of the 150bp and 300bp length floors,
    so contig_tau=0.25 alone already rejects unrelated sequence reliably
    against that null model). The 300bp floor is a margin against a
    different, untested risk: real non-homologous genomic sequence is not
    perfectly uniform-random (GC bias, low-complexity/repetitive
    stretches), which could in principle score better against theta than
    the idealized random-DNA control above. This margin has not been
    validated against structured/repeat-containing negative controls --
    an open gap, not a confirmed safety property.
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


def uncovered_gaps(depths: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Return half-open zero-depth reference intervals ``[start, end)`` at
    least `min_len` bases long.

    Used to gate the de novo assembly/anchoring step: it is only worth
    attempting when theta actually has a hole big enough for a contig to
    plausibly fill (anchor_contigs requires an aligned span >= min_anchor_len
    anyway, so a hole shorter than that could never be usefully filled).
    This is a per-iteration cost-saving gate, not a positional one: it
    controls whether the expensive assembly/anchoring step runs *this
    round* at all, not which theta positions an anchored contig is allowed
    to vote at. A contig's own alignment can (and empirically does) extend
    beyond the gap into positions ordinary reads already cover, at a small
    measured recall cost on some jobs -- don't describe this as
    restricting contig votes to gap positions; it doesn't."""
    if min_len <= 0:
        min_len = 1
    zero = depths == 0
    if not zero.any():
        return []
    padded = np.concatenate(([False], zero, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= min_len]


def alignment_ref_interval(aln: LocalAlignment) -> tuple[int, int]:
    """Return the half-open reference interval `[start, end)` consumed by
    an alignment (i.e. spanned by its `ref_aligned` bases, excluding gaps)."""
    ref_len = sum(base != "-" for base in aln.ref_aligned)
    return aln.ref_start, aln.ref_start + ref_len


def recruit_boundary_reads(
    placed: list[tuple[str, list[LocalAlignment]]],
    gaps: list[tuple[int, int]],
    flank: int,
) -> list[str]:
    """Select placed reads overlapping either reference-coordinate flank of
    a gap.

    A read is included once even when it has multiple accepted alignments
    or touches both flanks of the same or different gaps. `flank=150` means
    that for a gap `[a,b)` we use placed alignments overlapping
    `[a-150,a)` or `[b,b+150)`. This avoids global reassembly of all
    placed reads while giving the overlap assembler coordinate-supported
    sequence to bridge from at each boundary. No read-end labels are
    inferred from sequence: membership in a flank is determined purely
    from the accepted alignment's reference coordinates, and the
    overlap-layout assembler (`assemble_overlap_contigs`) then determines
    compatible suffix--prefix joins from the read sequences themselves."""
    if not flank or flank <= 0 or not gaps:
        return []
    boundary_reads: list[str] = []
    for read, alns in placed:
        for aln in alns:
            start, end = alignment_ref_interval(aln)
            if any(
                (start < gap_start and end > gap_start - flank)
                or (start < gap_end + flank and end > gap_end)
                for gap_start, gap_end in gaps
            ):
                boundary_reads.append(read)
                break
    return boundary_reads


def run_asem_hybrid_loop(
    theta_init: str,
    reads: list[str],
    align_read_fn: AlignReadFn,
    contig_tau: float = 0.25,
    min_anchor_len: int = 300,
    boundary_flank: int | None = 150,
    w: float = 0.1,
    max_iterations: int = 6,
    min_overlap: int = 20,
    n_workers: int = 1,
) -> AsemHybridResult:
    """Run ASEM-Hybrid. Ordinary placed-read voting is unchanged; the extra
    boundary reads (see `recruit_boundary_reads`) enter only the temporary
    overlap-assembly pool, and accepted contigs enter the existing
    position-counting update exactly like ordinary reads. Pass
    `boundary_flank=0` (or `None`) to disable boundary-read recruitment and
    reproduce the earlier unplaced-only variant."""
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
            placed: list[tuple[str, list[LocalAlignment]]] = []

            if pool is not None:
                results = pool.map(align_read_fn, reads, [theta] * len(reads), chunksize=32)
            else:
                results = (align_read_fn(r, theta) for r in reads)
            for read, alns in zip(reads, results):
                if alns:
                    n_placed += 1
                    placed.append((read, alns))
                    for aln in alns:
                        accumulate_alignment(counts, aln, n)
                else:
                    unplaced_reads.append(read)

            depths = counts.sum(axis=0)
            gaps = uncovered_gaps(depths, min_anchor_len)
            boundary_reads = recruit_boundary_reads(placed, gaps, boundary_flank or 0)
            if gaps:
                contigs = assemble_overlap_contigs(
                    unplaced_reads + boundary_reads, min_overlap=min_overlap
                )
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
                    n_boundary_reads=len(boundary_reads),
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
