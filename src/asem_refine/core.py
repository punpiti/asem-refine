"""Shared EM loop: position/symbol tallying, the match-state-only consensus
rule, and the iterate-until-stable loop. Ported from the paper's research
codebase (scripts/refine_reference/asem_core.py); see that file's docstring
and the paper (Section 2.2, "Approximate Structural EM algorithm") for the
full derivation.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .align import LocalAlignment, placement_base_calls

SYMBOLS = "ACGT-"
SYMBOL_IDX = {s: i for i, s in enumerate(SYMBOLS)}

# (read, theta) -> accepted local alignments for that read (already
# threshold-filtered by the caller; empty list means unplaced).
AlignReadFn = Callable[[str, str], list[LocalAlignment]]


@dataclass
class IterationStats:
    iteration: int
    n_placed: int
    n_unplaced: int
    theta_changed: bool


@dataclass
class AsemResult:
    theta: str
    history: list[IterationStats] = field(default_factory=list)
    theta_by_iteration: list[str] = field(default_factory=list)


def run_asem_em_loop(
    theta_init: str,
    reads: list[str],
    align_read_fn: AlignReadFn,
    w: float = 0.1,
    max_iterations: int = 6,
    n_workers: int = 1,
) -> AsemResult:
    theta = theta_init
    history: list[IterationStats] = []
    theta_by_iteration: list[str] = []
    prev_unplaced_count: int | None = None

    pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    try:
        for it in range(1, max_iterations + 1):
            theta_prev = theta
            n = len(theta)
            counts = np.zeros((len(SYMBOLS), n), dtype=np.int64)
            n_placed = 0
            n_unplaced = 0

            if pool is not None:
                results = pool.map(align_read_fn, reads, [theta] * len(reads), chunksize=32)
            else:
                results = (align_read_fn(r, theta) for r in reads)
            for alns in results:
                if alns:
                    n_placed += 1
                    for aln in alns:
                        accumulate_alignment(counts, aln, n)
                else:
                    n_unplaced += 1

            theta = build_theta(theta_prev, counts, w)
            theta_by_iteration.append(theta)

            changed = theta != theta_prev
            history.append(IterationStats(it, n_placed, n_unplaced, changed))

            if not changed and (
                prev_unplaced_count is None or n_unplaced >= prev_unplaced_count
            ):
                break
            prev_unplaced_count = n_unplaced
    finally:
        if pool is not None:
            pool.shutdown()

    return AsemResult(theta=theta, history=history, theta_by_iteration=theta_by_iteration)


def accumulate_alignment(counts: np.ndarray, aln: LocalAlignment, n: int) -> None:
    """Add one alignment's base calls into the (5, n) position/symbol count
    array in place. Calls outside A/C/G/T/- (e.g. an ambiguity code) carry
    no information about the true base and are skipped rather than counted."""
    for pos, base in placement_base_calls(aln).items():
        if 0 <= pos < n and base in SYMBOL_IDX:
            counts[SYMBOL_IDX[base], pos] += 1


def build_theta(theta_prev: str, counts: np.ndarray, w: float) -> str:
    n = len(theta_prev)
    depths = counts.sum(axis=0)
    mean_depth = depths.mean() if n else 0.0
    min_support = w * mean_depth
    well_supported = depths > min_support

    winner_idx = counts.argmax(axis=0)
    symbol_array = np.frombuffer(SYMBOLS.encode(), dtype="S1")
    winner_symbols = symbol_array[winner_idx]
    prior_symbols = np.frombuffer(theta_prev.encode(), dtype="S1")

    final = np.where(well_supported, winner_symbols, prior_symbols)
    return final[final != b"-"].tobytes().decode()
