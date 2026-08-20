"""E-step alignment rules: the plain single-shot rule and OJEMB's recursive
partial-alignment extension. Ported from
scripts/refine_reference/baseline_ieee_access.py and baseline_ojemb.py.
"""

from __future__ import annotations

import functools

from .align import MATCH_SCORE, LocalAlignment, align_local
from .core import AsemResult, run_asem_em_loop


def _align_read_plain(read: str, theta: str, tau: float) -> list[LocalAlignment]:
    """One local alignment attempt per read; accepted or discarded whole."""
    aln = align_local(read, theta)
    if aln is not None and aln.score >= tau * MATCH_SCORE * len(read):
        return [aln]
    return []


def _align_read_recursive(
    read: str, theta: str, tau: float, min_leftover_len: int
) -> list[LocalAlignment]:
    accepted: list[LocalAlignment] = []
    _recursive_align(read, theta, tau, min_leftover_len, accepted)
    return accepted


def _recursive_align(
    read: str, theta: str, tau: float, min_leftover_len: int, accepted: list[LocalAlignment]
) -> None:
    if not read:
        return
    aln = align_local(read, theta)
    if aln is None:
        return

    if aln.score >= tau * MATCH_SCORE * len(read):
        accepted.append(aln)

    read_ungapped_len = len(aln.read_aligned) - aln.read_aligned.count("-")
    prefix = read[: aln.read_start]
    suffix = read[aln.read_start + read_ungapped_len :]

    if len(prefix) >= min_leftover_len:
        _recursive_align(prefix, theta, tau, min_leftover_len, accepted)
    if len(suffix) >= min_leftover_len:
        _recursive_align(suffix, theta, tau, min_leftover_len, accepted)


def build_align_read_fn(tau: float, recursive: bool, min_leftover_len: int = 30):
    """The single point that turns (tau, recursive) into an AlignReadFn --
    shared by run_asem() below and the CLI's hybrid path, so both variants
    always use the exact same E-step rule."""
    if recursive:
        return functools.partial(_align_read_recursive, tau=tau, min_leftover_len=min_leftover_len)
    return functools.partial(_align_read_plain, tau=tau)


def run_asem(
    theta_init: str,
    reads: list[str],
    tau: float = 0.5,
    w: float = 0.1,
    max_iterations: int = 6,
    n_workers: int = 1,
    recursive: bool = True,
    min_leftover_len: int = 30,
) -> AsemResult:
    """Plain ASEM (recursive=False) or ASEM+recursive-partial-alignment
    (recursive=True, the default -- it never measured worse than plain ASEM
    on this project's benchmark grid and recovers real signal on real,
    noisy reads, so there is no reason to default it off)."""
    align_read_fn = build_align_read_fn(tau, recursive, min_leftover_len)
    return run_asem_em_loop(
        theta_init=theta_init,
        reads=reads,
        align_read_fn=align_read_fn,
        w=w,
        max_iterations=max_iterations,
        n_workers=n_workers,
    )
