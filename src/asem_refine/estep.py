"""E-step alignment rules: the plain single-shot rule and OJEMB's recursive
partial-alignment extension. Ported from
scripts/refine_reference/baseline_ieee_access.py and baseline_ojemb.py.
"""

from __future__ import annotations

import functools

from .align import MATCH_SCORE, LocalAlignment, align_local, reverse_complement
from .core import AsemResult, run_asem_em_loop


def _align_read_plain(
    read: str, theta: str, tau: float, try_reverse_complement: bool = True
) -> list[LocalAlignment]:
    """One local alignment attempt per read; accepted or discarded whole.
    `align_local` tries both strands by default, so orientation does not
    need to be resolved separately here."""
    aln = align_local(read, theta, try_reverse_complement=try_reverse_complement)
    if aln is not None and aln.score >= tau * MATCH_SCORE * len(read):
        return [aln]
    return []


def _align_read_recursive(
    read: str,
    theta: str,
    tau: float,
    min_leftover_len: int,
    try_reverse_complement: bool = True,
) -> list[LocalAlignment]:
    """Resolve the read's strand orientation once against the whole read,
    then recurse entirely within that orientation. Re-checking orientation
    per sub-fragment inside `_recursive_align` would be both wasteful and
    wrong: prefix/suffix are sliced from `read` using coordinates from a
    single alignment call, which are only valid if that call used the same
    orientation as the slicing."""
    oriented_read = read
    if try_reverse_complement:
        fwd = align_local(read, theta, try_reverse_complement=False)
        rc = reverse_complement(read)
        rev = align_local(rc, theta, try_reverse_complement=False)
        fwd_score = fwd.score if fwd is not None else -1.0
        rev_score = rev.score if rev is not None else -1.0
        oriented_read = rc if rev_score > fwd_score else read

    accepted: list[LocalAlignment] = []
    _recursive_align(oriented_read, theta, tau, min_leftover_len, accepted)
    return accepted


def _recursive_align(
    read: str, theta: str, tau: float, min_leftover_len: int, accepted: list[LocalAlignment]
) -> None:
    if not read:
        return
    aln = align_local(read, theta, try_reverse_complement=False)
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


def build_align_read_fn(
    tau: float, recursive: bool, min_leftover_len: int = 30, try_reverse_complement: bool = True
):
    """The single point that turns (tau, recursive, try_reverse_complement)
    into an AlignReadFn -- shared by run_asem() below and the CLI's hybrid
    path, so both variants always use the exact same E-step rule."""
    if recursive:
        return functools.partial(
            _align_read_recursive,
            tau=tau,
            min_leftover_len=min_leftover_len,
            try_reverse_complement=try_reverse_complement,
        )
    return functools.partial(_align_read_plain, tau=tau, try_reverse_complement=try_reverse_complement)


def run_asem(
    theta_init: str,
    reads: list[str],
    tau: float = 0.5,
    w: float = 0.1,
    max_iterations: int = 6,
    n_workers: int = 1,
    recursive: bool = True,
    min_leftover_len: int = 30,
    try_reverse_complement: bool = True,
) -> AsemResult:
    """Plain ASEM (recursive=False) or ASEM+recursive-partial-alignment
    (recursive=True, the default -- it never measured worse than plain ASEM
    on this project's benchmark grid and recovers real signal on real,
    noisy reads, so there is no reason to default it off)."""
    align_read_fn = build_align_read_fn(tau, recursive, min_leftover_len, try_reverse_complement)
    return run_asem_em_loop(
        theta_init=theta_init,
        reads=reads,
        align_read_fn=align_read_fn,
        w=w,
        max_iterations=max_iterations,
        n_workers=n_workers,
    )
