"""Local alignment primitives shared by every E-step variant.

Ported from the paper's research codebase (scripts/refine_reference/common.py)
with the evaluation-only pieces (which need a known true target and are not
part of normal usage) left out.
"""

from __future__ import annotations

from dataclasses import dataclass

import parasail

MATCH_SCORE = 2.0
_SW_MATRIX = parasail.matrix_create("ACGT", 2, -3)  # match=2, mismatch=-3
_SW_GAP_OPEN = 5
_SW_GAP_EXTEND = 2


@dataclass
class LocalAlignment:
    score: float
    ref_start: int  # 0-indexed offset into the reference where alignment begins
    read_start: int
    ref_aligned: str  # gapped, aligned-region only
    read_aligned: str


_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _align_local_one_strand(read: str, reference: str) -> LocalAlignment | None:
    res = parasail.sw_trace_striped_16(read, reference, _SW_GAP_OPEN, _SW_GAP_EXTEND, _SW_MATRIX)
    if res.score <= 0:
        return None
    tb = res.get_traceback("-")
    ref_aligned = tb.ref
    read_aligned = tb.query
    ref_ungapped_len = len(ref_aligned) - ref_aligned.count("-")
    read_ungapped_len = len(read_aligned) - read_aligned.count("-")
    ref_start = res.end_ref - ref_ungapped_len + 1
    read_start = res.end_query - read_ungapped_len + 1
    return LocalAlignment(
        score=float(res.score),
        ref_start=int(ref_start),
        read_start=int(read_start),
        ref_aligned=ref_aligned,
        read_aligned=read_aligned,
    )


def align_local(read: str, reference: str, try_reverse_complement: bool = True) -> LocalAlignment | None:
    """Local Striped Smith-Waterman alignment of `read` onto `reference`
    (parasail, Farrar 2007).

    Raw FASTQ (unlike the pre-oriented SEQ field of a mapped BAM record, or
    this package's own simulated benchmark reads) mixes reads from both
    strands. By default this tries both `read` and its reverse complement
    and keeps whichever scores higher, so a read's orientation does not need
    to be known or corrected beforehand. The returned alignment's coordinates
    always describe whichever orientation won; callers that need to know
    which one that was should reverse-complement `read` themselves and
    compare, since `LocalAlignment` does not carry a strand flag. Pass
    `try_reverse_complement=False` to skip the second alignment (e.g. when
    the caller already knows the read is correctly oriented, or is re-trying
    an already-oriented sub-fragment during recursive alignment)."""
    fwd = _align_local_one_strand(read, reference)
    if not try_reverse_complement:
        return fwd
    rev = _align_local_one_strand(reverse_complement(read), reference)
    if rev is None:
        return fwd
    if fwd is None or rev.score > fwd.score:
        return rev
    return fwd


def placement_base_calls(aln: LocalAlignment) -> dict[int, str]:
    """Map an alignment's read bases onto absolute reference-column positions.

    Only positions where the reference has a base contribute a call: either
    the read's base (a match/mismatch vote) or "-" (a deletion vote, when the
    read has a gap at that reference column). Insertions in the read (extra
    read bases not present in the reference) are dropped -- this
    match-state-only design does not let reads extend theta's length.
    """
    calls: dict[int, str] = {}
    ref_pos = aln.ref_start
    for ref_ch, read_ch in zip(aln.ref_aligned, aln.read_aligned):
        if ref_ch == "-":
            continue  # insertion in the read; not represented in theta
        calls[ref_pos] = read_ch
        ref_pos += 1
    return calls
