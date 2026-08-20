"""FASTA/FASTQ readers and writers. Handles plain and gzip-compressed files
transparently (by extension), since real sequencer output is almost always
`.fastq.gz`.
"""

from __future__ import annotations

import gzip
import os
from collections.abc import Iterator


def _open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_fasta(path: str) -> tuple[str, str]:
    """Read a single-record FASTA file. Returns (header, sequence)."""
    header = None
    parts: list[str] = []
    with _open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    raise ValueError(
                        f"{path} contains more than one FASTA record; "
                        "a reference sequence must be a single record"
                    )
                header = line[1:]
            elif line:
                parts.append(line)
    if header is None:
        raise ValueError(f"no FASTA record found in {path}")
    return header, "".join(parts).upper()


def write_fasta(path: str, header: str, sequence: str, line_width: int = 70) -> None:
    with open(path, "w") as fh:
        fh.write(f">{header}\n")
        for i in range(0, len(sequence), line_width):
            fh.write(sequence[i : i + line_width] + "\n")


def iter_fastq(path: str) -> Iterator[str]:
    """Yield each read's sequence from a FASTQ file (plain or .gz)."""
    with _open_text(path) as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                yield line.rstrip("\n").upper()


def read_reads(paths: list[str]) -> list[str]:
    """Load read sequences from one or more input files. Dispatches by
    extension: .fastq/.fq (optionally .gz) are parsed as FASTQ, anything
    else is treated as FASTA (one or more records, each record's sequence
    used as a single "read" -- lets e.g. an assembled contigs FASTA be fed
    back in as input reads)."""
    reads: list[str] = []
    for path in paths:
        name = path[:-3] if path.endswith(".gz") else path
        ext = os.path.splitext(name)[1].lower()
        if ext in (".fastq", ".fq"):
            reads.extend(iter_fastq(path))
        else:
            reads.extend(_iter_fasta_records(path))
    return reads


def _iter_fasta_records(path: str) -> Iterator[str]:
    seq_parts: list[str] = []
    with _open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if seq_parts:
                    yield "".join(seq_parts).upper()
                    seq_parts = []
            elif line:
                seq_parts.append(line)
    if seq_parts:
        yield "".join(seq_parts).upper()
