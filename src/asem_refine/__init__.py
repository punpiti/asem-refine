"""ASEM-Hybrid: sample-specific reference sequence estimation from
low-depth short reads, for cases where the closest available reference is
still meaningfully divergent from the sample.

Public API:
    read_fasta, write_fasta, read_reads   -- I/O
    resolve_reference                      -- accession-or-path -> local FASTA path
    run_asem                               -- plain / recursive ASEM
    run_asem_hybrid_loop                   -- ASEM + local de novo gap-filling
"""

from .estep import run_asem
from .hybrid import run_asem_hybrid_loop
from .ncbi import resolve_reference
from .seqio import read_fasta, read_reads, write_fasta

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "read_fasta",
    "write_fasta",
    "read_reads",
    "resolve_reference",
    "run_asem",
    "run_asem_hybrid_loop",
]
