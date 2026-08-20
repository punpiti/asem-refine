"""Command-line entry point: asem-refine --reference REF --reads READS [...]

  asem-refine --reference ref.fasta --reads reads.fastq --output theta.fasta
  asem-refine --reference NC_012920.1 --reads reads.fastq -o theta.fasta
  asem-refine -c config.txt          # NOVOPlasty-style config file

--reference accepts either a local FASTA path or an NCBI nucleotide
accession (e.g. NC_012920.1), fetched and cached automatically. --reads
accepts one or more FASTA/FASTQ files (plain or .gz).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__
from .estep import build_align_read_fn, run_asem
from .hybrid import run_asem_hybrid_loop
from .ncbi import resolve_reference
from .novoplasty_config import parse_config
from .seqio import read_fasta, read_reads, write_fasta

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "asem-refine", "ncbi")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asem-refine",
        description=(
            "Estimate a sample-specific reference sequence from low-depth "
            "short reads and a (possibly divergent) starting reference."
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    p.add_argument(
        "-r", "--reference",
        help="Reference sequence: a FASTA file, or an NCBI nucleotide accession (e.g. NC_012920.1) to fetch automatically. "
             "Same role as NOVOPlasty config.txt's \"Reference sequence\".",
    )
    p.add_argument(
        "--reads", nargs="+",
        help="One or more read files (FASTA/FASTQ, plain or .gz) -- covers NOVOPlasty's \"Combined reads\" / "
             "\"Forward reads\" + \"Reverse reads\" (this tool does not use pairing, so just list every file).",
    )
    p.add_argument(
        "-c", "--config",
        help="NOVOPlasty-style config.txt. Fills in --reference/--reads/--project-name/--output when not given explicitly on the command line.",
    )
    p.add_argument(
        "--project-name",
        help="Same as NOVOPlasty config.txt's \"Project name\". Used to name the output file when --output is not given.",
    )
    p.add_argument(
        "-o", "--output",
        help="Output FASTA path for the refined reference (default: <project-name>.theta.fasta, or theta.fasta).",
    )
    p.add_argument(
        "--cache-dir", default=DEFAULT_CACHE_DIR,
        help=f"Where to cache NCBI-fetched references (default: {DEFAULT_CACHE_DIR}).",
    )

    algo = p.add_argument_group("algorithm")
    algo.add_argument(
        "--hybrid", dest="hybrid", action="store_true", default=True,
        help="Enable local de novo gap-filling for regions no read can reach (default: on).",
    )
    algo.add_argument("--no-hybrid", dest="hybrid", action="store_false")
    algo.add_argument(
        "--recursive", dest="recursive", action="store_true", default=True,
        help="Enable recursive partial alignment for reads clipped by a structural breakpoint (default: on).",
    )
    algo.add_argument("--no-recursive", dest="recursive", action="store_false")
    algo.add_argument("--tau", type=float, default=0.5, help="Read-level alignment acceptance threshold (default: 0.5).")
    algo.add_argument("--w", type=float, default=0.1, help="M-step minimum-confidence weight (default: 0.1).")
    algo.add_argument("--max-iterations", type=int, default=6, help="Maximum EM iterations (default: 6).")
    algo.add_argument("--min-leftover-len", type=int, default=30, help="Minimum flank length for recursive re-alignment (default: 30).")
    algo.add_argument("--contig-tau", type=float, default=0.25, help="Hybrid: contig-anchor identity threshold (default: 0.25).")
    algo.add_argument("--min-anchor-len", type=int, default=300, help="Hybrid: minimum aligned contig-anchor length (default: 300).")
    algo.add_argument("--min-overlap", type=int, default=20, help="Hybrid: minimum read-read overlap to merge (default: 20).")
    algo.add_argument("-t", "--threads", type=int, default=1, help="Parallel worker processes (default: 1).")
    algo.add_argument("-q", "--quiet", action="store_true", help="Suppress per-iteration progress output.")

    return p


def _apply_config_defaults(args: argparse.Namespace) -> None:
    if not args.config:
        return
    cfg = parse_config(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))

    def _resolve(rel: str | None) -> str | None:
        if rel is None or os.path.isabs(rel) or os.path.exists(rel):
            return rel
        candidate = os.path.join(config_dir, rel)
        return candidate if os.path.exists(candidate) else rel

    if args.reference is None and cfg.reference:
        args.reference = _resolve(cfg.reference)
    if args.reads is None and cfg.reads:
        args.reads = [_resolve(r) for r in cfg.reads]
    if args.project_name is None and cfg.project_name:
        args.project_name = cfg.project_name
    if args.output is None and cfg.output_path and args.project_name:
        args.output = os.path.join(cfg.output_path, f"{args.project_name}.theta.fasta")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _apply_config_defaults(args)
    if args.output is None and args.project_name:
        args.output = f"{args.project_name}.theta.fasta"

    if not args.reference:
        parser.error("--reference is required (directly, or via --config)")
    if not args.reads:
        parser.error("--reads is required (directly, or via --config)")
    if not args.output:
        args.output = "theta.fasta"

    ref_path = resolve_reference(args.reference, args.cache_dir)
    ref_header, ref_seq = read_fasta(ref_path)
    if not args.quiet:
        print(f"reference: {ref_header} ({len(ref_seq)}bp) [{ref_path}]", file=sys.stderr)

    reads = read_reads(args.reads)
    if not reads:
        parser.error(f"no reads loaded from {args.reads}")
    if not args.quiet:
        print(f"reads: {len(reads)} from {args.reads}", file=sys.stderr)

    t0 = time.time()
    if args.hybrid:
        align_read_fn = build_align_read_fn(args.tau, args.recursive, args.min_leftover_len)
        result = run_asem_hybrid_loop(
            theta_init=ref_seq,
            reads=reads,
            align_read_fn=align_read_fn,
            contig_tau=args.contig_tau,
            min_anchor_len=args.min_anchor_len,
            w=args.w,
            max_iterations=args.max_iterations,
            min_overlap=args.min_overlap,
            n_workers=args.threads,
        )
    else:
        result = run_asem(
            theta_init=ref_seq,
            reads=reads,
            tau=args.tau,
            w=args.w,
            max_iterations=args.max_iterations,
            n_workers=args.threads,
            recursive=args.recursive,
            min_leftover_len=args.min_leftover_len,
        )
    elapsed = time.time() - t0

    if not args.quiet:
        for stats in result.history:
            extra = ""
            if args.hybrid:
                extra = f" contigs={stats.n_contigs} anchored={stats.n_contigs_anchored}"
            print(
                f"  iter {stats.iteration}: placed={stats.n_placed} "
                f"unplaced={stats.n_unplaced} theta_changed={stats.theta_changed}{extra}",
                file=sys.stderr,
            )
        print(f"done in {elapsed:.1f}s, {len(result.history)} iterations", file=sys.stderr)

    out_header = f"{ref_header} [refined by asem-refine v{__version__}]"
    write_fasta(args.output, out_header, result.theta)
    if not args.quiet:
        print(f"wrote {len(result.theta)}bp -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
