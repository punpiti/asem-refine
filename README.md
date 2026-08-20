# asem-refine

Sample-specific reference sequence estimation from low-depth short reads,
for cases where the closest available reference is still meaningfully
divergent from the sample (e.g. a related-but-not-identical species'
mitochondrial genome). Implements ASEM (EM / match-state-only Profile HMM
consensus) with optional recursive partial alignment and local de novo
gap-filling (ASEM-Hybrid) for regions no single read can reach.

See the accompanying paper for the method and its validation.

## Install

```bash
pip install git+https://github.com/punpiti/asem-refine.git
```

## Usage

```bash
# Local reference file
asem-refine --reference ref.fasta --reads reads.fastq --output theta.fasta

# Or fetch the reference from NCBI by accession
asem-refine --reference NC_012920.1 --reads reads.fastq -o theta.fasta

# Or reuse an existing NOVOPlasty config.txt
asem-refine -c config.txt
```

`--reference` accepts either a local FASTA file or an NCBI nucleotide
accession (fetched and cached under `~/.cache/asem-refine/ncbi/` by
default). `--reads` accepts one or more FASTA/FASTQ files, plain or
gzip-compressed.

Run `asem-refine --help` for the full option list, including the
hybrid/recursive toggles and EM hyperparameters (`--tau`, `--w`,
`--max-iterations`).

**Reproducing the paper's headline ASEM-Hybrid numbers**: the benchmark
grid reported in the paper ran ASEM-Hybrid on top of ASEM *without*
recursive partial alignment. `--recursive` defaults to on here (it never
measured worse and helps on real, noisy reads), so add `--no-recursive`
to match that exact configuration:

```bash
asem-refine --reference ref.fasta --reads reads.fastq --no-recursive -o theta.fasta
```

Hybrid+recursive together has not been separately benchmarked.

### NOVOPlasty config.txt compatibility

`-c config.txt` reads the same field names NOVOPlasty's config.txt uses
(`Project name`, `Reference sequence` / `Seed Input`, `Combined reads` /
`Forward reads` / `Reverse reads`, `Output path`), so an existing
NOVOPlasty setup can be pointed at this tool without re-deriving paths by
hand. Fields specific to NOVOPlasty's seed-and-extend algorithm (Genome
Range, K-mer, Chloroplast sequence, Heteroplasmy, ...) are not applicable
here and are ignored.
