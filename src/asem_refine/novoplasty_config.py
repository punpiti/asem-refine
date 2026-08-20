"""Parser for NOVOPlasty-style config.txt files.

Not every NOVOPlasty option applies to ASEM-Hybrid -- NOVOPlasty is a
seed-and-extend de novo assembler (Genome Range, K-mer, Chloroplast
sequence, Heteroplasmy section, ... only make sense for that algorithm) --
but a config.txt from an existing NOVOPlasty run typically already has
everything this tool needs (Reference sequence, reads, Project name, Output
path), so accepting the same file format lets someone switch tools without
re-deriving those paths by hand. Unrecognized fields are ignored, not
rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Fields we read; NOVOPlasty-specific fields not in this set (Genome Range,
# K-mer, Chloroplast sequence, MAF, ...) are simply not looked at.
_RECOGNIZED = {
    "project name",
    "reference sequence",
    "seed input",
    "combined reads",
    "forward reads",
    "reverse reads",
    "output path",
}


@dataclass
class NovoplastyConfig:
    project_name: str | None = None
    reference: str | None = None
    reads: list[str] | None = None
    output_path: str | None = None


def parse_config(path: str) -> NovoplastyConfig:
    values: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("-") or line.endswith(":"):
                continue
            m = re.match(r"^([^=]+?)\s*=\s*(.*)$", line)
            if not m:
                continue
            key, value = m.group(1).strip().lower(), m.group(2).strip()
            if key in _RECOGNIZED and value:
                values[key] = value

    reference = values.get("reference sequence") or values.get("seed input")

    reads: list[str] = []
    if "combined reads" in values:
        reads.append(values["combined reads"])
    if "forward reads" in values:
        reads.append(values["forward reads"])
    if "reverse reads" in values:
        reads.append(values["reverse reads"])

    return NovoplastyConfig(
        project_name=values.get("project name"),
        reference=reference,
        reads=reads or None,
        output_path=values.get("output path") or None,
    )
