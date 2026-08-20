"""Fetch a reference sequence from NCBI by accession, so --reference can be
either a local FASTA path or an accession/identifier to download on demand
(e.g. NC_012920.1). Downloads are cached locally so a repeated run (or a
second tool on the same reference) doesn't re-fetch.
"""

from __future__ import annotations

import os
import re

import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Loose but practical: NCBI nucleotide accessions are 1-2 letters + digits,
# optionally a version suffix (".1"), or a bare RefSeq-style "NC_012920.1".
_ACCESSION_RE = re.compile(r"^[A-Za-z]{1,2}_?[0-9]{5,8}(\.[0-9]+)?$")


def looks_like_accession(value: str) -> bool:
    return bool(_ACCESSION_RE.match(value.strip()))


def fetch_reference(accession: str, cache_dir: str, timeout: float = 30.0) -> str:
    """Return a local FASTA path for `accession`, downloading it from NCBI
    (Entrez efetch, nucleotide database) into `cache_dir` if not already
    cached there."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{accession}.fasta")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path

    params = {"db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text"}
    resp = requests.get(EUTILS_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    if not text.startswith(">"):
        raise ValueError(
            f"NCBI did not return a FASTA record for accession {accession!r} "
            f"(got: {text[:200]!r})"
        )

    tmp_path = cache_path + ".part"
    with open(tmp_path, "w") as fh:
        fh.write(text)
    os.replace(tmp_path, cache_path)
    return cache_path


def resolve_reference(value: str, cache_dir: str) -> str:
    """--reference accepts either a local file path or an NCBI accession.
    A path that exists on disk always wins (even if it happens to look like
    an accession); otherwise, if it looks like an accession, fetch it."""
    if os.path.exists(value):
        return value
    if looks_like_accession(value):
        return fetch_reference(value, cache_dir)
    raise FileNotFoundError(
        f"--reference {value!r} is neither an existing file nor a recognizable "
        "NCBI accession (expected something like NC_012920.1)"
    )
