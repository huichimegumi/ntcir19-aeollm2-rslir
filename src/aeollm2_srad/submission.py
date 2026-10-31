"""Combine OSODT content dimensions with SRAD readability by document identity."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aeollm2_osodt.submission import COLUMNS, read_predictions, write_submission


def merge_predictions(osodt_path: Path, srad_path: Path) -> pd.DataFrame:
    osodt = read_predictions(osodt_path)
    # SRAD deliberately leaves the other three score columns blank. Its TSV must
    # be parsed with tabs rather than the generic whitespace submission reader.
    srad = pd.read_csv(srad_path, sep="\t", encoding="utf-8-sig", dtype=str,
                       keep_default_na=False)
    if list(srad.columns) != COLUMNS or srad.empty:
        raise ValueError("SRAD requires a nonempty, headed seven-column TSV")
    qids = pd.to_numeric(srad["questionId"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(qids).all() or (qids <= 0).any() or (qids != np.floor(qids)).any():
        raise ValueError("SRAD questionId must be a positive integer")
    srad["questionId"] = qids.astype(int)
    if not (pd.to_numeric(srad["taskId"], errors="raise") == 0).all():
        raise ValueError("SRAD taskId must be 0")
    if not srad["answerId"].str.fullmatch(r"Doc_\d{3,}").all():
        raise ValueError("SRAD answerId must use official Doc_### IDs")
    keys = ["questionId", "answerId"]
    if srad.duplicated(keys).any():
        raise ValueError("Duplicate SRAD document keys")
    scores = pd.to_numeric(srad["readability_score"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 10)).any():
        raise ValueError("SRAD readability must be finite and in [0, 10]")
    srad["readability_score"] = scores
    old_keys = set(map(tuple, osodt[keys].to_numpy()))
    new_keys = set(map(tuple, srad[keys].to_numpy()))
    if old_keys != new_keys:
        raise ValueError(f"Document sets differ: missing SRAD={len(old_keys-new_keys)}, extra SRAD={len(new_keys-old_keys)}")
    aligned = srad.set_index(keys)["readability_score"]
    merged = osodt.copy()
    merged["readability_score"] = [aligned.loc[(q, a)] for q, a in osodt[keys].itertuples(index=False, name=None)]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osodt", type=Path, required=True)
    parser.add_argument("--srad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.osodt.resolve(), args.srad.resolve()}:
        parser.error("Output must not overwrite either input")
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    write_submission(merge_predictions(args.osodt, args.srad), args.output)
    print(f"SRAD+OSODT submission: {args.output}")


if __name__ == "__main__":
    main()
