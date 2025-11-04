#!/usr/bin/env python3
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import sys

"""
Convert all pickle files in a directory to CSV files saved in the same directory.

Configuration (edit these values in this file instead of passing CLI args):
    DIRECTORY  - path to directory to scan (string or Path). Example: " ." or "C:/data"
    RECURSIVE  - True to search subdirectories, False to scan only the directory
    OVERWRITE  - True to overwrite existing CSV files, False to skip them
"""

# ----- Configuration: edit these -----
DIRECTORY = Path(r"C:\Users\leyan\OneDrive\NTU\Y4 Sem1\SC4000 Machine Learning\SC4000 Project\run1 new")            # e.g. Path("/path/to/dir")
RECURSIVE = False
OVERWRITE = False
# -------------------------------------

PKL_EXTS = {".pkl", ".pickle", ".p"}


def load_pickle(path: Path):
    # Try pandas first (works for pd objects), then fallback to pickle
    try:
        return pd.read_pickle(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def obj_to_dataframe(obj):
    # Return a DataFrame for common types or None if not convertible
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    if isinstance(obj, (list, tuple, np.ndarray)):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return None
    if isinstance(obj, dict):
        try:
            # For dict of scalars -> one-row DataFrame; for dict of sequences -> columns
            return pd.DataFrame(obj)
        except Exception:
            # fallback: items as two columns
            return pd.DataFrame(list(obj.items()), columns=["key", "value"])
    return None


def convert_dir(directory: Path, recursive: bool = False, overwrite: bool = False):
    if not directory.exists() or not directory.is_dir():
        print(f"Directory not found: {directory}", file=sys.stderr)
        return 1

    pattern = "**/*" if recursive else "*"
    files = [p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() in PKL_EXTS]

    if not files:
        print("No pickle files found.")
        return 0

    for p in files:
        try:
            obj = load_pickle(p)
        except Exception as e:
            print(f"Failed to load {p}: {e}", file=sys.stderr)
            continue

        df = obj_to_dataframe(obj)
        if df is None:
            print(f"Unsupported object in {p} (type={type(obj)}), skipping.", file=sys.stderr)
            continue

        out_path = p.with_suffix(".csv")
        if out_path.exists() and not overwrite:
            print(f"Skipping existing CSV (set OVERWRITE=True to replace): {out_path}")
            continue

        try:
            # index=True preserves DataFrame index; change to index=False if not desired
            df.to_csv(out_path, index=True)
            print(f"Wrote: {out_path}")
        except Exception as e:
            print(f"Failed to write CSV for {p}: {e}", file=sys.stderr)

    return 0


def main():
    directory = Path(DIRECTORY).expanduser().resolve()
    sys.exit(convert_dir(directory, recursive=RECURSIVE, overwrite=OVERWRITE))


if __name__ == "__main__":
    main()
