"""
Zip the contents of `data/` so they can be uploaded to Google Colab.

Run from the repository root:

    python scripts/zip_data.py                       # everything in data/ -> data_colab.zip
    python scripts/zip_data.py -d ds003774-download  # only one dataset
    python scripts/zip_data.py --subjects 5          # only the first 5 subjects of each BIDS dataset (for testing)

Files inside the archive are rooted at `data/`, so extracting into the Colab working
directory reproduces the layout the notebook expects (`DATA_PATH = Path("data")`):

    from google.colab import drive
    drive.mount("/content/drive")
    !unzip -q /content/drive/MyDrive/data_colab.zip -d /content

Notes:
    - The datasets are git-annex checkouts: the large files are symlinks into `.git/annex`.
      Symlinks are followed so the real contents are stored, and `.git`/`.datalad` are left out.
      A broken symlink means the annexed file was never downloaded; the script stops and lists them.
    - `sourcedata/` (ds003774's raw recordings of an unrelated task) is skipped unless
      `--include-sourcedata` is given.
    - EEG `.set` and `.wav` files barely compress, so by default files are stored without
      compression, which is much faster. Use `--compress` to deflate them anyway.
"""

import argparse
import sys
import zipfile
from pathlib import Path

EXCLUDED_DIRS = {".git", ".datalad", ".ipynb_checkpoints", "__pycache__"}
EXCLUDED_SUFFIXES = {".lock"}
EXCLUDED_NAMES = {".gitattributes", ".gitignore", ".DS_Store"}
KEEP_AT_ROOT = {"README.md"}  # explanatory file in data/, not a dataset


def is_excluded(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    return (
        any(part in EXCLUDED_DIRS for part in rel.parts)
        or path.suffix in EXCLUDED_SUFFIXES
        or path.name in EXCLUDED_NAMES
    )


def collect_files(data_dir: Path, datasets: list[str] | None, n_subjects: int | None, include_sourcedata: bool):
    """Return the list of files to archive, paired with their path inside `data_dir`."""
    if datasets:
        entries = [data_dir / name for name in datasets]
        missing = [str(e) for e in entries if not e.exists()]
        if missing:
            sys.exit(f"Not found: {', '.join(missing)}")
    else:
        entries = sorted(p for p in data_dir.iterdir() if p.name not in KEEP_AT_ROOT)

    files = []
    for entry in entries:
        if entry.is_file():
            files.append(entry)
            continue
        kept_subjects = (
            sorted(p.name for p in entry.glob("sub-*") if p.is_dir())[:n_subjects]
            if n_subjects is not None
            else None
        )
        for path in sorted(entry.rglob("*")):
            if is_excluded(path, data_dir) or not (path.is_file() or path.is_symlink()):
                continue
            top = path.relative_to(entry).parts[0]
            if top == "sourcedata" and not include_sourcedata:
                continue
            if kept_subjects is not None:
                if top.startswith("sub-") and top not in kept_subjects:
                    continue
            files.append(path)
    return files


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="directory holding the datasets (default: data)")
    parser.add_argument("-o", "--output", type=Path, default=Path("data_colab.zip"), help="zip file to write (default: data_colab.zip)")
    parser.add_argument("-d", "--datasets", nargs="+", help="names of dataset directories or files in data/ to include (default: all)")
    parser.add_argument("--subjects", type=int, help="only include the first N `sub-*` directories of each dataset")
    parser.add_argument("--include-sourcedata", action="store_true", help="also include each dataset's `sourcedata/` (a different task, ~half the size of ds003774; not used by the notebook)")
    parser.add_argument("--compress", action="store_true", help="deflate files instead of storing them")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        sys.exit(f"Data directory not found: {data_dir}")
    output = args.output.resolve()

    files = [f for f in collect_files(data_dir, args.datasets, args.subjects, args.include_sourcedata) if f != output]

    # Broken symlinks are annexed files that were never fetched
    broken = [f for f in files if not f.exists()]
    if broken:
        listing = "\n  ".join(str(f.relative_to(data_dir)) for f in broken[:10])
        sys.exit(
            f"{len(broken)} file(s) are broken symlinks (annexed content not downloaded), e.g.:\n  {listing}\n"
            "Fetch them first (e.g. `datalad get` or `git annex get`)."
        )

    total = sum(f.stat().st_size for f in files)  # stat follows symlinks
    print(f"Archiving {len(files)} files ({human(total)}) from {data_dir}")

    compression = zipfile.ZIP_DEFLATED if args.compress else zipfile.ZIP_STORED
    done = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(output, "w", compression=compression, allowZip64=True) as zf:
            for i, f in enumerate(files, 1):
                arcname = Path("data") / f.relative_to(data_dir)
                zf.write(f, arcname)  # follows symlinks
                done += f.stat().st_size
                print(f"\r[{i}/{len(files)}] {human(done)} / {human(total)}", end="", flush=True)
    except BaseException:
        output.unlink(missing_ok=True)  # don't leave a truncated archive behind
        raise
    print(f"\nWrote {output} ({human(output.stat().st_size)})")


if __name__ == "__main__":
    main()
