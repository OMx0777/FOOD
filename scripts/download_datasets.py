"""Download the produce spoilage dataset from Kaggle and build train/validation/test splits.

Dataset: "Fresh and Stale Images of Fruits and Vegetables" (raghavrpotdar)
12 classes: fresh_/stale_ x apple, banana, bitter_gourd, capsicum, orange, tomato

IMPORTANT - leakage-safe split
------------------------------
The Kaggle images are heavily augmented: one source photo appears many times as
"Copy of X.jpg_0_123.jpg", "rotated_by_15_X.png", "vertical_flip_X.png", ...
A plain random split would put copies of the SAME photo into both train and
test, inflating accuracy. We recover the source-photo id of every file and split
by that id (GroupShuffleSplit idea), so test images are photos the model has
never seen in any augmented form.

Usage:
    python scripts/download_datasets.py            # download + split
    python scripts/download_datasets.py --skip-download
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_raw" / "fresh_stale"
OUT_DIR = ROOT / "dataset_vision"
KAGGLE_SLUG = "raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables"

SPLITS = {"train": 0.70, "validation": 0.15, "test": 0.15}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUG_PREFIX = re.compile(
    r"^(saltandpepper_|rotated_by_\d+_|translation_|vertical_flip_|horizontal_flip_|copy of )",
    re.IGNORECASE,
)


def source_photo_id(filename: str) -> str:
    """'Copy of IMG_1.jpg_0_77.jpg' -> 'img_1';  'rotated_by_15_Screen Shot.png' -> 'screen shot'."""
    name = filename
    while True:
        stripped = AUG_PREFIX.sub("", name)
        if stripped == name:
            break
        name = stripped
    name = re.sub(r"_\d+_\d+\.(jpe?g|png)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"( \(\d+\))?\.(jpe?g|png|webp|bmp)$", "", name, flags=re.IGNORECASE)
    return name.strip().lower()


def download():
    import kagglehub

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_SLUG} (~1.4 GB) ...")
    path = kagglehub.dataset_download(KAGGLE_SLUG, output_dir=str(RAW_DIR))
    print("Downloaded to", path)


def split_groups(groups: dict[str, list[Path]], rng: random.Random) -> dict[str, list[Path]]:
    """Assign whole source-photo groups to splits, balancing by image count."""
    keys = list(groups)
    rng.shuffle(keys)
    total = sum(len(v) for v in groups.values())
    assigned = {s: [] for s in SPLITS}
    counts = {s: 0 for s in SPLITS}

    # Guarantee at least one group in validation and test when possible.
    order = ["validation", "test"] if len(keys) >= 3 else []
    for split in order:
        k = keys.pop()
        assigned[split] += groups[k]
        counts[split] += len(groups[k])

    for k in keys:
        deficit = {s: SPLITS[s] * total - counts[s] for s in SPLITS}
        split = max(deficit, key=deficit.get)
        assigned[split] += groups[k]
        counts[split] += len(groups[k])
    return assigned


def link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)  # hard link: no extra disk space on the same drive
    except OSError:
        shutil.copy2(src, dst)


def build_splits(seed: int):
    class_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not class_dirs:
        sys.exit(f"No class folders found in {RAW_DIR}. Run without --skip-download first.")

    if OUT_DIR.exists():
        for split in SPLITS:
            shutil.rmtree(OUT_DIR / split, ignore_errors=True)

    rng = random.Random(seed)
    report = {"seed": seed, "classes": {}}
    for class_dir in class_dirs:
        cls = class_dir.name.lower().replace(" ", "_").replace("rotten", "stale")
        groups = defaultdict(list)
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXT:
                groups[source_photo_id(f.name)].append(f)

        assigned = split_groups(groups, rng)
        report["classes"][cls] = {"source_photos": len(groups)}
        for split, files in assigned.items():
            target = OUT_DIR / split / cls
            target.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(files):
                link_or_copy(f, target / f"{i:05d}{f.suffix.lower()}")
            report["classes"][cls][split] = len(files)
        print(f"{cls:22s} photos={len(groups):4d}  " +
              "  ".join(f"{s}={report['classes'][cls][s]}" for s in SPLITS))

    (OUT_DIR / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSplits written to {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.skip_download and not any(RAW_DIR.glob("fresh_*")):
        download()
    build_splits(args.seed)


if __name__ == "__main__":
    main()
