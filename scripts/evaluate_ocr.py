"""OCR + FSSAI test bench: compare the pipeline's readings with dataset_ocr/ground_truth_labels.json.

For every labelled photo it runs the full packaged-food pipeline and reports:
  * per-nutrient accuracy  (reading within 10 % or 0.05 g of the label value)
  * verdict agreement      (pipeline verdict vs the verdict implied by the true values)

Writes models/ocr_metrics.json.

Usage:
    python scripts/evaluate_ocr.py [--real-only]
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foodguard import config, pipeline  # noqa: E402

FIELDS = ["added_sugar_g", "total_sugar_g", "saturated_fat_g", "sodium_g"]


def close(got, truth) -> bool:
    return got is not None and abs(got - truth) <= max(0.10 * truth, 0.05)


def expected_verdict(entry: dict) -> str:
    """Verdict the system SHOULD give if it read the label perfectly (no expiry date entered)."""
    limits = config.load_thresholds()["packaged_categories"][entry["category"]]["limits"]
    values = entry["per_100"]
    for key, limit in limits.items():
        value = values.get(key)
        if value is None and key == "added_sugar_g":
            value = values.get("total_sugar_g")
        if value is not None and value > limit:
            return "UNSAFE"
    if any(values.get(k) is None for k in limits):
        return "CAUTION"
    return "SAFE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-only", action="store_true", help="skip synthetic_* images")
    args = parser.parse_args()

    gt = json.loads((config.DATASET_OCR_DIR / "ground_truth_labels.json").read_text(encoding="utf-8"))
    per_field = {f: {"correct": 0, "total": 0} for f in FIELDS}
    verdicts = {"agree": 0, "total": 0}
    rows = []

    for entry in gt["labels"]:
        path = config.DATASET_OCR_DIR / "raw_images" / entry["image"]
        if not path.exists() or (args.real_only and entry.get("synthetic")):
            continue
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        result = pipeline.analyse([img], entry["category"])
        packaged = result.get("packaged") or {}
        got = {k: v["value"] for k, v in packaged.get("nutrients", {}).items()}

        row = {"image": entry["image"], "verdict": result["verdict"]["status"]}
        for f in FIELDS:
            truth = entry["per_100"].get(f)
            if truth is None:
                continue
            ok = close(got.get(f), truth)
            per_field[f]["total"] += 1
            per_field[f]["correct"] += ok
            row[f] = f"{got.get(f)} vs {truth} {'OK' if ok else 'X'}"
        want = expected_verdict(entry)
        verdicts["total"] += 1
        verdicts["agree"] += result["verdict"]["status"] == want
        row["expected_verdict"] = want
        rows.append(row)
        print(json.dumps(row))

    if not rows:
        sys.exit("No labelled images found in dataset_ocr/raw_images/.")
    metrics = {
        "images": len(rows),
        "per_field_accuracy": {f: round(v["correct"] / v["total"], 3) if v["total"] else None
                               for f, v in per_field.items()},
        "verdict_agreement": round(verdicts["agree"] / verdicts["total"], 3),
        "rows": rows,
    }
    config.MODELS_DIR.mkdir(exist_ok=True)
    (config.MODELS_DIR / "ocr_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nPer-field accuracy:", metrics["per_field_accuracy"])
    print("Verdict agreement :", metrics["verdict_agreement"])


if __name__ == "__main__":
    main()
