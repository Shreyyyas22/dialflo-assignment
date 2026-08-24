#!/usr/bin/env python3
"""
Evaluation harness for Voice Gender/Age/Quality Analyzer.

Runs the model against a dataset of labeled audio files (Common Voice, VoxCeleb,
or custom CSV) and prints accuracy + confidence calibration metrics.

Supports:
- Common Voice validated.tsv / train.tsv (columns: path, age, gender)
- Custom CSV: file, gender, age, language  (headers auto-detected)
- JSON labels: [{"file": "...", "gender":"male","age":"31-45"}, ...]
- Directory inference: if no labels file, scans sample_audio/ and derives
  pseudo-labels from filename heuristics for demo purposes.

Metrics:
- Gender accuracy, Age bracket accuracy, Language accuracy (if labels present)
- Confusion matrix for gender
- Expected Calibration Error (ECE) for gender & age confidence
- Quality distribution + latency stats

Usage:
    python scripts/evaluate.py --data-dir sample_audio --labels labels.csv
    python scripts/evaluate.py --data-dir /path/to/common_voice --tsv validated.tsv
    python scripts/evaluate.py --data-dir sample_audio  # heuristic labels (demo)
    python scripts/evaluate.py --data-dir sample_audio --output results.json

"""
import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Allow running as `python scripts/evaluate.py` from repo root without PYTHONPATH hacks
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.inference import run_inference
from app.ml.model_manager import load_model
from app.services.age_service import process_age_prediction
from app.services.audio_service import decode_and_preprocess_audio
from app.services.gender_service import process_gender_prediction
from app.services.language_service import detect_language
from app.services.quality_service import analyze_audio_quality

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".weba", ".aac"}
AGE_BRACKETS = ["18-30", "31-45", "46-60", "60+", "unknown"]
GENDERS = ["male", "female", "unknown"]


def discover_audio_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def normalize_gender(g: str | None) -> str | None:
    if not g:
        return None
    g = str(g).strip().lower()
    if g in ("male", "m", "man", "boy"):
        return "male"
    if g in ("female", "f", "woman", "girl"):
        return "female"
    if g in ("unknown", "other", "child", ""):
        return "unknown"
    return g  # keep as-is for reporting


def normalize_age(a: str | None) -> str | None:
    if not a:
        return None
    a = str(a).strip()
    # Common Voice uses: teens, twenties, thirties, fourties, fifties, sixties, seventies, eighties
    cv_map = {
        "teens": "18-30",
        "twenties": "18-30",
        "thirties": "31-45",
        "fourties": "31-45",
        "forties": "31-45",
        "fifties": "46-60",
        "sixties": "60+",
        "seventies": "60+",
        "eighties": "60+",
        "nineties": "60+",
    }
    lower = a.lower()
    if lower in cv_map:
        return cv_map[lower]
    if lower in AGE_BRACKETS:
        return lower
    # Try numeric age -> bracket
    try:
        age_num = float(a)
        if age_num < 31:
            return "18-30"
        if age_num <= 45:
            return "31-45"
        if age_num <= 60:
            return "46-60"
        return "60+"
    except ValueError:
        pass
    # Already bracket-like e.g. "18-30"
    if "-" in a or a == "60+":
        return a
    return a


def load_labels(labels_path: Path | None, data_dir: Path) -> dict[str, dict]:
    """
    Returns dict: normalized_key -> {gender, age, language}
    Keys are both full resolved path and basename for flexible matching.
    """
    label_map: dict[str, dict] = {}
    if labels_path is None:
        # Auto-detect in data_dir
        candidates = [
            data_dir / "labels.csv",
            data_dir / "metadata.csv",
            data_dir / "validated.tsv",
            data_dir / "train.tsv",
            data_dir / "labels.json",
            data_dir / "metadata.json",
        ]
        for c in candidates:
            if c.exists():
                labels_path = c
                logger.info(f"Auto-detected labels file: {c}")
                break

    if labels_path is None or not labels_path.exists():
        logger.info("No labels file found — will use filename heuristics (demo mode).")
        return {}

    logger.info(f"Loading labels from {labels_path}")
    suffix = labels_path.suffix.lower()

    try:
        if suffix == ".json":
            data = json.loads(labels_path.read_text(encoding="utf-8"))
            # Support list or dict
            if isinstance(data, dict):
                data = [data]
            for entry in data:
                fname = entry.get("file") or entry.get("path") or entry.get("filename")
                if not fname:
                    continue
                gender = normalize_gender(entry.get("gender"))
                age = normalize_age(entry.get("age") or entry.get("age_bracket") or entry.get("bracket"))
                language = entry.get("language") or entry.get("lang") or entry.get("locale")
                if language:
                    language = str(language).strip().lower().split("-")[0].split("_")[0]
                else:
                    language = None
                rec = {"gender": gender, "age": age, "language": language}
                # Store multiple keys for matching
                for key in {fname, Path(fname).name, str((data_dir / fname).resolve()), Path(fname).stem}:
                    label_map[key] = rec
                    label_map[Path(key).name] = rec
            return label_map

        # CSV / TSV
        delimiter = "\t" if suffix == ".tsv" else ","
        # Try to sniff delimiter for csv
        with labels_path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)
            f.seek(0)
            if "\t" in sample and "," not in sample.splitlines()[0][:200]:
                delimiter = "\t"
            elif "," in sample:
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
                    delimiter = dialect.delimiter
                except csv.Error:
                    logger.debug("CSV dialect sniff failed, using default delimiter", exc_info=True)
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None:
                logger.warning("Labels file has no header — skipping.")
                return {}
            # Normalize fieldnames lower
            [h.strip().lower() for h in reader.fieldnames]
            # Map expected fields
            for row in reader:
                # Build lower-keyed row
                row_lower = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                # Find file key
                fname = (
                    row_lower.get("file")
                    or row_lower.get("path")
                    or row_lower.get("filename")
                    or row_lower.get("clip")
                    or row_lower.get("audio")
                )
                if not fname:
                    # Common Voice has 'path' without extension? e.g., common_voice_en_123.wav
                    continue
                gender = normalize_gender(row_lower.get("gender") or row_lower.get("sex"))
                # Age may be under 'age' or 'age_group'
                age_raw = row_lower.get("age") or row_lower.get("age_group") or row_lower.get("age_bracket") or row_lower.get("bracket")
                age = normalize_age(age_raw)
                language = row_lower.get("language") or row_lower.get("lang") or row_lower.get("locale")
                if language:
                    language = str(language).strip().lower().split("-")[0].split("_")[0].split("/")[0]
                else:
                    language = None
                rec = {"gender": gender, "age": age, "language": language}
                # Store multiple matching keys
                label_map[fname] = rec
                label_map[Path(fname).name] = rec
                # Also without extension stem
                label_map[Path(fname).stem] = rec
                # Full path attempts
                label_map[str((data_dir / fname).resolve())] = rec

    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to load labels: {e}")
        return {}
    logger.info(f"Loaded {len(label_map)} label entries (with alias keys).")
    return label_map


def heuristic_label_from_filename(path: Path) -> dict:
    """Fallback pseudo-label derived from filename for demo when no real labels."""
    name = path.stem.lower()
    gender = None
    age = None
    if "female" in name or "woman" in name or "girl" in name:
        gender = "female"
    elif "male" in name:
        gender = "male"
    if "young" in name or "youth" in name or "20" in name:
        age = "18-30"
    elif "adult" in name and "young" not in name:
        age = "31-45"
    elif "senior" in name or "old" in name or "60" in name:
        age = "60+"
    elif "real_speech" in name:
        gender = "male"
        age = "31-45"
    return {"gender": gender, "age": age, "language": "en"}


def find_label_for_file(file_path: Path, label_map: dict, data_dir: Path) -> dict:
    candidates = [
        str(file_path.resolve()),
        str(file_path),
        file_path.name,
        file_path.stem,
        str(file_path.relative_to(data_dir)) if file_path.is_relative_to(data_dir) else None,
    ]
    for c in candidates:
        if c and c in label_map:
            return label_map[c]
    # Try basename without extension alias
    if file_path.name in label_map:
        return label_map[file_path.name]
    return {}


def compute_ece(confidences: list[float], accuracies: list[int], n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    if not confidences:
        return 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # Include high edge in last bin
        if i == n_bins - 1:
            mask = [(c >= low and c <= high) for c in confidences]
        else:
            mask = [(c >= low and c < high) for c in confidences]
        bin_count = sum(mask)
        if bin_count == 0:
            continue
        bin_acc = sum(a for a, m in zip(accuracies, mask) if m) / bin_count
        bin_conf = sum(c for c, m in zip(confidences, mask) if m) / bin_count
        ece += abs(bin_acc - bin_conf) * (bin_count / total)
    return float(ece)


def evaluate_file(file_path: Path) -> dict | None:
    audio_bytes = file_path.read_bytes()
    t0 = time.perf_counter()
    try:
        signal, sr = decode_and_preprocess_audio(audio_bytes)
        quality = analyze_audio_quality(signal, sr)
        if quality.classification == "insufficient":
            gender_res = {"value": "unknown", "confidence": 0.0}
            age_res = {"bracket": "unknown", "confidence": 0.0}
            language_res = {"code": "unknown", "confidence": 0.0}
        else:
            raw_age_scalar, gender_probs = run_inference(signal, sr)
            g = process_gender_prediction(gender_probs)
            a = process_age_prediction(raw_age_scalar)
            l = detect_language(signal, sr, quality.classification)
            gender_res = {"value": g.value, "confidence": g.confidence}
            age_res = {"bracket": a.bracket, "confidence": a.confidence}
            language_res = {"code": l.code, "confidence": l.confidence}
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "gender": gender_res,
            "age": age_res,
            "language": language_res,
            "quality": quality.classification,
            "duration": quality.metrics.duration_seconds,
            "latency_ms": elapsed_ms,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to evaluate {file_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate voice analyzer on labeled dataset")
    parser.add_argument("--data-dir", type=str, default="sample_audio", help="Directory containing audio files")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels CSV/TSV/JSON file (auto-detected if not provided)")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path for detailed results")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to evaluate (0 = all)")
    parser.add_argument("--bins", type=int, default=10, help="Number of bins for ECE calculation")
    parser.add_argument("--demo-labels", action="store_true", help="Force using filename heuristics even if labels file exists")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data dir does not exist: {data_dir}")
        sys.exit(1)

    labels_path = Path(args.labels) if args.labels else None
    if args.demo_labels:
        label_map = {}
    else:
        label_map = load_labels(labels_path, data_dir)

    audio_files = discover_audio_files(data_dir)
    if not audio_files:
        logger.error(f"No audio files found in {data_dir}")
        sys.exit(1)

    if args.limit and args.limit > 0:
        audio_files = audio_files[: args.limit]

    logger.info(f"Discovered {len(audio_files)} audio files in {data_dir}")
    # Load model once
    logger.info("Loading model (may take a while on first run)...")
    try:
        load_model()
        logger.info("Model loaded.")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Model loading failed: {e}")
        logger.info("Continuing — evaluation will fail on inference steps. Ensure model is available.")

    # Storage for metrics
    results = []
    latencies: list[float] = []
    gender_confidences: list[float] = []
    gender_correct: list[int] = []
    age_confidences: list[float] = []
    age_correct: list[int] = []
    language_confidences: list[float] = []
    language_correct: list[int] = []

    gender_total_labeled = 0
    gender_correct_count = 0
    age_total_labeled = 0
    age_correct_count = 0
    lang_total_labeled = 0
    lang_correct_count = 0

    gender_confusion = Counter()
    quality_dist = Counter()

    n_processed = 0
    n_failed = 0
    n_skipped_no_label = 0

    for fp in audio_files:
        # Resolve expected label
        expected = find_label_for_file(fp, label_map, data_dir)
        if not expected or (not expected.get("gender") and not expected.get("age") and not expected.get("language")):
            if label_map:
                # Real labels mode but file not in map -> skip accuracy but still measure latency
                n_skipped_no_label += 1
                expected = {"gender": None, "age": None, "language": None}
            else:
                # Demo heuristic mode
                expected = heuristic_label_from_filename(fp)

        pred = evaluate_file(fp)
        if pred is None:
            n_failed += 1
            continue
        n_processed += 1
        latencies.append(pred["latency_ms"])
        quality_dist[pred["quality"]] += 1

        # Gender
        true_gender = expected.get("gender")
        pred_gender = pred["gender"]["value"]
        pred_gender_conf = pred["gender"]["confidence"]
        gender_confusion[(true_gender or "unlabeled", pred_gender)] += 1
        if true_gender and true_gender != "unknown":
            gender_total_labeled += 1
            is_correct = int(pred_gender == true_gender)
            gender_correct_count += is_correct
            gender_confidences.append(pred_gender_conf)
            gender_correct.append(is_correct)
        # Age
        true_age = expected.get("age")
        pred_age = pred["age"]["bracket"]
        pred_age_conf = pred["age"]["confidence"]
        if true_age and true_age != "unknown":
            age_total_labeled += 1
            is_correct_a = int(pred_age == true_age)
            age_correct_count += is_correct_a
            age_confidences.append(pred_age_conf)
            age_correct.append(is_correct_a)
        # Language
        true_lang = expected.get("language")
        pred_lang = pred["language"]["code"]
        pred_lang_conf = pred["language"]["confidence"]
        if true_lang and true_lang != "unknown":
            lang_total_labeled += 1
            is_correct_l = int(pred_lang == true_lang)
            lang_correct_count += is_correct_l
            language_confidences.append(pred_lang_conf)
            language_correct.append(is_correct_l)

        results.append(
            {
                "file": str(fp.relative_to(data_dir) if fp.is_relative_to(data_dir) else fp.name),
                "true_gender": true_gender,
                "pred_gender": pred_gender,
                "gender_confidence": pred_gender_conf,
                "true_age": true_age,
                "pred_age": pred_age,
                "age_confidence": pred_age_conf,
                "true_language": true_lang,
                "pred_language": pred_lang,
                "language_confidence": pred_lang_conf,
                "quality": pred["quality"],
                "duration": pred["duration"],
                "latency_ms": round(pred["latency_ms"], 2),
            }
        )
        # Per-file log
        logger.info(
            f"[{n_processed}/{len(audio_files)}] {fp.name}: "
            f"true gender={true_gender} pred={pred_gender}({pred_gender_conf}) | "
            f"true age={true_age} pred={pred_age}({pred_age_conf}) | "
            f"lang={pred_lang}({pred_lang_conf}) | q={pred['quality']} | {pred['latency_ms']:.1f}ms"
        )

    # Compute metrics
    gender_acc = (gender_correct_count / gender_total_labeled) if gender_total_labeled else None
    age_acc = (age_correct_count / age_total_labeled) if age_total_labeled else None
    lang_acc = (lang_correct_count / lang_total_labeled) if lang_total_labeled else None

    gender_ece = compute_ece(gender_confidences, gender_correct, n_bins=args.bins) if gender_confidences else None
    age_ece = compute_ece(age_confidences, age_correct, n_bins=args.bins) if age_confidences else None
    lang_ece = compute_ece(language_confidences, language_correct, n_bins=args.bins) if language_confidences else None

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    avg_gender_conf = float(np.mean(gender_confidences)) if gender_confidences else 0.0
    avg_age_conf = float(np.mean(age_confidences)) if age_confidences else 0.0

    print("\n" + "=" * 72)
    print(" EVALUATION REPORT ".center(72, "="))
    print("=" * 72)
    print(f"Dataset dir:          {data_dir} ({len(audio_files)} files, {n_processed} processed, {n_failed} failed)")
    if label_map or n_skipped_no_label == 0:
        print(f"Labels:               {'heuristic (demo)' if not label_map else (str(labels_path) if labels_path else 'auto-detected')}")
    else:
        print(f"Labels:               {n_skipped_no_label} files without labels (excluded from accuracy)")
    print(f"Quality distribution: {dict(quality_dist)}")
    print("-" * 72)
    print(" Accuracy:")
    if gender_total_labeled:
        print(f"  Gender:   {gender_correct_count}/{gender_total_labeled} = {gender_acc:.3f}  (avg conf {avg_gender_conf:.3f})")
    else:
        print("  Gender:   no labeled data")
    if age_total_labeled:
        print(f"  Age:      {age_correct_count}/{age_total_labeled} = {age_acc:.3f}  (avg conf {avg_age_conf:.3f})")
    else:
        print("  Age:      no labeled data")
    if lang_total_labeled:
        print(f"  Language: {lang_correct_count}/{lang_total_labeled} = {lang_acc:.3f}")
    else:
        print("  Language: no labeled data")
    print("-" * 72)
    print(" Calibration (ECE — lower is better):")
    if gender_ece is not None:
        print(f"  Gender ECE: {gender_ece:.4f}  (bins={args.bins})")
    if age_ece is not None:
        print(f"  Age ECE:    {age_ece:.4f}  (bins={args.bins})")
    if lang_ece is not None:
        print(f"  Language ECE: {lang_ece:.4f}")
    print("-" * 72)
    print(" Latency:")
    print(f"  Avg: {avg_latency:.2f} ms  |  P50: {p50:.2f} ms  |  P95: {p95:.2f} ms")
    print("-" * 72)
    print(" Gender Confusion (true -> pred):")
    for (true_g, pred_g), cnt in sorted(gender_confusion.items()):
        print(f"  {true_g:12} -> {pred_g:8} : {cnt}")
    print("=" * 72)
    if not label_map:
        print(" NOTE: Used filename heuristics as pseudo-labels. For real evaluation,")
        print(" provide --labels labels.csv with columns: file,gender,age (see sample).")

    if args.output:
        out_path = Path(args.output)
        payload = {
            "summary": {
                "total_files": len(audio_files),
                "processed": n_processed,
                "failed": n_failed,
                "gender_accuracy": gender_acc,
                "gender_total_labeled": gender_total_labeled,
                "gender_ece": gender_ece,
                "age_accuracy": age_acc,
                "age_total_labeled": age_total_labeled,
                "age_ece": age_ece,
                "language_accuracy": lang_acc,
                "language_total_labeled": lang_total_labeled,
                "language_ece": lang_ece,
                "quality_distribution": dict(quality_dist),
                "latency_avg_ms": avg_latency,
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
            },
            "per_file": results,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nDetailed JSON report written to: {out_path}")

    # Exit code: 0 if at least one file processed
    sys.exit(0 if n_processed > 0 else 1)


if __name__ == "__main__":
    main()
