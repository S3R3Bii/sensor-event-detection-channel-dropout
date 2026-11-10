"""Independently recompute saved metrics using only the Python standard library."""

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from math import fsum, isclose
from pathlib import Path


def average_precision(labels, probabilities):
    positives = sum(labels)
    if not positives:
        raise ValueError("Average precision requires at least one positive")
    pairs = sorted(zip(probabilities, labels, strict=True), reverse=True)
    cumulative_tp = 0
    cumulative_n = 0
    terms = []
    for _, values in itertools.groupby(pairs, key=lambda row: row[0]):
        tied = [row[1] for row in values]
        new_tp = sum(tied)
        cumulative_tp += new_tp
        cumulative_n += len(tied)
        terms.append((new_tp / positives) * cumulative_tp / cumulative_n)
    return fsum(terms)


def same(actual, expected, message):
    if actual is None or expected is None:
        matched = actual is expected
    else:
        matched = isclose(actual, expected, abs_tol=2e-12, rel_tol=2e-12)
    if not matched:
        raise AssertionError(f"{message}: {actual!r} vs {expected!r}")


def recompute(rows, cost):
    labels = [int(r["label"]) for r in rows]
    probabilities = [float(r["probability"]) for r in rows]
    raw_probabilities = [float(r["raw_probability"]) for r in rows]
    cutoff = [float(r["confidence_threshold"]) for r in rows]
    if not all(0 <= p <= 1 for p in (*probabilities, *raw_probabilities)):
        raise AssertionError("Invalid probability")
    kept = [max(p, 1 - p) >= c for p, c in zip(probabilities, cutoff, strict=True)]
    if [str(x) for x in kept] != [r["retained"] for r in rows]:
        raise AssertionError("Saved predictions do not respect the frozen confidence threshold")
    wrong = [(p >= 0.5) != bool(y) for y, p in zip(labels, probabilities, strict=True)]
    n = len(labels)
    n_kept = sum(kept)
    wrong_kept = sum(a and b for a, b in zip(kept, wrong, strict=True))
    reliability = defaultdict(list)
    for y, p in zip(labels, probabilities, strict=True):
        reliability[min(int(p * 10), 9)].append((y, p))
    bins = []
    for i, group in sorted(reliability.items()):
        bins.append({"bin": i, "n": len(group), "event_rate": sum(y for y, _ in group) / len(group),
                     "mean_probability": fsum(p for _, p in group) / len(group)})
    return {"windows": n, "events": sum(labels), "event_prevalence": sum(labels) / n,
            "average_precision": average_precision(labels, probabilities),
            "brier": fsum((p - y) ** 2 for y, p in zip(labels, probabilities, strict=True)) / n,
            "raw_brier": fsum((p - y) ** 2 for y, p in zip(labels, raw_probabilities, strict=True)) / n,
            "accuracy": 1 - sum(wrong) / n, "coverage": n_kept / n,
            "risk": wrong_kept / n_kept if n_kept else None, "calibration_bins": bins,
            "decision_cost": (wrong_kept + (n - n_kept) * cost) / n,
            "ece_10bin": fsum(x["n"] * abs(x["event_rate"] - x["mean_probability"]) for x in bins) / n}


def check_manifest(manifest, frozen, config):
    splits = ["train", "calibration", "policy", "test"]
    expected_test = set()
    fit_by_seed = {seed["seed"]: seed for seed in frozen["seeds"]}
    length = config["window_samples"]
    for seed in config["seeds"]:
        parts = {r["partition"]: r for r in manifest if r["seed"] == seed}
        if len(parts) != len(splits):
            raise AssertionError("Missing provenance partition")
        fit = fit_by_seed[seed]
        if fit["scaler_source_fingerprint"] != parts["train"]["fingerprint"]:
            raise AssertionError("Scaler source is not training data")
        for model, calibrator in fit["calibration"].items():
            if model != "prevalence" and calibrator["source_fingerprint"] != parts["calibration"]["fingerprint"]:
                raise AssertionError("Calibration source is not the calibration partition")
        for policy in fit["policies"].values():
            if policy["source_fingerprint"] != parts["policy"]["fingerprint"]:
                raise AssertionError("Threshold source is not the policy partition")
        used, preceding_end = set(), -1 - config["time_gap_samples"]
        for split in splits:
            block = parts[split]
            devices = block["devices"]
            if len(devices) != config["devices"][split] or block["windows"] != len(devices) * config["windows_per_device"]:
                raise AssertionError("Unexpected device or window count")
            ids = {r["device"] for r in devices}
            if len(ids) != len(devices) or used & ids or min(r["start"] for r in devices) <= preceding_end + config["time_gap_samples"]:
                raise AssertionError("Device or time overlap")
            used.update(ids)
            preceding_end = max(r["end"] for r in devices)
            if split == "test":
                for item in devices:
                    device = f"s{seed}-d{item['device']:02d}"
                    for i in range(config["windows_per_device"]):
                        expected_test.add((seed, device, f"{device}-w{i:03d}", item["start"] + i * length))
    return expected_test, fit_by_seed


def verify(run):
    report = json.loads((run / "results.json").read_text(encoding="utf-8"))
    frozen = json.loads((run / "validation_frozen.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "split_manifest.json").read_text(encoding="utf-8"))
    config = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))
    expected_test, fits = check_manifest(manifest, frozen, config)
    if frozen["final_test_generated"] is not False or frozen["selected_model"] != report["model_selected_before_test"]:
        raise AssertionError("Missing pre-test model-selection checkpoint")
    scenarios = ("clean", "brief", "sustained")
    models = ("prevalence", "logistic", "cnn")
    for model in models:
        cost = fsum(v["policies"][model]["validation_cost"] for v in fits.values()) / len(fits)
        same(cost, frozen["mean_cost_by_model"][model], "Validation selection arithmetic")
    if min(models, key=frozen["mean_cost_by_model"].__getitem__) != frozen["selected_model"]:
        raise AssertionError("Selected model does not minimize validation decision cost")

    groups, labels_by_window = defaultdict(list), {}
    with (run / "test_predictions.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["seed"]), row["device"], row["window_id"], int(row["start_sample"]))
            label = int(row["label"])
            if label not in (0, 1) or labels_by_window.setdefault(key, label) != label:
                raise AssertionError("Source labels differ between model/scenario views")
            model, scenario = row["model"], row["scenario"]
            if model not in models or scenario not in scenarios:
                raise AssertionError("Unexpected model or scenario")
            same(float(row["confidence_threshold"]), fits[key[0]]["policies"][model]["threshold"], "Test threshold differs from pre-test selection")
            drop = config["dropout"]
            expected_missing = {"clean": 0, "brief": drop["brief_length"], "sustained": drop["sustained_end"] - drop["sustained_start"]}
            same(float(row["missing_fraction"]), expected_missing[scenario] / (2 * config["window_samples"]), "Outage fraction")
            groups[model, scenario].append((key, row))

    result_count = 0
    all_stats = {}
    for model in models:
        for scenario in scenarios:
            group = groups[model, scenario]
            if len(group) != len(expected_test) or {key for key, _ in group} != expected_test:
                raise AssertionError("One model/scenario does not cover the exact held-out universe")
            rows = [r for _, r in group]
            actual = recompute(rows, config["policy"]["withhold_cost"])
            expected = report["metrics"][model][scenario]
            all_stats[model, scenario] = actual
            for key, value in actual.items():
                if key == "calibration_bins":
                    for x, y in zip(value, expected[key], strict=True):
                        for field in x:
                            same(x[field], y[field], f"{model}/{scenario}/calibration/{field}")
                else:
                    same(value, expected[key], f"{model}/{scenario}/{key}")
            result_count += len(rows)

    with (run / "metrics.csv").open(encoding="utf-8", newline="") as file:
        metrics_rows = list(csv.DictReader(file))
    if len(metrics_rows) != 9 or {(r["model"], r["scenario"]) for r in metrics_rows} != set(all_stats):
        raise AssertionError("Metrics CSV does not have the exact nine model/scenario rows")
    for row in metrics_rows:
        expected = all_stats[row["model"], row["scenario"]]
        for name in row.keys() - {"model", "scenario"}:
            same(float(row[name]) if row[name] else None, expected[name], f"Metrics CSV {name}")

    comparisons = report["bootstrap"]["comparisons"]
    for name, fields in comparisons.items():
        a, b = name.split("_minus_")
        if a == "cnn" or a == "logistic":
            _, scenario = b.split("_", maxsplit=1)
            group_a, group_b = (a, scenario), ("logistic", scenario)
        else:
            model, scenario = a.split("_", maxsplit=1)
            group_a, group_b = (model, scenario), (model, b)
        for field, evidence in fields.items():
            same(all_stats[group_a][field] - all_stats[group_b][field], evidence["estimate"], "Paired point contrast")

    with (run / "null_predictions.csv").open(encoding="utf-8", newline="") as file:
        null = list(csv.DictReader(file))
    y = [int(r["label"]) for r in null]
    p = [float(r["probability"]) for r in null]
    same(average_precision(y, p), report["null_control"]["average_precision"], "Shuffle-control AP")
    if result_count != report["counts"]["prediction_rows"] or len(expected_test) != report["counts"]["test_windows"]:
        raise AssertionError("Output counts do not match verified prediction rows")
    return {"run": run.name, "verified_metric_rows": 9, "verified_predictions": result_count,
            "verified_test_windows": len(expected_test), "verified_null_rows": len(null), "status": "PASS"}


def compare(first, second):
    files = ("frozen_design.json", "environment.json", "split_manifest.json", "validation_frozen.json", "test_predictions.csv",
             "metrics.csv", "results.json", "null_predictions.csv", "risk_coverage.csv", "model_comparison.png", "policy_and_calibration.png")
    same_bytes = {name: hashlib.sha256((first / name).read_bytes()).hexdigest() for name in files}
    for name, digest in same_bytes.items():
        if hashlib.sha256((second / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Fresh process changed the deterministic output: {name}")
    return {"byte_identical": same_bytes, "status": "PASS"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    result = {"primary": verify(args.run)}
    if args.compare:
        result["rerun"] = verify(args.compare)
        result["reproducibility"] = compare(args.run, args.compare)
    print(json.dumps(result, indent=2))
