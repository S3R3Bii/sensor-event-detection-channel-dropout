"""Freeze or run the end-to-end CPU sensor robustness benchmark."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parent
WHEEL = ROOT / "vendor" / "model_evidence-0.7.0-py3-none-any.whl"
DESIGN_FILES = ("protocol.json", "data.py", "models.py", "evaluate.py", "run.py", "verify_results.py", "test_benchmark.py")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, allow_nan=False, sort_keys=True)
        file.write("\n")


def csv_write(path, rows):
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_design():
    config = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    if config["classification_threshold"] != 0.5:
        raise ValueError("This frozen benchmark uses the fixed 0.5 label threshold")
    if sha256(WHEEL) != config["model_evidence"]["wheel_sha256"]:
        raise ValueError("The pinned Model Evidence wheel differs from the approved archive")
    return config, {name: sha256(ROOT / name) for name in DESIGN_FILES}


def bind_runtime(config):
    threads = str(config["cnn"]["cpu_threads"])
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = threads
    sys.path.insert(0, str(WHEEL))
    if importlib.metadata.version("model-evidence") != config["model_evidence"]["version"]:
        raise ValueError("Unexpected Model Evidence version")


def environment():
    names = ("numpy", "scipy", "scikit-learn", "torch", "matplotlib", "pandas", "model-evidence")
    return {"python": platform.python_version(), "platform": platform.system(), "machine": platform.machine(),
            "execution_device": "cpu", "packages": {n: importlib.metadata.version(n) for n in names}}


def plot_results(out, groups, metrics, config, winner):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from data import SCENARIOS
    from evaluate import policy_stats
    from models import MODELS

    colors = {"prevalence": "#62676D", "logistic": "#986008", "cnn": "#126782"}
    labels = {"prevalence": "Training prevalence", "logistic": "Logistic regression", "cnn": "1D CNN"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), layout="constrained")
    for name in MODELS:
        for ax, key in zip(axes, ("average_precision", "brier"), strict=True):
            ax.plot(SCENARIOS, [metrics[name][s][key] for s in SCENARIOS], marker="o",
                    color=colors[name], label=labels[name], linewidth=2)
            ax.grid(axis="y", alpha=.25)
    axes[0].set(ylabel="Average precision (higher is better)", ylim=(0, 1))
    axes[1].set(ylabel="Brier score (lower is better)", ylim=(0, .3))
    axes[0].legend(loc="best", frameon=False)
    fig.suptitle("Held-out sensor event detection across dropout conditions", weight="bold")
    fig.supxlabel("Same 1,800 synthetic windows in each condition")
    fig.savefig(out / "model_comparison.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained")
    curve = []
    scenario_style = {"clean": ("#15786E", "o"), "brief": ("#3B5898", "s"), "sustained": ("#963E38", "^")}
    for scenario, (color, marker) in scenario_style.items():
        subset = groups[winner, scenario]
        y = np.asarray([r["label"] for r in subset])
        p = np.asarray([r["probability"] for r in subset])
        points = []
        for threshold in config["policy"]["confidence_grid"]:
            value = policy_stats(y, p, threshold, config["policy"]["withhold_cost"])
            points.append(value)
            curve.append({"scenario": scenario, "confidence_threshold": threshold, **value})
        valid = [v for v in points if v["risk"] is not None]
        axes[0].plot([v["coverage"] for v in valid], [v["risk"] for v in valid], color=color, alpha=.8, label=scenario)
        m = metrics[winner][scenario]
        axes[0].scatter(m["coverage"], m["risk"], color=color, marker=marker, s=70, zorder=4, edgecolor="white")
        bins = m["calibration_bins"]
        axes[1].plot([b["mean_probability"] for b in bins], [b["event_rate"] for b in bins],
                     color=color, marker=marker, markersize=4, label=scenario)
    axes[0].set(xlabel="Coverage (share predicted)", ylabel="Error rate among predictions", xlim=(0, 1.02), ylim=(0, .38))
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#6B6B6B", linewidth=1)
    axes[1].set(xlabel="Mean calibrated probability", ylabel="Observed event fraction", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.22)
    axes[1].legend(loc="upper left", frameon=False)
    fig.suptitle(f"{labels[winner]} on held-out devices", weight="bold")
    fig.supxlabel("Markers on the left show thresholds selected before test. Curves are descriptive.")
    fig.savefig(out / "policy_and_calibration.png", dpi=170)
    plt.close(fig)
    csv_write(out / "risk_coverage.csv", curve)


def run(out, config, freeze):
    import numpy as np
    from scipy.special import expit
    from sklearn.metrics import average_precision_score
    import torch
    from threadpoolctl import threadpool_limits
    from model_evidence import audit_prediction, audit_split, validate_report

    from data import PARTITIONS, SCENARIOS, check_partitions, corrupt, make_batch, plan_devices, shuffled_training
    from evaluate import choose_model, group_rows, metric_row, paired_bootstrap, paired_examples, select_policies
    from models import MODELS, TrainScale, calibrate, cnn_inputs, fit_models, train_cnn, training_views

    start = perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    json_write(out / "environment.json", environment())
    json_write(out / "frozen_design.json", freeze)
    states, frozen = [], []
    with threadpool_limits(limits=1):
        for seed in config["seeds"]:
            plan = list(plan_devices(config, seed))
            batches = {split: make_batch(config, seed, plan, split) for split in PARTITIONS[:-1]}
            train, calibration, policy = (batches[s] for s in PARTITIONS[:-1])
            fit = fit_models(train, config, seed)
            calibration_map = calibrate(calibration, {s: fit.scores(calibration, config, s) for s in SCENARIOS})
            policies = select_policies(policy, {s: fit.scores(policy, config, s) for s in SCENARIOS}, calibration_map, config)
            frozen.append({"seed": seed, "policies": policies, "training_prior": fit.prior,
                           "scaler_source_fingerprint": fit.scale.source_fingerprint,
                           "scale_means": fit.scale.mean.tolist(), "scale_standard_deviations": fit.scale.std.tolist(),
                           "calibration": {name: asdict(value) for name, value in calibration_map.items()},
                           "cnn_loss_by_epoch": fit.loss_history})
            states.append((seed, plan, batches, fit, calibration_map, policies))
            print(f"Seed {seed}: fitted models and selected thresholds on validation devices", flush=True)

        winner, validation_cost = choose_model(frozen)
        # Generate the final test batches only after this persistent selection checkpoint.
        json_write(out / "validation_frozen.json", {"selected_model": winner, "mean_cost_by_model": validation_cost,
                                                     "seeds": frozen, "final_test_generated": False})
        print(f"Validation chose {winner}. Final test evaluation starts now.", flush=True)
        predictions, split_records, null_control = [], [], None
        for seed, plan, batches, fit, calibration_map, policies in states:
            test = make_batch(config, seed, plan, "test")
            batches["test"] = test
            checks = check_partitions(batches, config, audit_split)
            for split, batch in batches.items():
                split_records.append({"seed": seed, "partition": split, "windows": len(batch.ids),
                                      "events": int(batch.labels.sum()), "fingerprint": batch.fingerprint(),
                                      "devices": [item for item in plan if item["split"] == split],
                                      "incoming_audit": next((x["evidence"] for x in checks if x["later"] == split), [])})
            for scenario in SCENARIOS:
                _, mask = corrupt(test, config, scenario)
                fraction = 1 - mask.mean(axis=(1, 2))
                channels = np.where(mask.all(axis=(1, 2)), -1, (~mask).sum(axis=2).argmax(axis=1))
                scores = fit.scores(test, config, scenario)
                for name in MODELS:
                    raw = expit(scores[name].aligned(test))
                    p = calibration_map[name].apply(scores[name], test)
                    cutoff = policies[name]["threshold"]
                    for i, identity in enumerate(test.ids):
                        predictions.append({"seed": seed, "device": test.devices[i], "window_id": identity,
                                            "start_sample": int(test.starts[i]), "scenario": scenario, "model": name,
                                            "label": int(test.labels[i]), "raw_probability": float(raw[i]),
                                            "probability": float(p[i]), "confidence_threshold": cutoff,
                                            "retained": bool(max(p[i], 1 - p[i]) >= cutoff),
                                            "missing_fraction": float(fraction[i]), "channel_missing": int(channels[i])})
            if seed == config["seeds"][0]:
                null = shuffled_training(batches["train"], seed)
                scale = TrainScale.fit(null)
                inputs, y = training_views(null, config, scale)
                net, _ = train_cnn(torch.cat([cnn_inputs(*view) for view in inputs]), y, config["cnn"], seed)
                values, mask = corrupt(test, config, "clean")
                with torch.inference_mode():
                    p = expit(net(cnn_inputs(scale.transform(values, mask), mask)).numpy())
                null_control = {"seed": seed, "windows": len(p), "scenario": "clean", "event_prevalence": float(test.labels.mean()),
                                "average_precision": float(average_precision_score(test.labels, p)),
                                "interpretation": "Single fixed shuffle of training labels before view replication. Raw probabilities; no calibration or selection."}
                csv_write(out / "null_predictions.csv", [{"window_id": identity, "label": int(y), "probability": float(z)}
                                                          for identity, y, z in zip(test.ids, test.labels, p, strict=True)])
            print(f"Seed {seed}: scored {len(test.ids)} untouched source windows in three conditions", flush=True)

        csv_write(out / "test_predictions.csv", predictions)
        json_write(out / "split_manifest.json", split_records)
        groups = group_rows(predictions)
        cost = config["policy"]["withhold_cost"]
        metrics = {name: {scenario: metric_row(groups[name, scenario], cost) for scenario in SCENARIOS} for name in MODELS}
        audit_dir = out / "model_evidence"
        audit_dir.mkdir()
        for (name, scenario), rows in groups.items():
            y = np.array([r["label"] for r in rows])
            p = np.array([r["probability"] for r in rows])
            baseline = np.array([r["probability"] >= 0.5 for r in groups["prevalence", scenario]], dtype=int)
            audit = audit_prediction(y, (p >= 0.5).astype(int), task="classification", probabilities=p,
                                     baseline_predictions=baseline, split_role="final_holdout")
            validate_report(audit)
            for metric in ("average_precision", "brier"):
                if not np.isclose(audit["estimates"][metric], metrics[name][scenario][metric], atol=1e-12):
                    raise ValueError(f"Pinned Model Evidence disagrees on {metric}")
            json_write(audit_dir / f"{name}_{scenario}.json", audit)
        intervals = paired_bootstrap(groups, config)
        plot_results(out, groups, metrics, config, winner)
        counts = Counter(r["partition"] for r in split_records)
        json_write(out / "results.json", {"model_selected_before_test": winner, "mean_validation_cost": validation_cost,
                                          "metrics": metrics, "bootstrap": intervals, "null_control": null_control,
                                          "failure_analysis": paired_examples(groups, winner),
                                          "counts": {"seeds": len(states), "unique_source_windows": sum(r["windows"] for r in split_records),
                                                     "partition_records": dict(counts), "prediction_rows": len(predictions),
                                                     "test_windows": len(groups[winner, "clean"]), "test_devices": intervals["devices"],
                                                     "cnn_parameters": sum(v.numel() for v in states[0][3].cnn.parameters())}})
        csv_write(out / "metrics.csv", [{"model": n, "scenario": s, **{k: v for k, v in metrics[n][s].items() if k != "calibration_bins"}}
                                        for n in MODELS for s in SCENARIOS])
    json_write(out / "runtime.json", {"execution_seconds": perf_counter() - start,
                                       "completed_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Verified Model Evidence calculations for 9 groups. Results: {out.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="Create the design checkpoint without training or evaluation")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "reproduction")
    args = parser.parse_args()
    config, hashes = load_design()
    freeze_file = ROOT / "frozen_protocol.json"
    if args.freeze:
        json_write(freeze_file, {"frozen_utc": datetime.now(timezone.utc).isoformat(), "sha256": hashes,
                                 "state": "Design fixed before any benchmark fitting or test evaluation"})
        print("Frozen protocol and source hashes; no model fitted.")
        return
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    if hashes != freeze["sha256"]:
        raise ValueError("Source differs from the preregistered design")
    bind_runtime(config)
    run(args.out.resolve(), config, freeze)


if __name__ == "__main__":
    main()
