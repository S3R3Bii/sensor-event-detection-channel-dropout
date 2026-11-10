"""Validation policy selection and paired held-out evaluation."""

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

from data import SCENARIOS
from models import MODELS, require_split


def policy_stats(y, p, confidence_threshold, withhold_cost):
    confidence = np.maximum(p, 1 - p)
    retained = confidence >= confidence_threshold
    errors = (p >= 0.5) != y
    count = int(retained.sum())
    wrong = int(errors[retained].sum())
    return {"windows": len(y), "retained": count, "errors_retained": wrong,
            "coverage": count / len(y), "risk": wrong / count if count else None,
            "decision_cost": (wrong + withhold_cost * (len(y) - count)) / len(y)}


def select_policies(batch, scenario_scores, calibrations, config):
    require_split(batch, "policy")
    rule = config["policy"]
    selected = {}
    for name in MODELS:
        probs = {s: calibrations[name].apply(scenario_scores[s][name], batch) for s in SCENARIOS}
        trials = []
        for threshold in rule["confidence_grid"]:
            stats = {s: policy_stats(batch.labels, p, threshold, rule["withhold_cost"]) for s, p in probs.items()}
            if min(v["coverage"] for v in stats.values()) + 1e-12 >= rule["minimum_scenario_coverage"]:
                cost = np.mean([v["decision_cost"] for v in stats.values()])
                coverage = np.mean([v["coverage"] for v in stats.values()])
                trials.append((float(cost), -float(coverage), threshold, stats))
        cost, _, threshold, stats = min(trials, key=lambda row: row[:3])
        selected[name] = {"threshold": threshold, "validation_cost": cost, "scenarios": stats,
                          "source_fingerprint": batch.fingerprint()}
    return selected


def choose_model(frozen):
    # A tie keeps the simpler model. All rows here come from the policy split.
    cost = {name: float(np.mean([seed["policies"][name]["validation_cost"] for seed in frozen])) for name in MODELS}
    return min(MODELS, key=cost.__getitem__), cost


def calibration_bins(y, p):
    index = np.minimum((10 * p).astype(int), 9)
    rows = []
    for b in range(10):
        keep = index == b
        if keep.any():
            rows.append({"bin": b, "n": int(keep.sum()), "mean_probability": float(p[keep].mean()),
                         "event_rate": float(y[keep].mean())})
    return rows


def metric_row(rows, cost):
    y = np.asarray([r["label"] for r in rows])
    p = np.asarray([r["probability"] for r in rows])
    raw = np.asarray([r["raw_probability"] for r in rows])
    kept = np.asarray([r["retained"] for r in rows], dtype=bool)
    errors = (p >= 0.5) != y
    n = len(y)
    bins = calibration_bins(y, p)
    return {"windows": n, "events": int(y.sum()), "event_prevalence": float(y.mean()),
            "average_precision": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "raw_brier": float(brier_score_loss(y, raw)), "accuracy": float((~errors).mean()),
            "ece_10bin": sum(b["n"] / n * abs(b["mean_probability"] - b["event_rate"]) for b in bins),
            "calibration_bins": bins, "coverage": float(kept.mean()),
            "risk": float(errors[kept].mean()) if kept.any() else None,
            "decision_cost": float((errors[kept].sum() + cost * (~kept).sum()) / n)}


def group_rows(predictions):
    groups = defaultdict(list)
    for row in predictions:
        groups[row["model"], row["scenario"]].append(row)
    return groups


def paired_bootstrap(groups, config):
    """Cluster bootstrap over devices within each experiment seed, paired across views."""
    reference = groups["logistic", "clean"]
    y = np.asarray([r["label"] for r in reference])
    ids = [r["window_id"] for r in reference]
    device_indices = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(reference):
        device_indices[row["seed"]][row["device"]].append(i)
    probabilities, decision_costs = {}, {}
    for key, rows in groups.items():
        if [r["window_id"] for r in rows] != ids or [r["label"] for r in rows] != y.tolist():
            raise ValueError("Pairing mismatch in held-out rows")
        p = np.asarray([r["probability"] for r in rows])
        kept = np.asarray([r["retained"] for r in rows])
        probabilities[key] = p
        decision_costs[key] = np.where(kept, (p >= .5) != y, config["policy"]["withhold_cost"])

    comparisons = {
        "cnn_minus_logistic_clean": (("cnn", "clean"), ("logistic", "clean")),
        "cnn_minus_logistic_brief": (("cnn", "brief"), ("logistic", "brief")),
        "cnn_minus_logistic_sustained": (("cnn", "sustained"), ("logistic", "sustained")),
        "cnn_sustained_minus_clean": (("cnn", "sustained"), ("cnn", "clean")),
        "logistic_sustained_minus_clean": (("logistic", "sustained"), ("logistic", "clean")),
    }

    def contrasts(index):
        yy = y[index]
        ap = {k: average_precision_score(yy, v[index]) for k, v in probabilities.items() if k[0] != "prevalence"}
        brier = {k: np.square(v[index] - yy).mean() for k, v in probabilities.items()}
        cost = {k: v[index].mean() for k, v in decision_costs.items()}
        return {name: np.array([table[a] - table[b] for table in (ap, brier, cost)])
                for name, (a, b) in comparisons.items()}

    estimate = contrasts(np.arange(len(y)))
    samples = defaultdict(list)
    rng = np.random.default_rng(config["bootstrap_seed"])
    strata = [list(dev.values()) for dev in device_indices.values()]
    valid = 0
    for _ in range(config["bootstrap_draws"]):
        sampled = [d[int(i)] for d in strata for i in rng.integers(0, len(d), size=len(d))]
        index = np.concatenate(sampled)
        if len(np.unique(y[index])) != 2:
            continue
        valid += 1
        for name, row in contrasts(index).items():
            samples[name].append(row)
    if valid < 0.9 * config["bootstrap_draws"]:
        raise ValueError("Too few class-complete device bootstrap samples")
    output = {}
    for name, point in estimate.items():
        low, high = np.percentile(samples[name], (2.5, 97.5), axis=0)
        output[name] = {k: {"estimate": float(point[i]), "percentile_95": [float(low[i]), float(high[i])]}
                        for i, k in enumerate(("average_precision", "brier", "decision_cost"))}
    return {"draws": config["bootstrap_draws"], "valid_draws": valid, "devices": sum(map(len, strata)),
            "scope": "Resamples fixed test devices within seeds, conditional on fitted models and synthetic generator.",
            "comparisons": output}


def paired_examples(groups, model):
    clean, sustained = groups[model, "clean"], groups[model, "sustained"]
    candidates = []
    for a, b in zip(clean, sustained, strict=True):
        a_ok = (a["probability"] >= 0.5) == a["label"]
        b_ok = (b["probability"] >= 0.5) == b["label"]
        if a_ok and not b_ok:
            candidates.append({"window_id": a["window_id"], "device": a["device"], "label": a["label"],
                               "clean_probability": a["probability"], "sustained_probability": b["probability"],
                               "sustained_retained": b["retained"], "channel_missing": b["channel_missing"],
                               "sustained_missing_fraction": b["missing_fraction"]})
    # One true example of each class x policy outcome, deterministic window order.
    examples = {}
    for c in candidates:
        examples.setdefault((c["label"], c["sustained_retained"]), c)
    return {"clean_correct_sustained_wrong": len(candidates), "examples": list(examples.values())}
