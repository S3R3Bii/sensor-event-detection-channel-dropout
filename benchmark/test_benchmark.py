"""Regression and leakage tests for the frozen applied ML example."""

from dataclasses import replace
import unittest

from run import bind_runtime, load_design

CONFIG, _ = load_design()
bind_runtime(CONFIG)

import numpy as np
from model_evidence import audit_split
from sklearn.metrics import average_precision_score, brier_score_loss

from data import PARTITIONS, SCENARIOS, check_partitions, corrupt, make_batch, plan_devices, shuffled_training
from evaluate import choose_model, policy_stats, select_policies
from models import Calibration, MODELS, Scores, TrainScale, calibrate, fit_models
from verify_results import average_precision, recompute


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {**CONFIG, "windows_per_device": 8, "devices": {s: 3 for s in PARTITIONS}}
        cls.plan = list(plan_devices(cls.config, 101))
        cls.batches = {s: make_batch(cls.config, 101, cls.plan, s) for s in PARTITIONS}

    def test_generation_and_window_identity_reproduce(self):
        plan = list(plan_devices(self.config, 101))
        self.assertEqual(self.plan, plan)
        for split in PARTITIONS:
            batch = self.batches[split]
            again = make_batch(self.config, 101, plan, split)
            self.assertEqual(batch.fingerprint(), again.fingerprint())
            self.assertEqual(batch.signal.shape, (24, 2, 64))
            self.assertTrue(np.all(np.diff(batch.starts) >= 64))
        alternate = list(plan_devices(self.config, 203))
        self.assertNotEqual(self.batches["train"].fingerprint(), make_batch(self.config, 203, alternate, "train").fingerprint())

    def test_group_and_time_split_guards(self):
        checks = check_partitions(self.batches, self.config, audit_split)
        self.assertEqual(len(checks), 3)
        train, test = (self.batches[s] for s in ("train", "test"))
        for tampered in (replace(test, devices=(train.devices[0], *test.devices[1:])),
                         replace(test, starts=test.starts - test.starts[0])):
            with self.assertRaises(ValueError):
                check_partitions({**self.batches, "test": tampered}, self.config, audit_split)

    def test_duplicate_source_windows_rejected_even_with_new_ids(self):
        train, test = (self.batches[s] for s in ("train", "test"))
        duplicated = test.signal.copy()
        duplicated[0] = train.signal[0]
        for tampered in (replace(test, ids=(train.ids[0], *test.ids[1:])), replace(test, signal=duplicated)):
            with self.assertRaises(ValueError):
                check_partitions({**self.batches, "test": tampered}, self.config, audit_split)

    def test_dropout_is_label_independent_and_paired(self):
        batch = self.batches["test"]
        _, clean = corrupt(batch, self.config, "clean")
        values, brief = corrupt(batch, self.config, "brief")
        _, sustained = corrupt(batch, self.config, "sustained")
        self.assertTrue(clean.all())
        self.assertTrue((~brief).sum(axis=(1, 2)).min() == 12)
        self.assertTrue((~sustained).sum(axis=(1, 2)).max() == 52)
        self.assertTrue(np.all(sustained <= brief))
        self.assertTrue(np.all(values[~brief] == 0))
        _, reversed_labels = corrupt(replace(batch, labels=1 - batch.labels), self.config, "brief")
        np.testing.assert_array_equal(brief, reversed_labels)

    def test_scaler_is_fitted_on_training_only_and_stays_fixed(self):
        train, test = (self.batches[s] for s in ("train", "test"))
        with self.assertRaises(ValueError):
            fit_models(test, self.config, 101)
        scale = TrainScale.fit(train)
        mean, std = scale.mean.copy(), scale.std.copy()
        scale.transform(test.signal + 8000, np.ones_like(test.signal, dtype=bool))
        np.testing.assert_array_equal(mean, scale.mean)
        np.testing.assert_array_equal(std, scale.std)
        self.assertEqual(scale.source_fingerprint, train.fingerprint())

    def test_calibration_and_policy_do_not_accept_test_rows(self):
        for operation in (lambda b: calibrate(b, {}), lambda b: select_policies(b, {}, {}, self.config)):
            with self.assertRaises(ValueError):
                operation(self.batches["test"])
        batch = self.batches["calibration"]
        z = np.linspace(-2, 2, len(batch.ids))
        scores = {s: {name: Scores(batch.ids, z) for name in MODELS} for s in SCENARIOS}
        trained = calibrate(batch, scores)
        self.assertEqual(trained["cnn"].source_fingerprint, batch.fingerprint())
        policy_batch = self.batches["policy"]
        scores = {s: {name: Scores(policy_batch.ids, z) for name in MODELS} for s in SCENARIOS}
        rules = select_policies(policy_batch, scores, trained, self.config)
        for rule in rules.values():
            self.assertEqual(rule["source_fingerprint"], policy_batch.fingerprint())
            self.assertGreaterEqual(min(x["coverage"] for x in rule["scenarios"].values()), .7)

    def test_probability_alignment_rejects_reordered_ids_and_wrong_length(self):
        batch = self.batches["test"]
        cal = Calibration(1., 0., "fixture")
        for prediction in (Scores(batch.ids[::-1], np.zeros(len(batch.ids))), Scores(batch.ids, np.zeros(2)),
                           Scores(batch.ids, np.full(len(batch.ids), np.nan))):
            with self.assertRaises(ValueError):
                cal.apply(prediction, batch)

    def test_withholding_has_known_cost_and_coverage(self):
        labels = np.array([1, 0, 1, 0])
        p = np.array([.95, .10, .48, .67])
        result = policy_stats(labels, p, .8, .18)
        self.assertEqual((result["retained"], result["errors_retained"], result["coverage"], result["risk"]), (2, 0, .5, 0.))
        self.assertAlmostEqual(result["decision_cost"], .09)
        self.assertEqual(choose_model([{"policies": {m: {"validation_cost": .12} for m in MODELS}}])[0], "prevalence")

    def test_independent_metric_arithmetic_including_ties(self):
        labels = [1, 0, 1, 0]
        p = [.8, .8, .5, .2]
        self.assertAlmostEqual(average_precision(labels, p), .25 + 1/3)
        self.assertAlmostEqual(average_precision(labels, p), average_precision_score(labels, p))
        rows = [{"label": str(y), "probability": str(z), "raw_probability": str(z), "confidence_threshold": ".7",
                 "retained": str(max(z, 1-z) >= .7)} for y, z in zip(labels, p)]
        result = recompute(rows, .18)
        self.assertAlmostEqual(result["brier"], brier_score_loss(labels, p))
        self.assertAlmostEqual(result["coverage"], .75)

    def test_null_shuffle_preserves_training_label_count(self):
        train = self.batches["train"]
        null = shuffled_training(train, 101)
        self.assertEqual(null.ids, train.ids)
        self.assertEqual(int(null.labels.sum()), int(train.labels.sum()))
        self.assertFalse(np.array_equal(null.labels, train.labels))
        with self.assertRaises(ValueError):
            shuffled_training(self.batches["test"], 101)


if __name__ == "__main__":
    unittest.main(verbosity=2)
