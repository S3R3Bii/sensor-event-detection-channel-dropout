"""Deterministic synthetic streams, windows and paired sensor corruptions."""

from dataclasses import dataclass, replace
import hashlib

import numpy as np


PARTITIONS = ("train", "calibration", "policy", "test")
SCENARIOS = ("clean", "brief", "sustained")


@dataclass(frozen=True)
class Batch:
    split: str
    ids: tuple[str, ...]
    devices: tuple[str, ...]
    starts: np.ndarray
    signal: np.ndarray
    labels: np.ndarray

    def fingerprint(self):
        h = hashlib.sha256()
        h.update("\n".join(self.ids).encode())
        for array in (self.starts, self.signal, self.labels):
            h.update(array.tobytes())
        return h.hexdigest()


def generator(seed, device, purpose):
    return np.random.default_rng(np.random.SeedSequence([seed, device, purpose]))


def plan_devices(config, seed):
    """Allocate independent devices and nonoverlapping chronological blocks first."""
    counts = config["devices"]
    n = config["windows_per_device"] * config["window_samples"]
    separation = config["time_gap_samples"] + 1
    device_numbers = generator(seed, 0, 1).permutation(sum(counts.values()))
    offset = 0
    for split in PARTITIONS:
        for number in device_numbers[offset:offset + counts[split]]:
            yield {"split": split, "device": int(number), "start": offset * (n + separation), "end": offset * (n + separation) + n - 1}
            offset += 1


def make_stream(config, seed, item):
    """Simulate two synchronized channels with common and channel-specific nuisances."""
    rng = generator(seed, item["device"], 2)
    length = config["window_samples"]
    n = config["windows_per_device"]
    labels = (rng.random(n) < config["event_probability"]).astype(np.int8)
    t = np.arange(n * length, dtype=float)
    nuisance = rng.uniform(0.09, 0.26, size=(2, 1)) * np.sin(
        t[None, :] * rng.uniform(0.08, 0.17, size=(2, 1)) + rng.uniform(0, 6.28, size=(2, 1)))
    # Mild regime drift is attached to acquisition block, not to the target label.
    offset = rng.normal(0.0, 0.24, size=(2, 1)) + 0.035 * PARTITIONS.index(item["split"])
    gain = rng.uniform(0.68, 1.27, size=(2, 1))
    sensor = offset + nuisance + rng.normal(0.0, rng.uniform(0.23, 0.40), size=(2, t.size))
    x = np.arange(length, dtype=float)
    for j, label in enumerate(labels):
        center = rng.integers(21, 42)
        width = rng.uniform(1.6, 3.1)
        delay = rng.integers(1, 5)
        amplitude = rng.uniform(0.45, 1.26)
        polarity = rng.choice([-1, 1])
        style = rng.integers(0, 3)
        for channel in range(2):
            c = center + channel * delay
            pulse = np.exp(-0.5 * ((x - c) / width) ** 2)
            echo = np.exp(-0.5 * ((x - c - 5) / (width * 1.12)) ** 2)
            if label:
                shape = pulse - 0.81 * echo
            elif style == 0:
                shape = pulse + 0.35 * echo
            elif style == 1:
                # Short isolated peaks are plausible non-event distractors.
                shape = 0.73 * pulse
            else:
                # In one channel an event-like shape occurs without cross-channel agreement.
                shape = pulse - 0.81 * echo if channel == j % 2 else 0.58 * echo
            sensor[channel, j * length:(j + 1) * length] += polarity * gain[channel, 0] * amplitude * shape
    return sensor.astype(np.float32), labels


def make_batch(config, seed, plan, split):
    """Window only devices already assigned to this one partition."""
    length = config["window_samples"]
    chunks, labels, ids, devices, starts = [], [], [], [], []
    for item in plan:
        if item["split"] != split:
            continue
        stream, target = make_stream(config, seed, item)
        device = f"s{seed}-d{item['device']:02d}"
        chunks.append(stream.reshape(2, -1, length).transpose(1, 0, 2))
        labels.append(target)
        for w in range(len(target)):
            ids.append(f"{device}-w{w:03d}")
            devices.append(device)
            starts.append(item["start"] + w * length)
    return Batch(split, tuple(ids), tuple(devices), np.array(starts, dtype=np.int64),
                 np.concatenate(chunks), np.concatenate(labels))


def corrupt(batch, config, scenario):
    """Use one random channel and nested masks per window across both outage lengths."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario}")
    mask = np.ones_like(batch.signal, dtype=bool)
    if scenario != "clean":
        p = config["dropout"]
        for i, identity in enumerate(batch.ids):
            # Window identity, not labels, drives corruption. Stable across process runs.
            key = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
            rng = np.random.default_rng(key)
            channel = rng.integers(2)
            if scenario == "brief":
                start = rng.integers(p["brief_start_min"], p["brief_start_max"] + 1)
                end = start + p["brief_length"]
            else:
                start, end = p["sustained_start"], p["sustained_end"]
            mask[i, channel, start:end] = False
    return np.where(mask, batch.signal, 0.0), mask


def check_partitions(batches, config, audit_split):
    """Reject identity/content reuse and audit ordered separation with Model Evidence."""
    seen_ids, seen_devices, seen_windows = set(), set(), set()
    prior = None
    audits = []
    for split in PARTITIONS:
        batch = batches[split]
        current_ids, current_devices = set(batch.ids), set(batch.devices)
        if len(current_ids) != len(batch.ids) or seen_ids & current_ids or seen_devices & current_devices:
            raise ValueError("Source window or device overlap across partitions")
        signatures = {hashlib.sha256(w.tobytes()).digest() for w in batch.signal}
        if len(signatures) != len(batch.ids) or seen_windows & signatures:
            raise ValueError("Repeated window content across partitions")
        seen_ids.update(current_ids)
        seen_devices.update(current_devices)
        seen_windows.update(signatures)
        if prior is not None:
            groups = (*prior.devices, *batch.devices)
            end_starts = prior.starts + config["window_samples"] - 1
            times = np.concatenate((end_starts, batch.starts))
            pivot = len(prior.ids)
            checks = audit_split(np.arange(pivot), np.arange(pivot, len(groups)), groups=groups, times=times,
                                 strategy="time", gap=config["time_gap_samples"])
            if any(row["status"] != "no_issue_detected" for row in checks):
                raise ValueError("Model Evidence detected group or time leakage")
            audits.append({"earlier": prior.split, "later": split, "evidence": checks})
        prior = batch
    return audits


def shuffled_training(batch, seed):
    if batch.split != "train":
        raise ValueError("Shuffle control must be confined to training")
    rng = generator(seed, 0, 53)
    return replace(batch, labels=rng.permutation(batch.labels))
