"""CPU models and fit-stage guards for the sensor benchmark."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from data import SCENARIOS, corrupt


MODELS = ("prevalence", "logistic", "cnn")


def require_split(batch, split):
    if batch.split != split:
        raise ValueError(f"Expected {split} partition, received {batch.split}")


@dataclass(frozen=True)
class Scores:
    ids: tuple[str, ...]
    logits: np.ndarray

    def aligned(self, batch):
        values = np.asarray(self.logits, dtype=np.float64)
        if self.ids != batch.ids or values.shape != batch.labels.shape or not np.isfinite(values).all():
            raise ValueError("Prediction values and labels are not aligned by source window")
        return values


@dataclass(frozen=True)
class TrainScale:
    mean: np.ndarray
    std: np.ndarray
    source_fingerprint: str

    @classmethod
    def fit(cls, batch):
        require_split(batch, "train")
        return cls(batch.signal.mean(axis=(0, 2), dtype=np.float64),
                   batch.signal.std(axis=(0, 2), dtype=np.float64).clip(1e-8), batch.fingerprint())

    def transform(self, values, mask):
        z = (values - self.mean[None, :, None]) / self.std[None, :, None]
        return np.where(mask, z, 0.0).astype(np.float32)


def window_features(values, mask):
    """Fixed signal summaries with explicit observed fractions and short lag features."""
    count = mask.sum(axis=2).clip(1)
    mean = values.sum(axis=2) / count
    rms = np.sqrt(np.square(values).sum(axis=2) / count)
    dev = np.where(mask, values - mean[:, :, None], 0.0)
    std = np.sqrt(np.square(dev).sum(axis=2) / count)
    peak = np.abs(values).max(axis=2)
    adjacent = mask[:, :, 1:] & mask[:, :, :-1]
    roughness = (np.abs(np.diff(values, axis=2)) * adjacent).sum(axis=2) / adjacent.sum(axis=2).clip(1)
    columns = [mask.mean(axis=2), mean, rms, std, peak, roughness]
    for lag in (2, 5):
        paired = mask[:, :, :-lag] & mask[:, :, lag:]
        columns.append((values[:, :, :-lag] * values[:, :, lag:] * paired).sum(axis=2) /
                       paired.sum(axis=2).clip(1))
    for segment in np.array_split(np.arange(values.shape[2]), 4):
        observed = mask[:, :, segment]
        energy = np.square(values[:, :, segment]).sum(axis=2) / observed.sum(axis=2).clip(1)
        columns.extend((observed.mean(axis=2), energy))
    both = mask[:, 0] & mask[:, 1]
    synchronous = (values[:, 0] * values[:, 1] * both).sum(axis=1) / both.sum(axis=1).clip(1)
    columns.extend((synchronous[:, None], both.mean(axis=1)[:, None]))
    return np.concatenate(columns, axis=1)


class SequenceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(4, 12, 7, padding=3), nn.ReLU(),
                                  nn.Conv1d(12, 16, 5, padding=2), nn.ReLU())
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        h = self.conv(x)
        return self.output(torch.cat((h.mean(dim=2), h.amax(dim=2)), dim=1)).squeeze(1)


def cnn_inputs(values, mask):
    return torch.from_numpy(np.concatenate((values, mask.astype(np.float32)), axis=1))


def train_cnn(inputs, labels, config, seed):
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = SequenceCNN().cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    target = torch.as_tensor(labels, dtype=torch.float32)
    batch_rng = torch.Generator(device="cpu").manual_seed(seed + 3000)
    history = []
    for _ in range(config["epochs"]):
        total = 0.0
        order = torch.randperm(len(target), generator=batch_rng)
        for index in order.split(config["batch_size"]):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs[index]), target[index])
            loss.backward()
            opt.step()
            total += loss.item() * len(index)
        history.append(total / len(target))
    return model.eval(), history


@dataclass
class Fitted:
    scale: TrainScale
    prior: float
    logistic: object
    cnn: SequenceCNN
    loss_history: list[float]

    def scores(self, batch, config, scenario):
        values, mask = corrupt(batch, config, scenario)
        scaled = self.scale.transform(values, mask)
        with torch.inference_mode():
            nn_scores = self.cnn(cnn_inputs(scaled, mask)).numpy()
        return {
            "prevalence": Scores(batch.ids, np.full(len(batch.ids), logit(self.prior))),
            "logistic": Scores(batch.ids, self.logistic.decision_function(window_features(scaled, mask))),
            "cnn": Scores(batch.ids, nn_scores),
        }


def training_views(batch, config, scale):
    require_split(batch, "train")
    views = []
    for scenario in config["training_scenarios"]:
        values, mask = corrupt(batch, config, scenario)
        views.append((scale.transform(values, mask), mask))
    return views, np.tile(batch.labels, len(views))


def fit_models(batch, config, seed):
    scale = TrainScale.fit(batch)
    views, labels = training_views(batch, config, scale)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
    logistic.fit(np.concatenate([window_features(*view) for view in views]), labels)
    inputs = torch.cat([cnn_inputs(*view) for view in views])
    cnn, history = train_cnn(inputs, labels, config["cnn"], seed)
    return Fitted(scale, float(batch.labels.mean()), logistic, cnn, history)


@dataclass(frozen=True)
class Calibration:
    slope: float
    intercept: float
    source_fingerprint: str

    def apply(self, scores, batch):
        return expit(self.slope * scores.aligned(batch) + self.intercept)


def calibrate(batch, scenario_scores):
    """Fit one sigmoid per learned model on separate calibration devices only."""
    require_split(batch, "calibration")
    output = {"prevalence": Calibration(1.0, 0.0, "training prevalence unchanged")}
    y = np.tile(batch.labels, len(SCENARIOS))
    for name in ("logistic", "cnn"):
        x = np.concatenate([scenario_scores[s][name].aligned(batch) for s in SCENARIOS])
        fit = LogisticRegression(C=10.0, max_iter=500).fit(x[:, None], y)
        output[name] = Calibration(float(fit.coef_[0, 0]), float(fit.intercept_[0]), batch.fingerprint())
    return output
