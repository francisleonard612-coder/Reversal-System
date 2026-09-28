"""
Level 2: calibrated P(win) model -- training, promotion, and live inference
(spec Sections 21-26, 43).

WHAT THIS ADDS TO LEVEL 1. Level 1 can say "this is a confirmed reversal
setup" but not "this wins 56% of the time", so it cannot compute expected
value and has to fall back on a payout floor (strategy/edge_engine.py).
This module supplies the missing probability -- but only when the evidence
says it's real:

1. Walk-forward evaluation first (models/baseline.py): the model is fitted on
   everything before each block and scored ON that block, never on data it
   saw.
2. PROMOTION IS EARNED, NOT ASSUMED. A trained model is only marked
   PROMOTED when, on every held-out block, (a) its predictions correlate
   POSITIVELY with real outcomes above the floor, and (b) it beats the
   trivial "always predict the base rate" forecaster on Brier score. A
   model that fails is still saved (for inspection) as REJECTED, and the
   live bot refuses to trade on it.
3. Calibration: final model = logistic regression on the older 80% of data
   + Platt scaling fitted on the newest 20%. The same newest slice gives
   per-probability-bucket sample sizes, used for a conservative LOWER bound
   on P(win); the live EV gate can require that lower bound, not the point
   estimate, to clear break-even.
4. Fingerprinted to the config it was trained under. Change a feature
   parameter (RSI period, candle size, horizon...) and the saved model no
   longer describes the features the bot computes -- the live bot refuses
   to load it rather than silently mis-serving it.

The artifact is plain JSON (weights, scaler, calibration) -- no pickle, so
loading a model file can never execute code.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from models.baseline import train_and_evaluate_walk_forward
from models.dataset import MODEL_FEATURE_NAMES, build_training_examples

MODEL_ID = "level2_logreg_platt"
PROMOTED = "PROMOTED"
REJECTED = "REJECTED"

#: config sections whose values change what a feature MEANS. A model trained
#: under one set is invalid under another.
_FINGERPRINT_SECTIONS = ("candles", "stretch", "volatility", "momentum",
                         "price_action", "structure", "regime", "reversal")


def config_fingerprint(cfg: dict) -> str:
    relevant = {k: cfg.get(k) for k in _FINGERPRINT_SECTIONS}
    relevant["duration_bars_approx"] = cfg["contract"]["duration_bars_approx"]
    blob = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------- data

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    epochs: np.ndarray
    level1_score: np.ndarray
    symbols: list[str]
    n_by_symbol: dict


def build_dataset(candles_by_symbol: dict, cfg: dict) -> Dataset:
    """Replays every symbol's stored candles through the live feature code
    and stacks the labeled examples, ordered by time."""
    rows, ys, eps, l1 = [], [], [], []
    n_by = {}
    for sym, candles in candles_by_symbol.items():
        ex = [e for e in build_training_examples(candles, cfg) if e.label is not None]
        n_by[sym] = len(ex)
        for e in ex:
            rows.append([e.features[n] for n in MODEL_FEATURE_NAMES])
            ys.append(1.0 if e.label else 0.0)
            eps.append(e.epoch)
            l1.append(e.blended_score)
    order = np.argsort(np.asarray(eps, dtype=float), kind="stable")
    as_arr = lambda v: np.asarray(v, dtype=float)[order] if v else np.zeros((0,))
    X = np.asarray(rows, dtype=float)[order] if rows else np.zeros((0, len(MODEL_FEATURE_NAMES)))
    return Dataset(X=X, y=as_arr(ys), epochs=as_arr(eps), level1_score=as_arr(l1),
                   symbols=sorted(candles_by_symbol), n_by_symbol=n_by)


# --------------------------------------------------------------- evaluation

@dataclass
class BlockComparison:
    n_test: int
    base_rate: float
    model_corr: float
    level1_corr: float
    model_brier: float
    base_rate_brier: float


@dataclass
class EvaluationReport:
    blocks: list[BlockComparison] = field(default_factory=list)
    walk_forward_text: str = ""
    corr_floor: float = 0.05

    def promotion_failures(self) -> list[str]:
        """Empty list == promotable."""
        fails = []
        if len(self.blocks) < 2:
            return [f"only {len(self.blocks)} evaluable held-out block(s) -- need at least 2"]
        for i, b in enumerate(self.blocks, 1):
            if not (b.model_corr == b.model_corr) or b.model_corr < self.corr_floor:
                fails.append(f"block {i}: held-out correlation {b.model_corr:+.4f} "
                             f"below +{self.corr_floor:.2f}")
            if not b.model_brier < b.base_rate_brier:
                fails.append(f"block {i}: Brier {b.model_brier:.4f} does not beat "
                             f"base-rate forecast {b.base_rate_brier:.4f}")
        return fails

    def report(self) -> str:
        L = [self.walk_forward_text, "",
             "LEVEL 2 vs LEVEL 1 on identical held-out blocks",
             f"{'block':>5} {'n':>7} {'base':>6} {'L1 corr':>9} {'model corr':>11} "
             f"{'model brier':>12} {'base brier':>11}"]
        for i, b in enumerate(self.blocks, 1):
            L.append(f"{i:>5} {b.n_test:>7} {b.base_rate:>6.3f} {b.level1_corr:>+9.4f} "
                     f"{b.model_corr:>+11.4f} {b.model_brier:>12.4f} {b.base_rate_brier:>11.4f}")
        fails = self.promotion_failures()
        L.append("")
        L.append("PROMOTION: PASS -- model may drive live decisions if level2.enabled"
                 if not fails else "PROMOTION: FAIL -- the live bot will NOT trade on this model:")
        L.extend(f"  - {f}" for f in fails)
        return "\n".join(L)


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(ds: Dataset, *, n_blocks: int = 5, min_block_size: int = 200,
             corr_floor: float = 0.05) -> EvaluationReport:
    wf = train_and_evaluate_walk_forward(ds.X, ds.y, ds.epochs, n_blocks=n_blocks,
                                         min_block_size=min_block_size,
                                         feature_names=MODEL_FEATURE_NAMES)
    # Rebuild the exact same block boundaries baseline.py used (data is
    # already epoch-sorted) to score Level 1 and the base-rate forecaster
    # on identical held-out rows.
    n = len(ds.y)
    block = n // n_blocks
    rep = EvaluationReport(walk_forward_text=wf.report(), corr_floor=corr_floor)
    bi = 0
    for b in range(1, n_blocks):
        train_end, test_end = b * block, min((b + 1) * block, n)
        y_tr, y_te = ds.y[:train_end], ds.y[train_end:test_end]
        if len(y_te) < min_block_size or len(set(y_tr.tolist())) < 2:
            continue
        wb = wf.blocks[bi]
        bi += 1
        base = float(y_tr.mean())
        rep.blocks.append(BlockComparison(
            n_test=len(y_te), base_rate=base, model_corr=wb.correlation,
            level1_corr=_corr(ds.level1_score[train_end:test_end], y_te),
            model_brier=wb.brier,
            base_rate_brier=float(np.mean((base - y_te) ** 2))))
    return rep


# ----------------------------------------------------------------- the model

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class Level2Model:
    feature_names: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]
    coef: list[float]
    intercept: float
    platt_a: float
    platt_b: float
    bins: list[dict]              # calibrated-probability buckets on the newest slice
    promotion_status: str
    promotion_failures: list[str]
    version: str
    fingerprint: str
    trained_on: dict
    model_id: str = MODEL_ID

    @property
    def promoted(self) -> bool:
        return self.promotion_status == PROMOTED

    def predict(self, features: dict) -> tuple[float, float, float]:
        """-> (raw_probability, calibrated_probability, calibrated_lower_bound)."""
        x = np.array([float(features[n]) for n in self.feature_names])
        z = (x - np.asarray(self.scaler_mean)) / np.asarray(self.scaler_scale)
        raw = float(_sigmoid(float(z @ np.asarray(self.coef)) + self.intercept))
        cal = float(_sigmoid(self.platt_a * _logit(raw) + self.platt_b))
        n = 0
        for b in self.bins:
            if b["lo"] <= cal <= b["hi"]:
                n = b["n"]
                break
        if n <= 0:  # outside anything seen in calibration: no basis for confidence
            return raw, cal, 0.0
        # one-sided 95% bound from the calibration-bucket sample size
        lower = cal - 1.645 * math.sqrt(max(cal * (1 - cal), 1e-9) / n)
        return raw, cal, max(0.0, lower)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Level2Model":
        return cls(**json.loads(Path(path).read_text()))


def fit_final(ds: Dataset, cfg: dict, report: EvaluationReport, *,
              calibration_fraction: float = 0.2) -> Level2Model:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = len(ds.y)
    split = int(n * (1 - calibration_fraction))
    X_fit, y_fit = ds.X[:split], ds.y[:split]
    X_cal, y_cal = ds.X[split:], ds.y[split:]
    if len(set(y_fit.tolist())) < 2 or len(set(y_cal.tolist())) < 2:
        raise ValueError("need both wins and losses in the fit and calibration slices")

    scaler = StandardScaler().fit(X_fit)
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit((X_fit - scaler.mean_) / scale, y_fit)
    raw_cal = lr.predict_proba((X_cal - scaler.mean_) / scale)[:, 1]
    platt = LogisticRegression(max_iter=1000).fit(_logit(raw_cal).reshape(-1, 1), y_cal)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    p_cal = _sigmoid(a * _logit(raw_cal) + b)

    bins = []
    for chunk in np.array_split(np.argsort(p_cal), 10):
        if len(chunk) == 0:
            continue
        bins.append({"lo": float(p_cal[chunk].min()), "hi": float(p_cal[chunk].max()),
                     "n": int(len(chunk)), "mean_predicted": float(p_cal[chunk].mean()),
                     "actual": float(y_cal[chunk].mean())})
    # contiguous buckets: a probability landing between two buckets' observed
    # extremes belongs to the lower one, not to "never seen". Only values
    # beyond the calibration slice's overall min/max count as unseen.
    for prev, nxt in zip(bins, bins[1:]):
        prev["hi"] = nxt["lo"]

    fails = report.promotion_failures()
    return Level2Model(
        feature_names=list(MODEL_FEATURE_NAMES), scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scale.tolist(), coef=lr.coef_[0].tolist(),
        intercept=float(lr.intercept_[0]), platt_a=a, platt_b=b, bins=bins,
        promotion_status=PROMOTED if not fails else REJECTED, promotion_failures=fails,
        version=time.strftime("%Y%m%d-%H%M%S"), fingerprint=config_fingerprint(cfg),
        trained_on={"symbols": ds.symbols, "n_examples": int(n), "n_by_symbol": ds.n_by_symbol,
                    "epoch_start": float(ds.epochs.min()), "epoch_end": float(ds.epochs.max()),
                    "base_rate": float(ds.y.mean())})


def load_for_live(path: str, cfg: dict, logger) -> Level2Model | None:
    """The model the live bot may trade on, or None (with the reason logged).
    Never raises -- a missing or invalid model means Level 1 behavior."""
    try:
        model = Level2Model.load(path)
    except FileNotFoundError:
        logger.warning("level2.enabled but no model at %s -- run `python main.py --train`; "
                       "trading on Level 1 rules only", path)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("level2 model at %s unreadable (%s) -- Level 1 rules only", path, exc)
        return None
    if not model.promoted:
        logger.warning("level2 model %s is %s (%s) -- NOT trading on it; Level 1 rules only",
                       model.version, model.promotion_status, "; ".join(model.promotion_failures))
        return None
    if model.fingerprint != config_fingerprint(cfg):
        logger.error("level2 model %s was trained under a different feature/horizon config "
                     "-- its inputs no longer mean what it learned. Retrain. Level 1 rules only.",
                     model.version)
        return None
    return model
