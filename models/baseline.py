"""
Level 2 baseline model (spec Sections 21, 23, 43).

LOGISTIC REGRESSION FIRST, DELIBERATELY (Section 21's own ordering: "Do not
use deep learning merely because it sounds advanced," and Section 2's
progression more generally). A gradient-boosted challenger only gets built
once this baseline has a real number to beat -- there is no number yet,
which is the point of this module.

CALIBRATION IS NOT OPTIONAL (Section 23). A logistic regression's raw
output is already a probability in form, but "predicting 0.6" and "being
right 60% of the time" are different claims -- Platt scaling or isotonic
regression is what closes that gap, and both are evaluated here via
reliability buckets, not assumed.

THE PROMOTION BAR (Section 43, Section 32's spirit even before a real
champion/challenger system exists): out-of-sample walk-forward evaluation,
not a single train/test split, and the criterion this module actually
reports on is DIRECTIONAL AGREEMENT WITH THE ALREADY-ESTABLISHED
CORRELATION FLOOR used earlier in this conversation (0.05, sign-consistent
across blocks) -- a model whose calibrated probability doesn't clear that
bar against real held-out outcomes has not earned a place in the trading
decision, no matter how good its training-set Brier score looks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from models.dataset import FEATURE_NAMES


@dataclass
class BlockResult:
    n_train: int
    n_test: int
    brier: float
    log_loss: float
    correlation: float           # calibrated P(win) vs actual outcome, held-out
    reliability: list[tuple]     # (bucket_mean_predicted, bucket_actual_rate, bucket_n)
    coefficients: dict           # feature name -> fitted weight, for this block


@dataclass
class WalkForwardModelReport:
    blocks: list[BlockResult] = field(default_factory=list)

    def _corrs(self) -> list[float]:
        return [b.correlation for b in self.blocks if b.correlation == b.correlation]

    def sign_consistent(self, floor: float = 0.05) -> bool:
        """Same bar as the sibling bot's walk-forward harness and the same
        one used earlier in this conversation on the raw blended score:
        every block's correlation must share a sign AND clear the floor.
        A small sd around a sign that flips, or around a value near zero,
        is noise either way -- see app/backtest/simulator.py's
        WalkForwardReport in the Even/Odd repo for the fuller reasoning."""
        vals = self._corrs()
        if len(vals) < 2:
            return False
        same_sign = all(v > 0 for v in vals) or all(v < 0 for v in vals)
        strong_enough = all(abs(v) >= floor for v in vals)
        return same_sign and strong_enough

    def report(self) -> str:
        L = [f"LEVEL 2 BASELINE -- walk-forward, {len(self.blocks)} blocks"]
        for i, b in enumerate(self.blocks, 1):
            L.append(f"\n--- block {i} (train={b.n_train}, test={b.n_test}) ---")
            L.append(f"  brier={b.brier:.4f}  log_loss={b.log_loss:.4f}  "
                     f"held-out corr(P(win), outcome)={b.correlation:+.4f}")
            L.append("  reliability (predicted -> actual, n):")
            for pred, actual, n in b.reliability:
                L.append(f"    {pred:.2f} -> {actual:.2f}  (n={n})")
        corrs = [b.correlation for b in self.blocks]
        L.append(f"\ncorrelation per block: " + ", ".join(f"{c:+.4f}" for c in corrs))
        if self.sign_consistent():
            L.append("-> sign-consistent AND clears the 0.05 floor. This is "
                     "the first evidence in this project that a model is "
                     "finding something the fixed blended score wasn't.")
        else:
            vals = self._corrs()
            if vals and all(abs(v) < 0.05 for v in vals):
                L.append("-> every block is below |0.05| -- no reproducible "
                         "signal yet, whatever the sign. Do NOT promote this "
                         "model or use it to drive trading decisions.")
            else:
                L.append("-> sign flips across blocks -- not reproducible. "
                         "Do NOT promote this model or use it to drive "
                         "trading decisions.")
        return "\n".join(L)


def _pearson(xs, ys) -> float:
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs) < 3:
        return float("nan")
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def train_and_evaluate_walk_forward(X: np.ndarray, y: np.ndarray,
                                    epochs: np.ndarray, *, n_blocks: int = 5,
                                    min_block_size: int = 200
                                    ) -> WalkForwardModelReport:
    """Section 41's TRAIN -> VALIDATE -> TEST -> DEPLOY -> ROLL FORWARD,
    applied to a real classifier: fit on everything before a block,
    evaluate ON that block, advance. A held-out block's own correlation
    between the model's calibrated probability and the REAL outcome is the
    number that decides whether this model means anything -- not the
    training-set fit, which a logistic regression with 19 features can
    make look good on noise alone given enough of it.
    """
    order = np.argsort(epochs)
    X, y, epochs = X[order], y[order], epochs[order]
    n = len(y)
    block = n // n_blocks
    if block < min_block_size:
        raise ValueError(
            f"{n} examples over {n_blocks} blocks gives {block} per block, "
            f"below the minimum of {min_block_size} needed to evaluate "
            f"anything -- use more data or fewer blocks")

    report = WalkForwardModelReport()
    for b in range(1, n_blocks):
        train_end = b * block
        test_end = min((b + 1) * block, n)
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]
        if len(X_test) < min_block_size or len(set(y_train.tolist())) < 2:
            continue

        # Fit the scaler on TRAIN ONLY -- fitting it on train+test would
        # leak the test fold's own feature distribution into training,
        # which is a real form of lookahead even though no label crosses
        # the boundary. Unscaled features (atr_pct ~1e-3, adx 0-100,
        # n_candles_available in the hundreds) left lbfgs failing to
        # converge in earlier runs, which would have made a null result
        # untrustworthy -- an unconverged fit could be masking real signal
        # as easily as confirming its absence.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit(X_train_scaled, y_train)
        p_test = model.predict_proba(X_test_scaled)[:, 1]

        brier = float(brier_score_loss(y_test, p_test))
        try:
            ll = float(log_loss(y_test, p_test, labels=[0.0, 1.0]))
        except ValueError:
            ll = float("nan")
        corr = _pearson(p_test, y_test)

        try:
            actual, predicted = calibration_curve(y_test, p_test, n_bins=5,
                                                   strategy="quantile")
            bins = np.array_split(np.argsort(p_test), 5)
            reliability = [(float(predicted[i]), float(actual[i]), len(bins[i]))
                          for i in range(len(predicted))]
        except ValueError:
            reliability = []

        coefficients = dict(zip(FEATURE_NAMES, model.coef_[0].tolist()))
        report.blocks.append(BlockResult(
            n_train=len(X_train), n_test=len(X_test), brier=brier, log_loss=ll,
            correlation=corr, reliability=reliability, coefficients=coefficients))

    return report


def top_coefficients(report: WalkForwardModelReport, k: int = 8) -> list[tuple]:
    """Averages each feature's fitted coefficient across blocks and ranks by
    magnitude. Coefficients are in STANDARDIZED units (per training fold's
    own StandardScaler) -- comparable across features regardless of each
    one's native scale (atr_pct ~1e-3 vs adx 0-100), unlike raw-unit
    coefficients would be. A quick read on which evidence the model
    actually leans on, for comparison against Level 1's fixed weights
    (reversal/scoring.py) -- though see train_and_evaluate_walk_forward's
    caller: coefficients from a model that failed sign_consistent() are a
    description of what the model fit to, not evidence any of it is real.
    """
    if not report.blocks:
        return []
    sums = {name: 0.0 for name in FEATURE_NAMES}
    for b in report.blocks:
        for name, coef in b.coefficients.items():
            sums[name] += coef
    avg = {name: v / len(report.blocks) for name, v in sums.items()}
    return sorted(avg.items(), key=lambda kv: -abs(kv[1]))[:k]
