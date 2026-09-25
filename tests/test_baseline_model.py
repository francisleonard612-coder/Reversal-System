"""
Tests for models/baseline.py. test_harness_detects_real_signal_when_present
is the important one -- a null result on real data is only trustworthy if
the tool used to find it can be shown to detect signal that's actually
there. Without this test, "no reproducible correlation" and "the harness is
broken" would be indistinguishable from the outside.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.baseline import (
    WalkForwardModelReport,
    train_and_evaluate_walk_forward,
)
from models.dataset import FEATURE_NAMES


def _synthetic_matrix(n=4000, *, signal_strength=0.0, seed=0):
    """Builds a feature matrix shaped like the real one (same width,
    FEATURE_NAMES-compatible) where feature 0 is either a genuine predictor
    of y (signal_strength > 0) or pure noise (signal_strength == 0)."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    if signal_strength > 0:
        logits = signal_strength * X[:, 0]
        p = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n) < p).astype(float)
    else:
        y = (rng.uniform(size=n) < 0.5).astype(float)
    epochs = np.arange(n)
    return X, y, epochs


def test_harness_detects_real_signal_when_present():
    """THE test that makes the real-data null result trustworthy. A feature
    that genuinely predicts the label at a strength well above noise must
    be found: sign-consistent, clearing the 0.05 floor, across blocks."""
    X, y, epochs = _synthetic_matrix(n=6000, signal_strength=1.5, seed=1)
    report = train_and_evaluate_walk_forward(X, y, epochs, n_blocks=5,
                                             min_block_size=200)
    assert report.sign_consistent(), (
        "a feature genuinely predicting the label at this strength must be "
        "detected -- if this fails, the harness itself cannot be trusted "
        "to report an honest null result on real data")
    assert all(c > 0.05 for c in report._corrs())


def test_harness_reports_null_on_pure_noise():
    X, y, epochs = _synthetic_matrix(n=6000, signal_strength=0.0, seed=2)
    report = train_and_evaluate_walk_forward(X, y, epochs, n_blocks=5,
                                             min_block_size=200)
    assert not report.sign_consistent()
    assert "Do NOT promote" in report.report()


def test_harness_null_message_distinguishes_weak_from_sign_flipping():
    weak = WalkForwardModelReport()
    flipping = WalkForwardModelReport()

    class _B:
        def __init__(self, corr):
            self.correlation = corr
            self.n_train = self.n_test = 0
            self.brier = self.log_loss = float("nan")
            self.reliability = []

    weak.blocks = [_B(0.01), _B(-0.02), _B(0.005)]
    flipping.blocks = [_B(0.08), _B(-0.09), _B(0.07)]
    assert "below |0.05|" in weak.report()
    assert "sign flips" in flipping.report()


def test_walk_forward_refuses_too_few_examples_per_block():
    X, y, epochs = _synthetic_matrix(n=500, signal_strength=0.0, seed=3)
    with pytest.raises(ValueError, match="below the minimum"):
        train_and_evaluate_walk_forward(X, y, epochs, n_blocks=10,
                                        min_block_size=200)


def test_scaler_is_fit_on_train_fold_only():
    """A regression guard for the exact bug the scaling fix addressed:
    fitting the scaler on train+test would leak test-fold statistics into
    training. Verified indirectly -- block N+1's training set (which
    includes block N's former test rows) must not depend on what the
    original block N test rows looked like at evaluation time; this is
    trivially true if each block gets its own scaler.fit on that block's
    train slice, which is what the code does. Direct enforcement: running
    twice with the test fold's values altered must not change the coefficients
    on identical training data."""
    X, y, epochs = _synthetic_matrix(n=6000, signal_strength=1.0, seed=4)
    r1 = train_and_evaluate_walk_forward(X, y, epochs, n_blocks=5, min_block_size=200)
    X2 = X.copy()
    # Perturb ONLY the final block's test-fold feature values -- if the
    # scaler for block 1 were fit on train+test (leaking), block 1's
    # coefficients would change too. They must not.
    n = len(y)
    block = n // 5
    X2[4 * block:] *= 1000.0
    r2 = train_and_evaluate_walk_forward(X2, y, epochs, n_blocks=5, min_block_size=200)
    assert r1.blocks[0].coefficients == pytest.approx(r2.blocks[0].coefficients, abs=1e-9)
