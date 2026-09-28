"""Level 2 training/promotion/serving. Uses feature matrices directly (fast);
the full replay path is exercised in test_dataset.py."""
from __future__ import annotations

import logging

import numpy as np
import pytest
import yaml

from models.dataset import MODEL_FEATURE_NAMES
from models.level2 import (PROMOTED, REJECTED, Dataset, Level2Model, config_fingerprint,
                           evaluate, fit_final, load_for_live)


def _cfg():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def _dataset(signal: bool, n=6000, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(MODEL_FEATURE_NAMES)))
    logit = 1.2 * X[:, 0] if signal else np.zeros(n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return Dataset(X=X, y=y, epochs=np.arange(n, dtype=float),
                   level1_score=rng.normal(size=n), symbols=["R_100"], n_by_symbol={"R_100": n})


def test_real_signal_is_promoted_and_calibrated():
    ds = _dataset(signal=True)
    rep = evaluate(ds, n_blocks=5, min_block_size=200)
    assert rep.promotion_failures() == []
    model = fit_final(ds, _cfg(), rep)
    assert model.promotion_status == PROMOTED
    for b in model.bins:   # calibration: predicted tracks actual on the held-out slice
        assert abs(b["mean_predicted"] - b["actual"]) < 0.1


def test_noise_is_rejected():
    ds = _dataset(signal=False)
    rep = evaluate(ds, n_blocks=5, min_block_size=200)
    assert rep.promotion_failures()
    assert fit_final(ds, _cfg(), rep).promotion_status == REJECTED


def test_anti_predictive_model_is_not_promoted():
    """Sign-consistent but NEGATIVE correlation must fail -- baseline.py's own
    sign_consistent() would accept it."""
    from models.level2 import BlockComparison, EvaluationReport
    rep = EvaluationReport(blocks=[BlockComparison(500, .5, -0.2, 0, .24, .25)] * 3)
    assert rep.promotion_failures()


def test_prediction_lower_bound_is_conservative():
    ds = _dataset(signal=True)
    model = fit_final(ds, _cfg(), evaluate(ds))
    x = {n: 0.0 for n in MODEL_FEATURE_NAMES}
    x[MODEL_FEATURE_NAMES[0]] = 1.5
    raw, cal, lo = model.predict(x)
    assert 0 < lo < cal < 1


def test_roundtrip_and_live_gates(tmp_path):
    cfg = _cfg()
    log = logging.getLogger("t")
    good = fit_final(_dataset(True), cfg, evaluate(_dataset(True)))
    path = tmp_path / "m.json"
    good.save(str(path))
    loaded = load_for_live(str(path), cfg, log)
    assert loaded is not None
    x = {n: 0.3 for n in MODEL_FEATURE_NAMES}
    assert loaded.predict(x) == good.predict(x)

    cfg2 = _cfg()
    cfg2["contract"]["duration_bars_approx"] = 3          # different label horizon
    assert config_fingerprint(cfg2) != config_fingerprint(cfg)
    assert load_for_live(str(path), cfg2, log) is None

    bad = fit_final(_dataset(False), cfg, evaluate(_dataset(False)))
    bad.save(str(path))
    assert load_for_live(str(path), cfg, log) is None
    assert load_for_live(str(tmp_path / "missing.json"), cfg, log) is None


def test_artifact_is_plain_json_not_pickle(tmp_path):
    import json
    m = fit_final(_dataset(True), _cfg(), evaluate(_dataset(True)))
    m.save(str(tmp_path / "m.json"))
    json.loads((tmp_path / "m.json").read_text())
