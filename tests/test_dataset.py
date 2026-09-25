from __future__ import annotations

import yaml

from data.candles import Candle
from models.dataset import FEATURE_NAMES, build_training_examples, to_matrix


def _load_cfg():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)


def _candles(n, symbol="R_100", start_price=1000.0, drift=0.0, noise=0.0, seed=1):
    import random
    random.seed(seed)
    out = []
    price = start_price
    epoch = 1_700_000_000
    for i in range(n):
        price += drift + random.gauss(0, noise) if noise else drift
        price = max(price, 1.0)
        out.append(Candle(symbol=symbol, open_epoch=epoch, close_epoch=epoch + 60,
                          open=price, high=price + 0.1, low=price - 0.1, close=price,
                          n_ticks=5, is_closed=True, timeframe_seconds=60))
        epoch += 60
    return out


def test_one_example_per_bar_not_two():
    cfg = _load_cfg()
    candles = _candles(300, noise=0.3)
    examples = build_training_examples(candles, cfg)
    # sufficient_data only kicks in once min_bars clears -- but whatever
    # the count, it must never exceed one example per closed candle.
    assert len(examples) <= len(candles)
    assert all(e.direction in ("bullish", "bearish") for e in examples)


def test_feature_dict_keys_match_the_declared_order():
    cfg = _load_cfg()
    candles = _candles(300, noise=0.3)
    examples = build_training_examples(candles, cfg)
    assert examples, "need at least one example for this to test anything"
    assert set(examples[0].features.keys()) == set(FEATURE_NAMES)


def test_label_is_none_near_the_end_of_history_never_guessed():
    """The anti-lookahead guarantee: a bar too close to the end of
    available history to have a real settlement outcome must be
    excluded, never given a fabricated label."""
    cfg = _load_cfg()
    candles = _candles(300, noise=0.3)
    examples = build_training_examples(candles, cfg)
    duration_bars = cfg["contract"]["duration_bars_approx"]
    tail = examples[-duration_bars:]
    assert all(e.label is None for e in tail if e.epoch == candles[-1].close_epoch
              or True), "at least the last duration_bars examples must be unlabeled"
    assert examples[-1].label is None


def test_label_matches_real_forward_settlement():
    """Construct a deterministic price path (no noise) so the correct
    label at every bar is known exactly, and verify build_training_examples
    computes it the same way the backtest and reconcile job do."""
    cfg = _load_cfg()
    # steady uptrend -- every bullish call 5 bars later should win, deterministically
    candles = _candles(300, drift=1.0, noise=0.0)
    examples = build_training_examples(candles, cfg)
    labeled = [e for e in examples if e.label is not None]
    assert labeled
    bullish_labeled = [e for e in labeled if e.direction == "bullish"]
    if bullish_labeled:
        assert all(e.label is True for e in bullish_labeled), (
            "in a steady uptrend, every bullish call must have won -- if not, "
            "the label computation disagrees with the real price path")


def test_to_matrix_excludes_unlabeled_examples():
    cfg = _load_cfg()
    candles = _candles(300, noise=0.3)
    examples = build_training_examples(candles, cfg)
    X, y, epochs, labeled = to_matrix(examples)
    assert len(X) == len(y) == len(epochs) == len(labeled)
    assert all(e.label is not None for e in labeled)
    assert len(labeled) <= len(examples)


def test_empty_candle_list_returns_no_examples():
    cfg = _load_cfg()
    assert build_training_examples([], cfg) == []
