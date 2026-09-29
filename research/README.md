# Forex / stock-index Rise/Fall research (29 Sep 2026)

Reproduces the verdict recorded in the design doc. Input: the 5-minute candle
cache written by `tools/rise_fall_research.py` (`data/research/candles/`),
copied to `research/candles/` (not committed).

    python features.py        # builds fx_features.pkl (every 15 min, 21 pairs)
    # then the analyses in wf.py / common.py (walk-forward, holdout)

Split: development before 2026-06-01, holdout 2026-06-01..2026-09-29, touched
once with two pre-committed candidates.

| Test (out of sample) | Win | Break-even |
| --- | --- | --- |
| Model, forex, p>=0.62, 120m, holdout | 52.8% (n=1,574; 95% CI 47-59%) | 55% |
| Fade >3 sigma stretch, 60m, holdout | 47.2% (n=559) | 55% |
| Model, stock indices, Jun-Sep walk-forward | 49-51% | 55% |

Hours 20:00-23:59 UTC are excluded: the daily rollover produces a spurious
~73% "edge" from quote artifacts that a Rise/Fall buyer cannot capture.
