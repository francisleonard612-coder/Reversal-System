"""
Configuration loading (spec Section 56). config/settings.yaml holds every
bounded parameter; env vars override the leaves an operator actually needs
to touch per-deploy (see .env.example). Nothing here silently repairs an
invalid config -- see `validate()`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    pass


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None else int(v)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else v.strip()


class Settings:
    def __init__(self, path: str | None = None):
        load_dotenv()
        path = path or os.getenv("CONFIG_PATH", "config/settings.yaml")
        with open(path) as f:
            self.raw: dict = yaml.safe_load(f)

        self.mode = _env_str("TRADING_MODE", self.raw.get("mode", "research"))
        self.use_real_account = _env_bool("DERIV_USE_REAL", False)
        if self.mode == "live" and not self.use_real_account:
            raise ConfigError("TRADING_MODE=live requires DERIV_USE_REAL=true")
        if self.mode != "live" and self.use_real_account:
            raise ConfigError("DERIV_USE_REAL=true requires TRADING_MODE=live")

        symbols_env = os.getenv("SYMBOLS")
        self.symbols = ([s.strip() for s in symbols_env.split(",") if s.strip()]
                        if symbols_env else self.raw.get("symbols", ["R_100"]))

        self.currency = _env_str("CURRENCY", self.raw.get("currency", "USD"))

        d = self.raw.setdefault("deriv", {})
        d["app_id"] = _env_str("DERIV_APP_ID", d.get("app_id", "1089"))
        d["api_token"] = _env_str("DERIV_API_TOKEN", d.get("api_token", ""))
        d["ws_url"] = _env_str("DERIV_WS_URL", d.get("ws_url",
                               "wss://ws.derivws.com/websockets/v3"))
        d["auth_mode"] = _env_str("DERIV_AUTH_MODE", d.get("auth_mode", "otp"))
        d["api_base_url"] = _env_str("DERIV_API_BASE_URL",
                                     d.get("api_base_url", "https://api.derivws.com"))
        d["account_id"] = _env_str("DERIV_ACCOUNT_ID", d.get("account_id", ""))
        d["request_timeout"] = _env_float("DERIV_REQUEST_TIMEOUT",
                                          d.get("request_timeout", 15.0))
        d["max_requests_per_minute"] = _env_int(
            "DERIV_MAX_REQUESTS_PER_MINUTE", d.get("max_requests_per_minute", 300))
        self.deriv = d

        r = self.raw.setdefault("risk", {})
        r["base_stake"] = _env_float("BASE_STAKE", r.get("base_stake", 1.0))
        r["max_stake"] = _env_float("MAX_STAKE", r.get("max_stake", 5.0))
        r["max_daily_loss"] = _env_float("MAX_DAILY_LOSS", r.get("max_daily_loss", 25.0))
        r["max_drawdown"] = _env_float("MAX_DRAWDOWN", r.get("max_drawdown", 50.0))
        r["max_consecutive_losses"] = _env_int(
            "MAX_CONSECUTIVE_LOSSES", r.get("max_consecutive_losses", 8))
        self.risk = r

        s = self.raw.setdefault("staking", {})
        s["method"] = _env_str("STAKING_METHOD", s.get("method", "fixed"))
        # Section 37: martingale off by default, without exception, for this
        # system -- unlike the sibling Even/Odd bot, there is no override
        # path here that flips this default; martingale_enabled is read from
        # config/settings.yaml only, not from env, so a stray environment
        # variable can never turn it on.
        s.setdefault("martingale_enabled", False)
        self.staking = s

        self.storage_path = _env_str("DB_PATH", self.raw.get("storage", {}).get(
            "path", "data/reversal.db"))
        self.db_backend = _env_str("DB_BACKEND", "sqlite")
        self.database_url = _env_str("DATABASE_URL", "")

        c = self.raw.setdefault("contract", {})
        c["duration"] = _env_int("CONTRACT_DURATION", c.get("duration", 5))
        c["duration_unit"] = _env_str("CONTRACT_DURATION_UNIT", c.get("duration_unit", "m"))
        c.setdefault("duration_bars_approx", 5)

        r.setdefault("max_trades_per_day", 100)
        r.setdefault("max_concurrent_trades", 1)
        r.setdefault("cooldown_seconds", 0.0)
        r.setdefault("stale_tick_seconds", 30.0)
        r.setdefault("max_proposal_age_seconds", 5.0)
        r.setdefault("settlement_grace_seconds", 60.0)
        r["max_trades_per_day"] = _env_int("MAX_TRADES_PER_DAY", r["max_trades_per_day"])
        r["cooldown_seconds"] = _env_float("COOLDOWN_SECONDS", r["cooldown_seconds"])

        l2 = self.raw.setdefault("level2", {})
        l2["enabled"] = _env_bool("LEVEL2_ENABLED", l2.get("enabled", False))
        l2["model_path"] = _env_str("LEVEL2_MODEL_PATH", l2.get("model_path", "data/level2_model.json"))
        l2["min_ev"] = _env_float("LEVEL2_MIN_EV", l2.get("min_ev", 0.0))
        l2.setdefault("require_lower_bound_edge", True)
        l2.setdefault("n_blocks", 5)
        l2.setdefault("min_block_size", 200)
        l2.setdefault("promotion_corr_floor", 0.05)
        self.level2 = l2

        ex = self.raw.setdefault("expiry", {})
        ex["enabled"] = _env_bool("EXPIRY_CALIBRATION", ex.get("enabled", True))
        ex["require_edge"] = _env_bool("EXPIRY_REQUIRE_EDGE", ex.get("require_edge", True))
        ex["min_samples"] = _env_int("EXPIRY_MIN_SAMPLES", ex.get("min_samples", 200))
        hs = os.getenv("EXPIRY_HORIZONS")
        if hs:
            ex["horizons"] = [h.strip() for h in hs.split(",") if h.strip()]
        ex.setdefault("horizons", ["3t", "5t", "7t", "10t", "1m", "2m", "3m", "5m"])
        for k, v in (("lookback_days", 14), ("credible_quantile", 0.05), ("mc_draws", 4000),
                     ("pool_symbols", True), ("proposals_to_price", 3),
                     ("refresh_seconds", 300), ("seconds_per_tick", 2.0)):
            ex.setdefault(k, v)
        from strategy.expiry import parse_horizons
        try:
            self.expiry_horizons = parse_horizons(ex["horizons"])
        except ValueError as exc:
            raise ConfigError(f"expiry.horizons: {exc}") from exc
        self.expiry = ex

        ops = self.raw.setdefault("operations", {})
        ops.setdefault("reconcile_interval_seconds", 300)
        ops.setdefault("balance_refresh_seconds", 300)
        self.operations = ops

        self.log_level = _env_str("LOG_LEVEL", self.raw.get("logging", {}).get(
            "level", "INFO"))

        self._validate()

    def _validate(self) -> None:
        weights = self.raw["reversal"]["weights"]
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"reversal.weights must sum to 1.0, got {total:.4f}: {weights}")
        if self.raw["staking"].get("martingale_enabled") and self.mode == "live":
            logger.warning(
                "martingale_enabled=true in live mode -- Section 37 defaults "
                "this off for a reason; confirm this was deliberate")
        if self.risk["base_stake"] > self.risk["max_stake"]:
            raise ConfigError("risk.base_stake must not exceed risk.max_stake")
        self.horizon_matches_research = self._check_contract_horizon()

    def contract_seconds(self) -> float | None:
        """Live contract length in seconds, or None for tick contracts
        (whose wall-clock length isn't fixed)."""
        c = self.raw["contract"]
        unit = str(c["duration_unit"]).lower()
        mult = {"s": 1, "m": 60, "h": 3600}.get(unit)
        return None if mult is None else float(c["duration"]) * mult

    def _check_contract_horizon(self) -> bool:
        c = self.raw["contract"]
        tf = self.raw["candles"]["timeframe_seconds"]
        live_s = self.contract_seconds()
        research_s = c["duration_bars_approx"] * tf
        if live_s is None:
            logger.warning(
                "contract is %s %s but backtests, reconciliation and Level 2 "
                "labels all measure %d candles (%ds). Nothing in this repo "
                "validates a tick-length contract -- its results are unmeasured.",
                c["duration"], c["duration_unit"], c["duration_bars_approx"], research_s)
            return False
        if abs(live_s - research_s) > 1e-9:
            raise ConfigError(
                f"live contract lasts {live_s:.0f}s ({c['duration']}{c['duration_unit']}) "
                f"but research measures {research_s}s (contract.duration_bars_approx="
                f"{c['duration_bars_approx']} x {tf}s candles). They must match, or "
                f"every backtest/reconcile/Level 2 number describes a different bet "
                f"than the one being traded.")
        return True

    def __getitem__(self, key: str):
        return self.raw[key]

    def get(self, key: str, default=None):
        return self.raw.get(key, default)
