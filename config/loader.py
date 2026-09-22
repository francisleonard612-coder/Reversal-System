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

    def __getitem__(self, key: str):
        return self.raw[key]

    def get(self, key: str, default=None):
        return self.raw.get(key, default)
