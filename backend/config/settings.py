import os


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name, "") or ""
    return [s.strip() for s in raw.split(",") if s.strip()]


class Settings:
    # canonical env-driven fields
    ENV              = os.getenv("ENV", "dev")
    DEBUG            = _env_bool("DEBUG", True)
    MYSTIC_BACKEND   = os.getenv("MYSTIC_BACKEND", "http://127.0.0.1:9000")

    DISPLAY_EXCHANGE = os.getenv("DISPLAY_EXCHANGE", "binanceus")
    DISPLAY_TOP_N    = _env_int("DISPLAY_TOP_N", 5)
    DISPLAY_SYMBOLS  = _env_list("DISPLAY_SYMBOLS")

    # --------- Back-compat aliases expected by older modules ----------
    # e.g. code uses settings.exchange / settings.top_n / settings.symbols
    @property
    def exchange(self) -> str:
        return self.DISPLAY_EXCHANGE

    @property
    def top_n(self) -> int:
        return self.DISPLAY_TOP_N

    @property
    def symbols(self) -> list[str]:
        return self.DISPLAY_SYMBOLS

    # allow settings["KEY"] lookups as a last resort
    def __getitem__(self, key: str):
        return getattr(self, key, os.getenv(key))

    # and a very forgiving getattr fallback to env
    def __getattr__(self, name: str):
        v = os.getenv(name)
        if v is not None:
            return v
        raise AttributeError(name)


settings = Settings()
