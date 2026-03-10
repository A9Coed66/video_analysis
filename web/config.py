"""Application configuration using pydantic-settings.

All settings are loaded from environment variables (or .env file).
Required: API_KEYS, REDIS_URL must be non-empty.
Optional: HF_TOKEN — if empty, Pipeline_Service is disabled.
"""

from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Voice Separator Web.

    Environment variables are mapped to lowercase field names automatically.
    For example, ``REDIS_URL`` env var maps to ``redis_url``.
    """

    # --- Required ---
    redis_url: str = "redis://redis:6379/0"
    api_keys: str = ""  # comma-separated list of valid API keys

    # --- File / Job ---
    max_file_size: int = 500 * 1024 * 1024  # 500 MB
    job_ttl_hours: int = 24
    allowed_origins: str = "http://localhost"

    # --- ML / HuggingFace ---
    hf_token: str = ""  # optional – empty disables Pipeline_Service

    # --- Model checkpoint paths ---
    dprnn_checkpoint_dir: str = "/models/dprnn"
    pipeline_model_dir: str = "/models/pipeline"

    # --- GPU assignment ---
    gpu_separation: int = 0
    gpu_pipeline: int = 1

    # --- Server ---
    uvicorn_workers: int = 4
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # -- Validators ----------------------------------------------------------

    @field_validator("api_keys")
    @classmethod
    def api_keys_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "API_KEYS is required. "
                "Provide a comma-separated list of API keys."
            )
        return v.strip()

    @model_validator(mode="after")
    def redis_url_must_not_be_empty(self) -> "Settings":
        if not self.redis_url or not self.redis_url.strip():
            raise ValueError(
                "REDIS_URL is required. "
                "Provide a valid Redis connection URL (e.g. redis://redis:6379/0)."
            )
        return self

    # -- Derived properties ---------------------------------------------------

    @property
    def api_key_list(self) -> list[str]:
        """Split comma-separated API_KEYS into a deduplicated list."""
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def pipeline_enabled(self) -> bool:
        """Pipeline_Service is available only when HF_TOKEN is provided."""
        return bool(self.hf_token and self.hf_token.strip())


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so the config is parsed once per process."""
    return Settings()
