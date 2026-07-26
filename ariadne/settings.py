from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://ariadne:ariadne@localhost:5433/ariadne"
    test_database_url: str = "postgresql+psycopg://ariadne:ariadne@localhost:5433/ariadne_test"
    redis_url: str = "redis://localhost:6380/0"

    tmdb_api_key: str = ""

    # Requests per second against TMDB. Their published ceiling is far higher; this is
    # deliberate politeness for a hobby project hitting a free API.
    tmdb_rate_limit: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
