from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",  # Ignores undeclared environment variables
        env_file=".env",
        case_sensitive=False,
    )

    PROJECT_NAME: str = "AI Trace API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Redis Configuration
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB_CACHE: int = 0
    REDIS_DB_BROKER: int = 1
    REDIS_DB_RESULTS: int = 2

    HF_TOKEN: Optional[str] = None

    @property
    def REDIS_CACHE_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_CACHE}"

    @property
    def CELERY_BROKER_URL(self) -> str:
        return (
            f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_BROKER}"
        )

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_RESULTS}"

    # Cache TTL (24 Hours)
    CACHE_TTL_SECONDS: int = 86400


settings = Settings()