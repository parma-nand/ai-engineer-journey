from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str = "Resume Summarizer API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    MAX_FILE_SIZE_MB: int = 5
    MAX_PAGES: int = 5
    UPLOAD_DIR: str = "uploads"

    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000", "http://localhost:8501"]

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()