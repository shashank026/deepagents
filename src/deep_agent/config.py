import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    model_provider: str = os.getenv("MODEL_PROVIDER", "ollama")
    model_name: str = os.getenv("MODEL_NAME", "qwen3:8b")
    ollama_base_url: str = os.getenv(
        "MODAL_BASE_URL",
        "http://localhost:11434",
    )


settings = Settings()