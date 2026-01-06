import os
from dotenv import load_dotenv

load_dotenv()  # carga .env automáticamente

def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value
