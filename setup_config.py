

import os
from pathlib import Path



CONFIG = {
    "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
    "HOST": "0.0.0.0",
    "PORT": "8080",
    "DEBUG": "true",
    "USE_GEMINI_FEATURES": "true",
    "ENABLE_TILE_CACHE": "true",
}


def create_env_file():
  
    env_path = Path(__file__).parent / ".env"
    
    with open(env_path, "w") as f:
        for key, value in CONFIG.items():
            f.write(f"{key}={value}\n")
    
    print(f"Created: {env_path}")


def set_environment():

    for key, value in CONFIG.items():
        os.environ[key] = value
    print("Environment variables set")


def create_directories():

    base = Path(__file__).parent
    
    dirs = [
        base / "data",
        base / "data" / "cache",
        base / "data" / "cache" / "tiles",
        base / "models",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")


if __name__ == "__main__":
    print("Setting up Karts Intelligence Agent configuration...\n")
    
    create_directories()
    create_env_file()
    set_environment()
    
    print("\nConfiguration complete!")
    key = CONFIG["GEMINI_API_KEY"]
    print(f"\nGemini API Key: {'set (' + key[:8] + '...)' if key else 'not set (add GEMINI_API_KEY to .env)'}")
    print(f"Server will run on: http://{CONFIG['HOST']}:{CONFIG['PORT']}")

