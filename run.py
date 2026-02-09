import os
import sys
import asyncio
import argparse
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))


# API key from .env or Render environment (never hardcode)
os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "8080")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("USE_GEMINI_FEATURES", "true")


def main():
    parser = argparse.ArgumentParser(description="Sinkhole Scanner - Real Data Edition")
    parser.add_argument("--data", action="store_true", help="Fetch real data from FGS/USGS/NHD")
    parser.add_argument("--train", action="store_true", help="Train model with real FGS sinkhole data")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()

    
    Path("data/cache/tiles").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

   
    if args.data:
        print("\n[*] Fetching REAL data from live sources...")
        print("    - Florida Geological Survey (sinkholes, geology)")
        print("    - USGS 3DEP (elevation)")
        print("    - National Hydrography Dataset (water)")
        print()
        from backend.data.download_data import main as download_main
        asyncio.run(download_main())
        print()


    if args.train:
        print("\n[*] Training model with REAL FGS sinkhole data...")
        from backend.ml.train_model import main as train_main
        asyncio.run(train_main())
        print()

    print("\n" + "="*60)
    print("  SINKHOLE SCANNER - Real Data Edition")
    print("="*60)
    print(f"\n  Server:   http://{args.host}:{args.port}")
    print(f"  API Docs: http://{args.host}:{args.port}/docs")
    print(f"\n  Data Sources:")
    print(f"    - Florida Geological Survey (LIVE)")
    print(f"    - USGS 3DEP Elevation (LIVE)")
    print(f"    - National Hydrography Dataset (LIVE)")
    print(f"\n  Gemini API: Configured")
    print("="*60 + "\n")
    
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()

