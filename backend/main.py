"""
Sinkhole Susceptibility Scanner - Main FastAPI Application
"""

# Load .env from project root first so OPENTOPOGRAPHY_API_KEY etc. are available
import os
from pathlib import Path
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

# Vertex AI on Render: allow credentials from env (no key file on disk)
_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if _creds_json and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    import tempfile
    import json
    try:
        json.loads(_creds_json)
        _fd, _path = tempfile.mkstemp(suffix=".json", prefix="gcp-creds-")
        with os.fdopen(_fd, "w") as f:
            f.write(_creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _path
        print("[*] Vertex AI: using credentials from GOOGLE_APPLICATION_CREDENTIALS_JSON")
    except Exception as e:
        print(f"[!] GOOGLE_APPLICATION_CREDENTIALS_JSON invalid: {e}")

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings, FloridaAOI, WinterParkAOI
from backend.api.tiles import router as tiles_router
from backend.api.analysis import router as analysis_router
from backend.api.terrain import router as terrain_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - startup and shutdown"""
    print("[*] Sinkhole Susceptibility Scanner Starting...")
    print(f"    Map: {FloridaAOI.NAME} (full state)")
    print(f"    Bounding Box: {FloridaAOI.BBOX}")
    print(f"    Area: {FloridaAOI.get_area_km2():.0f} km2")
    
    # Initialize model if exists
    model_path = settings.base_dir / settings.ml_model_path
    if model_path.exists():
        print(f"    Model loaded from: {model_path}")
    else:
        print("    [!] No trained model found - will use heuristic susceptibility")
    
    yield
    
    print("[*] Sinkhole Scanner shutting down...")


# Create FastAPI application
app = FastAPI(
    title="Sinkhole Susceptibility Scanner",
    description="AI-powered sinkhole susceptibility mapping for Winter Park, Florida",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(tiles_router, prefix="/api/tiles", tags=["tiles"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["analysis"])
app.include_router(terrain_router, prefix="/api", tags=["terrain"])


# WebSocket connection manager for progress updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket endpoint for streaming scan progress"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Echo back or process commands
            await websocket.send_json({"status": "received", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# API endpoints
@app.get("/api/config")
async def get_config():
    """Return AOI configuration for frontend (Florida full state)."""
    return {
        "aoi": {
            "name": FloridaAOI.NAME,
            "description": FloridaAOI.DESCRIPTION,
            "bbox": FloridaAOI.BBOX,
            "center": [FloridaAOI.CENTER_LAT, FloridaAOI.CENTER_LON],
            "geojson": FloridaAOI.get_geojson_bbox(),
            "area_km2": FloridaAOI.get_area_km2(),
        },
        "map": {
            "minZoom": FloridaAOI.MIN_ZOOM,
            "maxZoom": FloridaAOI.MAX_ZOOM,
            "defaultZoom": FloridaAOI.DEFAULT_ZOOM,
            "tileSize": FloridaAOI.TILE_SIZE,
        },
        "features": {
            "geminiEnabled": bool(settings.gemini_api_key),
            "modelTrained": (settings.base_dir / settings.ml_model_path).exists(),
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "karst-intelligence-agent"}


# Serve static frontend files
frontend_path = Path(__file__).resolve().parent.parent / "frontend"
print(f"    Frontend path: {frontend_path}")
print(f"    Frontend exists: {frontend_path.exists()}")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend application"""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(content=f"""
    <html>
        <head><title>Karst Intelligence Agent</title></head>
        <body>
            <h1>Sinkhole Susceptibility Scanner</h1>
            <p>Frontend not found at: {frontend_path}</p>
            <p>API documentation: <a href="/docs">/docs</a></p>
        </body>
    </html>
    """)


# Mount static files AFTER defining routes
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

