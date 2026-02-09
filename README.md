# Karst Intelligence Agent

An autonomous AI-powered web application for sinkhole susceptibility mapping and early warning, using satellite imagery, geospatial data, and **Google Gemini 3** for multi-step analysis, scan validation, and alert drafting.

**Target Area:** Winter Park, Florida (Central Florida Karst Region)

> **Contest submission:** See [SUBMISSION.md](SUBMISSION.md) for the Gemini integration description, project link, repo, and demo video checklist.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Leaflet)                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ Scanner UI  │  │ Heatmap Tiles│  │ Feature Boxes/Masks    │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/WebSocket
┌────────────────────────────▼────────────────────────────────────┐
│                     FastAPI Backend                              │
│  ┌──────────┐  ┌───────────────┐  ┌────────────────────────┐    │
│  │ Tile API │  │ ML Inference  │  │ Gemini Feature Extract │    │
│  └──────────┘  └───────────────┘  └────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      Data Pipeline                               │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────────────┐  │
│  │ Satellite  │  │ DEM/Terrain │  │ Geology/Sinkhole Inv.    │  │
│  │ (Sentinel) │  │ (USGS)      │  │ (Florida Geological Srv) │  │
│  └────────────┘  └─────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend dev server, optional)
- Google Cloud account with Gemini API access

### Installation

```bash
# Clone and enter directory
cd sinkhole-scanner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Download initial data for Winter Park AOI
python -m backend.data.download_data

# Run the application
python -m backend.main
```

### Access the App
Open http://localhost:8000 in your browser

### Deploy on Render
One Web Service runs the full app (API + frontend). See **[docs/DEPLOY_RENDER.md](docs/DEPLOY_RENDER.md)** for:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Required env vars: `DEBUG=false`, `GEMINI_API_KEY` (or Vertex AI settings)

You can use the **render.yaml** blueprint in the repo root to create the service, then set secrets in the Render dashboard.

## 📊 Data Sources

| Layer | Source | Resolution |
|-------|--------|------------|
| Optical Imagery | Sentinel-2 (Copernicus) | 10m |
| DEM | USGS 3DEP | 10m |
| Geology | Florida Geological Survey | Vector |
| Sinkhole Inventory | FGS Subsidence Incident Reports | Points |
| Karst Features | USGS Karst Map | Vector |

## 🤖 ML Pipeline

The susceptibility model uses **XGBoost** trained on:
- **Spectral features**: NDVI, NDWI, brightness indices
- **Terrain features**: Slope, curvature, TWI, sink-fill depressions
- **Geology features**: Distance to karst units, fault proximity, lithology class
- **Hydrology**: Drainage density, distance to water bodies

## 🔮 Gemini Integration

Gemini is used for:
1. **Weak labeling**: Detecting sinkhole-like depressions in imagery
2. **Feature extraction**: Structured JSON of risk factors per tile
3. **Quality control**: Flagging model/imagery conflicts

## 📁 Project Structure

```
sinkhole-scanner/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration & AOI settings
│   ├── api/
│   │   ├── tiles.py         # Tile serving endpoints
│   │   ├── analysis.py      # Analysis endpoints
│   │   └── websocket.py     # Progress streaming
│   ├── ml/
│   │   ├── features.py      # Feature engineering
│   │   ├── model.py         # XGBoost model
│   │   └── inference.py     # Inference pipeline
│   ├── gemini/
│   │   ├── client.py        # Gemini API client
│   │   └── prompts.py       # Prompt templates
│   └── data/
│       ├── download_data.py # Data fetching utilities
│       └── preprocessing.py # Data preprocessing
├── frontend/
│   ├── index.html           # Main application
│   ├── css/
│   │   └── style.css        # Styling
│   └── js/
│       ├── app.js           # Main application logic
│       ├── map.js           # Map initialization
│       ├── scanner.js       # Scanner animation
│       └── api.js           # API client
├── data/                    # Downloaded/cached data
├── models/                  # Trained models
├── requirements.txt
├── .env.example
└── README.md
```

## 🎨 Features

- **Interactive Map**: Pan/zoom with base layer options
- **Scanning Animation**: Real-time tile processing visualization
- **Susceptibility Heatmap**: Color-coded probability overlay
- **Feature Detection**: Bounding boxes for detected sinkhole candidates
- **Analysis Reports**: Per-tile and AOI-wide statistics

## 📜 License

MIT License - See LICENSE file

