"""
Configuration for Sinkhole Susceptibility Scanner
Target: Winter Park, Florida
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    google_cloud_project: Optional[str] = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    planetary_computer_api_key: Optional[str] = Field(default=None, alias="PLANETARY_COMPUTER_API_KEY")
    opentopography_api_key: Optional[str] = Field(default=None, alias="OPENTOPOGRAPHY_API_KEY")
    
    # Vertex AI Configuration
    google_cloud_region: str = Field(default="us-central1", alias="GOOGLE_CLOUD_REGION")
    use_vertex_ai: bool = Field(default=False, alias="USE_VERTEX_AI")
    
    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=True, alias="DEBUG")
    
    # Model (field name avoids Pydantic protected namespace 'model_')
    ml_model_path: str = Field(default="models/sinkhole_susceptibility.joblib", alias="MODEL_PATH")
    use_gemini_features: bool = Field(default=True, alias="USE_GEMINI_FEATURES")
    
    # Tile cache
    enable_tile_cache: bool = Field(default=True, alias="ENABLE_TILE_CACHE")
    clear_cache_on_startup: bool = Field(default=False, alias="CLEAR_CACHE_ON_STARTUP")
    
    # Computed paths (set after init)
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    
    def model_post_init(self, __context):
        # Set derived paths
        object.__setattr__(self, 'data_dir', self.base_dir / "data")
        object.__setattr__(self, 'models_dir', self.base_dir / "models")
        object.__setattr__(self, 'cache_dir', self.data_dir / "cache")
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        (self.cache_dir / "tiles").mkdir(exist_ok=True)
        
        # Clear cache on startup if enabled
        if self.clear_cache_on_startup:
            self._clear_tile_cache()
    
    def _clear_tile_cache(self):
        """Clear all cached tiles"""
        tiles_dir = self.cache_dir / "tiles"
        if tiles_dir.exists():
            count = 0
            for tile_file in tiles_dir.glob("*.png"):
                tile_file.unlink()
                count += 1
            if count > 0:
                print(f"[Config] Cleared {count} cached tiles on startup")
    
    class Config:
        # Load .env from project root so OPENTOPOGRAPHY_API_KEY etc. are found regardless of cwd
        env_file = str(Path(__file__).resolve().parent.parent / ".env")
        env_file_encoding = "utf-8"
        populate_by_name = True


# Winter Park, Florida - Area of Interest Configuration
class WinterParkAOI:
    """
    Winter Park, Florida AOI Configuration
    
    Winter Park sits on the Central Florida Karst terrain, part of the 
    Floridan Aquifer System. The area is underlain by Eocene-Oligocene 
    limestone (Ocala Limestone, Suwannee Limestone) making it highly 
    susceptible to cover-collapse and cover-subsidence sinkholes.
    """
    
    # Bounding box [west, south, east, north] in WGS84
    BBOX = [-81.4200, 28.5500, -81.3200, 28.6300]
    
    # Center coordinates
    CENTER_LAT = 28.5983
    CENTER_LON = -81.3510
    
    # Name and description
    NAME = "Winter Park, Florida"
    DESCRIPTION = "Central Florida Karst Region - High sinkhole susceptibility zone"
    
    # Coordinate reference systems
    CRS_WGS84 = "EPSG:4326"
    CRS_UTM = "EPSG:32617"  # UTM Zone 17N for Central Florida
    CRS_WEB_MERCATOR = "EPSG:3857"
    
    # Tile settings
    TILE_SIZE = 512  # pixels
    MIN_ZOOM = 10
    MAX_ZOOM = 18
    DEFAULT_ZOOM = 14
    
    # Resolution settings (meters per pixel at different zooms)
    ANALYSIS_RESOLUTION = 10  # 10m resolution for analysis
    
    # Grid cell size for susceptibility calculation (meters)
    GRID_CELL_SIZE = 30
    
    # Feature buffer distances (meters)
    FAULT_BUFFER_DISTANCES = [100, 250, 500, 1000]
    KARST_BUFFER_DISTANCES = [50, 100, 250, 500]
    WATER_BUFFER_DISTANCES = [50, 100, 250]
    
    @classmethod
    def get_geojson_bbox(cls):
        """Return AOI as GeoJSON polygon"""
        west, south, east, north = cls.BBOX
        return {
            "type": "Feature",
            "properties": {
                "name": cls.NAME,
                "description": cls.DESCRIPTION
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [west, south],
                    [east, south],
                    [east, north],
                    [west, north],
                    [west, south]
                ]]
            }
        }
    
    @classmethod
    def get_area_km2(cls):
        """Calculate approximate area in square kilometers"""
        west, south, east, north = cls.BBOX
        # Approximate using local coordinate conversion
        lat_km = (north - south) * 111.0  # ~111 km per degree latitude
        lon_km = (east - west) * 111.0 * 0.877  # cos(28.6°) ≈ 0.877
        return lat_km * lon_km


# Florida state extent – main map shows full state
FLORIDA_BBOX = (-87.64, 24.52, -80.03, 31.0)  # (west, south, east, north) WGS84


class FloridaAOI:
    """Florida state extent for main map (full state view)."""
    BBOX = FLORIDA_BBOX
    NAME = "Florida"
    DESCRIPTION = "State-wide sinkhole susceptibility"
    CENTER_LAT = 27.8
    CENTER_LON = -81.8
    MIN_ZOOM = 6
    MAX_ZOOM = 18
    DEFAULT_ZOOM = 7
    TILE_SIZE = 512

    @classmethod
    def get_geojson_bbox(cls):
        west, south, east, north = cls.BBOX
        return {
            "type": "Feature",
            "properties": {"name": cls.NAME, "description": cls.DESCRIPTION},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
            },
        }

    @classmethod
    def get_area_km2(cls):
        west, south, east, north = cls.BBOX
        lat_km = (north - south) * 111.0
        lon_km = (east - west) * 111.0 * 0.9
        return lat_km * lon_km


# Florida Geological Survey Data Sources
class FloridaDataSources:
    """URLs and metadata for Florida geological data"""
    
    # Florida Geological Survey - Subsidence Incident Reports (VERIFIED WORKING Jan 2026)
    FGS_SINKHOLE_URL = "https://ca.dep.state.fl.us/arcgis/rest/services/OpenData/FGS_SUBSIDENCE/MapServer/0"
    
    # Florida Geological Survey - Geology/Sediment Distribution (VERIFIED WORKING Jan 2026)
    FGS_GEOLOGY_URL = "https://ca.dep.state.fl.us/arcgis/rest/services/OpenData/FGS_PUBLIC/MapServer/0"
    
    # Florida Geological Survey - Karst Districts
    FGS_KARST_URL = "https://ca.dep.state.fl.us/arcgis/rest/services/OpenData/FGS_PUBLIC/MapServer/10"
    
    # USGS 3DEP Elevation
    USGS_3DEP_URL = "https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Elevation"
    
    # Planetary Computer - Sentinel-2
    SENTINEL2_COLLECTION = "sentinel-2-l2a"
    
    # National Hydrography Dataset
    NHD_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"
    
    # Karst geology keywords for filtering
    KARST_LITHOLOGY_KEYWORDS = [
        "limestone",
        "dolomite", 
        "ocala",
        "suwannee",
        "avon park",
        "floridan",
        "carbonate"
    ]


# Susceptibility model configuration
class ModelConfig:
    """ML model configuration"""
    
    # Feature columns for XGBoost model
    SPECTRAL_FEATURES = [
        "ndvi",           # Normalized Difference Vegetation Index
        "ndwi",           # Normalized Difference Water Index
        "brightness",     # Overall brightness
        "blue_ratio",     # Blue band ratio (water/moisture)
        "red_edge",       # Vegetation stress indicator
    ]
    
    TERRAIN_FEATURES = [
        "elevation",
        "slope",
        "aspect",
        "curvature_plan",
        "curvature_prof",
        "twi",            # Topographic Wetness Index
        "sink_depth",     # Filled DEM - actual DEM (depression depth)
        "roughness",
    ]
    
    GEOLOGY_FEATURES = [
        "dist_to_karst",
        "dist_to_fault",
        "lithology_class",
        "karst_density",
    ]
    
    HYDROLOGY_FEATURES = [
        "dist_to_water",
        "drainage_density",
        "flow_accumulation",
    ]
    
    # XGBoost hyperparameters
    XGBOOST_PARAMS = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    # Class weights for imbalanced data (sinkholes are rare)
    POSITIVE_CLASS_WEIGHT = 10.0
    
    # Probability thresholds for classification
    THRESHOLD_LOW = 0.2
    THRESHOLD_MEDIUM = 0.4
    THRESHOLD_HIGH = 0.6
    THRESHOLD_VERY_HIGH = 0.8


# Gemini configuration
class GeminiConfig:
    """Gemini 3 API configuration for agentic sinkhole analysis (model: gemini-3-pro-preview)"""
    
    # gemini-3-pro-preview - reasoning model, multimodal (text, images, audio, video, PDF)
    # 1M token context window, multimodal (text, images, audio, video, PDF)
    MODEL_NAME = "gemini-3-pro-preview"
    
    # Alternative Gemini 3 models:
    # - gemini-3-flash-preview: Faster, for high-volume tasks
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 60
    MAX_TILES_PER_BATCH = 10
    
    # Temperature for deterministic outputs (low for consistency)
    TEMPERATURE = 0.1
    
    # Maximum tokens for responses
    MAX_OUTPUT_TOKENS = 8192
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    # Gemini 3 Agentic Features
    THINKING_LEVEL = "medium"  # low, medium, high - controls reasoning depth
    MEDIA_RESOLUTION = "high"  # low, medium, high - for multimodal inputs
    ENABLE_THOUGHT_SIGNATURES = True  # Maintain chain-of-reasoning across calls


# Load settings
settings = Settings()

