"""
Tile API endpoints for serving susceptibility map tiles
"""

import io
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, JSONResponse
import mercantile

from backend.config import settings, WinterParkAOI, ModelConfig
from backend.ml.real_inference import RealSusceptibilityInference


router = APIRouter()

# Initialize inference engine (lazy loaded)
_inference_engine: Optional[RealSusceptibilityInference] = None
_data_loaded: bool = False


def get_inference_engine() -> RealSusceptibilityInference:
    """Get or create the real inference engine"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = RealSusceptibilityInference()
    return _inference_engine


async def ensure_data_loaded():
    """Ensure AOI data is loaded"""
    global _data_loaded
    if not _data_loaded:
        engine = get_inference_engine()
        await engine.load_aoi_data()
        _data_loaded = True


@router.post("/preload")
@router.get("/data-status")
async def preload_or_data_status():
    """
    Preload all AOI data (including Sentinel-2 and ground displacement) and return status.
    Used by the scanner UI to show which data sources were loaded (e.g. Sentinel, OPERA).
    """
    await ensure_data_loaded()
    engine = get_inference_engine()
    status = engine.get_loaded_data_status()
    return status


def tile_pixel_mask_inside_aoi(
    tile_bounds: Tuple[float, float, float, float],
    tile_size: int,
    aoi_bbox: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Return a 2D boolean mask (tile_size x tile_size) where True = pixel center
    is inside the AOI. Used to clip the susceptibility overlay to the valid
    footprint so nodata areas are transparent (no "big square" edges).
    """
    west, south, east, north = tile_bounds
    aoi_west, aoi_south, aoi_east, aoi_north = aoi_bbox
    h, w = tile_size, tile_size
    # Pixel (i, j): center lon/lat (top-left = (0,0) → (west, north))
    j = np.arange(w)
    i = np.arange(h)
    lon = west + (j + 0.5) / w * (east - west)
    lat = north - (i + 0.5) / h * (north - south)
    lon_2d = np.broadcast_to(lon, (h, w))
    lat_2d = np.broadcast_to(lat[:, np.newaxis], (h, w))
    mask = (
        (lon_2d >= aoi_west) & (lon_2d <= aoi_east) &
        (lat_2d >= aoi_south) & (lat_2d <= aoi_north)
    )
    return mask


def create_heatmap_image(
    susceptibility: np.ndarray,
    colormap: str = "risk"
) -> bytes:
    """
    Convert susceptibility array to a colored PNG image
    
    Args:
        susceptibility: 2D array of probabilities [0, 1]
        colormap: Color scheme to use
    
    Returns:
        PNG image bytes
    """
    # Ensure valid range
    susceptibility = np.clip(susceptibility, 0, 1)
    
    # Create RGBA image
    height, width = susceptibility.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Risk colormap: green -> yellow -> orange -> red
    # Low risk (0-0.2): Green with low opacity
    # Medium-Low (0.2-0.4): Yellow-green
    # Medium (0.4-0.6): Orange
    # High (0.6-0.8): Red-orange
    # Very High (0.8-1.0): Dark red
    
    for i in range(height):
        for j in range(width):
            p = susceptibility[i, j]
            
            if p < 0.05:
                # Very low - transparent
                rgba[i, j] = [0, 0, 0, 0]
            elif p < 0.2:
                # Low - green
                rgba[i, j] = [34, 197, 94, int(80 + p * 200)]
            elif p < 0.4:
                # Medium-low - yellow-green
                t = (p - 0.2) / 0.2
                rgba[i, j] = [
                    int(34 + t * (234 - 34)),
                    int(197 - t * (17)),
                    int(94 - t * 94),
                    int(120 + t * 50)
                ]
            elif p < 0.6:
                # Medium - orange
                t = (p - 0.4) / 0.2
                rgba[i, j] = [
                    int(234 + t * (249 - 234)),
                    int(180 - t * (65)),
                    0,
                    int(170 + t * 30)
                ]
            elif p < 0.8:
                # High - red-orange
                t = (p - 0.6) / 0.2
                rgba[i, j] = [
                    int(249 - t * 30),
                    int(115 - t * 80),
                    0,
                    int(200 + t * 25)
                ]
            else:
                # Very high - dark red
                t = (p - 0.8) / 0.2
                rgba[i, j] = [
                    int(220 - t * 50),
                    int(35 - t * 35),
                    int(t * 50),
                    int(225 + t * 30)
                ]
    
    # Create PIL image and save to bytes
    img = Image.fromarray(rgba, mode='RGBA')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    return buffer.getvalue()


def _viridis_rgb(p: np.ndarray) -> np.ndarray:
    """
    Perceptually uniform Viridis-like ramp: 0 = dark purple, 1 = yellow.
    p in [0, 1]; returns (N, 3) uint8 RGB.
    """
    # Viridis key points (R, G, B) 0-255 at p = 0, 0.25, 0.5, 0.75, 1
    keys = np.array([
        [68, 1, 84],      # 0.00
        [59, 82, 139],    # 0.25
        [33, 145, 140],   # 0.50
        [94, 201, 98],    # 0.75
        [253, 231, 37],   # 1.00
    ], dtype=np.float64)
    idx = np.clip(p * 4, 0, 4)  # 0..4
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, 4)
    t = idx - i0
    r = (1 - t) * keys[i0, 0] + t * keys[i1, 0]
    g = (1 - t) * keys[i0, 1] + t * keys[i1, 1]
    b = (1 - t) * keys[i0, 2] + t * keys[i1, 2]
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def create_heatmap_vectorized(
    susceptibility: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> bytes:
    """
    Professional susceptibility tile: continuous Viridis-style ramp + nodata mask.
    
    Fix 3: nodata outside valid footprint (mask=False) → fully transparent.
    Fix 4: Perceptually uniform color scheme (Viridis: low=dark purple, high=yellow);
    opacity 0.35–0.55 so overlay supports the basemap instead of overpowering it.
    """
    susceptibility = np.clip(susceptibility, 0, 1)
    height, width = susceptibility.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    if mask is None:
        mask = np.ones((height, width), dtype=bool)
    
    # Continuous Viridis ramp for all pixels inside mask
    flat = susceptibility[mask]
    rgb = _viridis_rgb(flat)
    rgba[mask, 0] = rgb[:, 0]
    rgba[mask, 1] = rgb[:, 1]
    rgba[mask, 2] = rgb[:, 2]
    # Alpha: visible for submission; same scale across all tiles
    alpha = (140 + susceptibility[mask] * 90).astype(np.float32)
    rgba[mask, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    
    # Feathered edge at AOI boundary (no hard line)
    feathered = gaussian_filter(mask.astype(np.float32), sigma=6)
    feathered = np.clip(feathered, 0, 1)
    rgba[:, :, 3] = (rgba[:, :, 3].astype(np.float32) * feathered).astype(np.uint8)
    rgba[feathered < 0.02, :] = 0
    
    img = Image.fromarray(rgba, mode='RGBA')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    return buffer.getvalue()


@router.get("/susceptibility/{z}/{x}/{y}.png")
async def get_susceptibility_tile(
    z: int,
    x: int,
    y: int,
    tile_size: int = Query(default=256, ge=64, le=1024),
):
    """
    Get susceptibility heatmap tile as PNG (XYZ tile scheme for Leaflet).
    
    CRS/bounds (Fix 1): All bounds are WGS84 (EPSG:4326).
    - Tile bounds from mercantile: (west, south, east, north) = (lon_min, lat_min, lon_max, lat_max).
    - We do NOT swap lat/lon; overlay is georeferenced to these bounds.
    - Pixel (0,0) = top-left of tile = (west, north); last pixel = (east, south).
    
    Returns a semi-transparent overlay with perceptually uniform (Viridis-style)
    color ramp; nodata outside AOI is fully transparent (clip + mask).
    """
    # Validate zoom level
    if z < WinterParkAOI.MIN_ZOOM or z > WinterParkAOI.MAX_ZOOM:
        raise HTTPException(
            status_code=400,
            detail=f"Zoom level must be between {WinterParkAOI.MIN_ZOOM} and {WinterParkAOI.MAX_ZOOM}"
        )
    
    # Get tile bounds
    tile = mercantile.Tile(x=x, y=y, z=z)
    bounds = mercantile.bounds(tile)
    
    # Check if tile intersects AOI
    west, south, east, north = WinterParkAOI.BBOX
    if (bounds.east < west or bounds.west > east or 
        bounds.north < south or bounds.south > north):
        # Return fully transparent tile (nodata outside AOI)
        empty = np.zeros((tile_size, tile_size), dtype=np.float32)
        mask_nodata = np.zeros((tile_size, tile_size), dtype=bool)
        return Response(
            content=create_heatmap_vectorized(empty, mask=mask_nodata),
            media_type="image/png"
        )
    
    # Ensure real data is loaded
    await ensure_data_loaded()
    
    # Use same cache key as your existing good tiles (z_x_y_256.png) so we serve them
    cache_key = f"{z}_{x}_{y}_{tile_size}"
    cache_path = settings.cache_dir / "tiles" / f"{cache_key}.png"
    
    # Serve from cache if we have this exact tile.
    if settings.enable_tile_cache and cache_path.exists():
        return Response(
            content=cache_path.read_bytes(),
            media_type="image/png"
        )
    
    # When zoomed in past z=15, serve the parent tile at z=15 so the overlay does not disappear.
    # Leaflet requests one tile per grid cell; we return the covering z=15 tile for each.
    if z > 15 and settings.enable_tile_cache:
        delta = z - 15
        x15 = x >> delta
        y15 = y >> delta
        parent_key = f"15_{x15}_{y15}_{tile_size}"
        parent_path = settings.cache_dir / "tiles" / f"{parent_key}.png"
        if parent_path.exists():
            return Response(
                content=parent_path.read_bytes(),
                media_type="image/png"
            )
    
    # No cache and no parent: return transparent (do not generate).
    empty = np.zeros((tile_size, tile_size), dtype=np.float32)
    transparent = create_heatmap_vectorized(empty, mask=np.zeros((tile_size, tile_size), dtype=bool))
    return Response(content=transparent, media_type="image/png")


@router.get("/features/{z}/{x}/{y}.json")
async def get_feature_tile(
    z: int,
    x: int,
    y: int,
):
    """
    Get REAL sinkhole features for a tile as GeoJSON
    
    Returns actual sinkholes from Florida Geological Survey + Gemini detections
    """
    # Ensure real data is loaded
    await ensure_data_loaded()
    
    tile = mercantile.Tile(x=x, y=y, z=z)
    bounds = mercantile.bounds(tile)
    
    # Check if tile intersects AOI
    west, south, east, north = WinterParkAOI.BBOX
    if (bounds.east < west or bounds.west > east or 
        bounds.north < south or bounds.south > north):
        return {"type": "FeatureCollection", "features": []}
    
    # Get REAL features from FGS data + Gemini
    try:
        engine = get_inference_engine()
        features = await engine.detect_features(
            bounds=(bounds.west, bounds.south, bounds.east, bounds.north),
            zoom=z
        )
    except Exception as e:
        print(f"[!] Feature detection error: {e}")
        features = {"type": "FeatureCollection", "features": []}
    
    return features


@router.get("/info/{z}/{x}/{y}")
async def get_tile_info(z: int, x: int, y: int):
    """Get metadata about a specific tile"""
    tile = mercantile.Tile(x=x, y=y, z=z)
    bounds = mercantile.bounds(tile)
    
    return {
        "tile": {"z": z, "x": x, "y": y},
        "bounds": {
            "west": bounds.west,
            "south": bounds.south,
            "east": bounds.east,
            "north": bounds.north,
        },
        "center": {
            "lat": (bounds.south + bounds.north) / 2,
            "lon": (bounds.west + bounds.east) / 2,
        },
        "in_aoi": is_tile_in_aoi(bounds),
    }


def is_tile_in_aoi(bounds) -> bool:
    """Check if tile bounds intersect the AOI"""
    west, south, east, north = WinterParkAOI.BBOX
    return not (
        bounds.east < west or 
        bounds.west > east or 
        bounds.north < south or 
        bounds.south > north
    )

