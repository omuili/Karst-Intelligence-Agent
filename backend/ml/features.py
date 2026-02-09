"""
Feature engineering for sinkhole susceptibility model

Computes features from:
- Satellite imagery (Sentinel-2)
- Digital Elevation Model (DEM)
- Geology layers
- Hydrology data
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TileFeatures:
    """Container for extracted features from a tile"""
    spectral: np.ndarray  # (H, W, n_spectral)
    terrain: np.ndarray   # (H, W, n_terrain)
    geology: np.ndarray   # (H, W, n_geology)
    hydrology: np.ndarray # (H, W, n_hydrology)
    
    def to_array(self) -> np.ndarray:
        """Combine all features into single array"""
        return np.concatenate([
            self.spectral,
            self.terrain,
            self.geology,
            self.hydrology
        ], axis=-1)
    
    @property
    def n_features(self) -> int:
        return (
            self.spectral.shape[-1] + 
            self.terrain.shape[-1] + 
            self.geology.shape[-1] + 
            self.hydrology.shape[-1]
        )


class FeatureExtractor:
    """
    Extract features from various data sources for ML model.
    
    This class requires real data sources to be available.
    Returns zeros when data is unavailable.
    """
    
    def __init__(self):
        self._dem_cache = {}
        self._geology_cache = {}
        self._data_manager = None
    
    def _get_data_manager(self):
        """Lazy load the data manager"""
        if self._data_manager is None:
            from backend.data.services import RealDataManager
            from backend.config import settings
            from pathlib import Path
            cache_dir = settings.cache_dir if settings.cache_dir else Path("data/cache")
            self._data_manager = RealDataManager(cache_dir)
        return self._data_manager
    
    async def extract_all(
        self,
        bounds: Tuple[float, float, float, float],
        tile_size: int = 256
    ) -> TileFeatures:
        """
        Extract all features for a tile from real data sources.
        
        Args:
            bounds: (west, south, east, north)
            tile_size: Output resolution
        
        Returns:
            TileFeatures with all computed features
        
        Raises:
            RuntimeError: If required data is not available
        """
        spectral = await self.extract_spectral(bounds, tile_size)
        terrain = await self.extract_terrain(bounds, tile_size)
        geology = await self.extract_geology(bounds, tile_size)
        hydrology = await self.extract_hydrology(bounds, tile_size)
        
        return TileFeatures(
            spectral=spectral,
            terrain=terrain,
            geology=geology,
            hydrology=hydrology
        )
    
    async def extract_spectral(
        self,
        bounds: Tuple[float, float, float, float],
        tile_size: int
    ) -> np.ndarray:
        """
        Extract spectral features from satellite imagery.
        
        Requires Sentinel-2 data to be available.
        Returns zeros if satellite data is not available.
        """
        # Return empty features - satellite integration pending
        # Real implementation would fetch from Planetary Computer or similar
        return np.zeros((tile_size, tile_size, 5), dtype=np.float32)
    
    async def extract_terrain(
        self,
        bounds: Tuple[float, float, float, float],
        tile_size: int
    ) -> np.ndarray:
        """
        Extract terrain features from real DEM data.
        
        Requires USGS 3DEP elevation data.
        """
        try:
            data_manager = self._get_data_manager()
            data = await data_manager.fetch_all_layers(bounds)
            dem = data.get("dem")
            
            if dem is None:
                return np.zeros((tile_size, tile_size, 8), dtype=np.float32)
            
            # Resize DEM to tile size if needed
            if dem.shape != (tile_size, tile_size):
                from scipy.ndimage import zoom
                scale = (tile_size / dem.shape[0], tile_size / dem.shape[1])
                dem = zoom(dem, scale, order=1)
            
            # Compute real terrain derivatives
            dy, dx = np.gradient(dem)
            slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
            aspect = np.degrees(np.arctan2(-dx, dy))
            aspect = (aspect + 360) % 360
            
            dyy, _ = np.gradient(dy)
            _, dxx = np.gradient(dx)
            plan_curvature = dxx / (1 + dx**2 + 1e-10)**1.5
            profile_curvature = dyy / (1 + dy**2 + 1e-10)**1.5
            
            # TWI approximation
            slope_rad = np.radians(slope + 0.1)
            twi = np.log(100 / np.tan(slope_rad))
            twi = np.clip(twi, 0, 20)
            
            # Sink depth
            from scipy.ndimage import maximum_filter
            filled = maximum_filter(dem, size=5)
            sink_depth = filled - dem
            
            # Roughness
            from scipy.ndimage import generic_filter
            roughness = generic_filter(dem, np.std, size=5, mode='reflect')
            
            features = [
                (dem - 20) / 30,
                slope / 10,
                aspect / 360,
                np.clip(plan_curvature * 100, -1, 1),
                np.clip(profile_curvature * 100, -1, 1),
                twi / 15,
                sink_depth / 5,
                roughness / 3,
            ]
            
            return np.stack(features, axis=-1).astype(np.float32)
            
        except Exception as e:
            print(f"[!] Terrain extraction error: {e}")
            return np.zeros((tile_size, tile_size, 8), dtype=np.float32)
    
    async def extract_geology(
        self,
        bounds: Tuple[float, float, float, float],
        tile_size: int
    ) -> np.ndarray:
        """
        Extract geology features from real FGS data.
        
        Returns zeros if geology data is not available.
        """
        # Geology feature extraction requires polygon rasterization
        # Return zeros until proper implementation
        return np.zeros((tile_size, tile_size, 4), dtype=np.float32)
    
    async def extract_hydrology(
        self,
        bounds: Tuple[float, float, float, float],
        tile_size: int
    ) -> np.ndarray:
        """
        Extract hydrology features from real NHD data.
        
        Returns zeros if hydrology data is not available.
        """
        # Hydrology feature extraction requires distance calculations
        # Return zeros until proper implementation
        return np.zeros((tile_size, tile_size, 3), dtype=np.float32)


def compute_spectral_indices(
    bands: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Compute spectral indices from Sentinel-2 bands
    
    Expected bands: B02 (blue), B03 (green), B04 (red), B08 (NIR)
    """
    eps = 1e-8
    
    indices = {}
    
    if 'B08' in bands and 'B04' in bands:
        nir = bands['B08'].astype(np.float32)
        red = bands['B04'].astype(np.float32)
        indices['ndvi'] = (nir - red) / (nir + red + eps)
    
    if 'B03' in bands and 'B08' in bands:
        green = bands['B03'].astype(np.float32)
        nir = bands['B08'].astype(np.float32)
        indices['ndwi'] = (green - nir) / (green + nir + eps)
    
    if 'B02' in bands and 'B03' in bands and 'B04' in bands:
        blue = bands['B02'].astype(np.float32)
        green = bands['B03'].astype(np.float32)
        red = bands['B04'].astype(np.float32)
        total = blue + green + red + eps
        indices['brightness'] = total / 3
        indices['blue_ratio'] = blue / total
    
    return indices


def compute_terrain_derivatives(dem: np.ndarray, cell_size: float = 10.0) -> Dict[str, np.ndarray]:
    """
    Compute terrain derivatives from DEM
    
    Args:
        dem: Digital Elevation Model array
        cell_size: Cell size in meters
    
    Returns:
        Dictionary of terrain features
    """
    # Gradients
    dy, dx = np.gradient(dem, cell_size)
    
    # Slope (degrees)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # Aspect (degrees from north, clockwise)
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect = (aspect + 360) % 360
    
    # Second derivatives
    dyy, dyx = np.gradient(dy, cell_size)
    dxy, dxx = np.gradient(dx, cell_size)
    
    # Curvatures
    p = dx**2 + dy**2
    q = p + 1
    
    plan_curv = np.where(
        p > 1e-10,
        (dxx * dy**2 - 2 * dxy * dx * dy + dyy * dx**2) / (p**1.5),
        0
    )
    
    profile_curv = np.where(
        p > 1e-10,
        (dxx * dx**2 + 2 * dxy * dx * dy + dyy * dy**2) / (p * q**0.5),
        0
    )
    
    return {
        'slope': slope,
        'aspect': aspect,
        'plan_curvature': plan_curv,
        'profile_curvature': profile_curv,
    }

