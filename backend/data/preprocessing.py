"""
Data preprocessing utilities for sinkhole susceptibility analysis

Handles:
- Raster alignment and resampling
- Feature stack creation
- Data normalization
- Training data preparation
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RasterStack:
    """Container for aligned raster data"""
    data: np.ndarray  # (bands, height, width)
    band_names: List[str]
    transform: any
    crs: str
    bounds: Tuple[float, float, float, float]
    nodata: float = -9999.0


class RasterPreprocessor:
    """
    Preprocesses raster data for ML model input
    """
    
    def __init__(self, target_resolution: float = 10.0):
        """
        Args:
            target_resolution: Target resolution in meters
        """
        self.target_resolution = target_resolution
        self.stats: Dict[str, Dict[str, float]] = {}
    
    def align_rasters(
        self,
        raster_paths: Dict[str, Path],
        bounds: Tuple[float, float, float, float],
        target_crs: str = "EPSG:32617",
    ) -> RasterStack:
        """
        Align multiple rasters to common grid
        
        Args:
            raster_paths: Dictionary of name -> path to rasters
            bounds: Target bounds (west, south, east, north)
            target_crs: Target coordinate reference system
        
        Returns:
            RasterStack with aligned data
        """
        import rasterio
        from rasterio.warp import reproject, Resampling, calculate_default_transform
        from rasterio.transform import from_bounds
        
        west, south, east, north = bounds
        
        # Calculate output dimensions
        width = int((east - west) * 111000 / self.target_resolution)  # Approximate
        height = int((north - south) * 111000 / self.target_resolution)
        
        transform = from_bounds(west, south, east, north, width, height)
        
        aligned_bands = []
        band_names = []
        
        for name, path in raster_paths.items():
            with rasterio.open(path) as src:
                # Reproject to target grid
                data = np.zeros((height, width), dtype=np.float32)
                
                reproject(
                    source=rasterio.band(src, 1),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )
                
                aligned_bands.append(data)
                band_names.append(name)
        
        return RasterStack(
            data=np.stack(aligned_bands),
            band_names=band_names,
            transform=transform,
            crs=target_crs,
            bounds=bounds,
        )
    
    def compute_indices(
        self,
        sentinel_stack: RasterStack,
    ) -> Dict[str, np.ndarray]:
        """
        Compute spectral indices from Sentinel-2 bands
        
        Expected bands: B02 (blue), B03 (green), B04 (red), B08 (NIR)
        """
        eps = 1e-8
        indices = {}
        
        # Get bands by name
        band_dict = {
            name: sentinel_stack.data[i]
            for i, name in enumerate(sentinel_stack.band_names)
        }
        
        # NDVI
        if 'B08' in band_dict and 'B04' in band_dict:
            nir = band_dict['B08'].astype(np.float32)
            red = band_dict['B04'].astype(np.float32)
            indices['ndvi'] = (nir - red) / (nir + red + eps)
        
        # NDWI
        if 'B03' in band_dict and 'B08' in band_dict:
            green = band_dict['B03'].astype(np.float32)
            nir = band_dict['B08'].astype(np.float32)
            indices['ndwi'] = (green - nir) / (green + nir + eps)
        
        # Brightness
        if all(b in band_dict for b in ['B02', 'B03', 'B04']):
            blue = band_dict['B02'].astype(np.float32)
            green = band_dict['B03'].astype(np.float32)
            red = band_dict['B04'].astype(np.float32)
            indices['brightness'] = (blue + green + red) / 3
            
            total = blue + green + red + eps
            indices['blue_ratio'] = blue / total
        
        return indices
    
    def compute_terrain_features(
        self,
        dem: np.ndarray,
        cell_size: float = 10.0,
    ) -> Dict[str, np.ndarray]:
        """
        Compute terrain derivatives from DEM
        """
        from scipy.ndimage import gaussian_filter, generic_filter, maximum_filter
        
        features = {}
        
        # Smooth DEM slightly
        dem_smooth = gaussian_filter(dem.astype(np.float32), sigma=1)
        
        # Elevation (normalized)
        features['elevation'] = dem_smooth
        
        # Gradients
        dy, dx = np.gradient(dem_smooth, cell_size)
        
        # Slope (degrees)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        features['slope'] = slope
        
        # Aspect (degrees from north)
        aspect = np.degrees(np.arctan2(-dx, dy))
        features['aspect'] = (aspect + 360) % 360
        
        # Second derivatives for curvature
        dyy, _ = np.gradient(dy, cell_size)
        _, dxx = np.gradient(dx, cell_size)
        
        # Plan and profile curvature (simplified)
        features['plan_curvature'] = dxx
        features['profile_curvature'] = dyy
        
        # TWI - Topographic Wetness Index (simplified)
        slope_rad = np.radians(slope + 0.1)
        # Assuming uniform contributing area for simplicity
        features['twi'] = np.log(1000 / np.tan(slope_rad))
        
        # Sink depth - depression detection
        # Fill DEM and compute difference
        filled = maximum_filter(dem_smooth, size=5)
        features['sink_depth'] = np.clip(filled - dem_smooth, 0, None)
        
        # Surface roughness
        features['roughness'] = generic_filter(
            dem_smooth, np.std, size=5, mode='reflect'
        )
        
        return features
    
    def normalize_features(
        self,
        features: Dict[str, np.ndarray],
        fit: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Normalize features to [0, 1] range
        
        Args:
            features: Dictionary of feature arrays
            fit: If True, compute and store statistics
        
        Returns:
            Normalized features
        """
        normalized = {}
        
        for name, data in features.items():
            if fit:
                # Compute statistics
                valid_mask = np.isfinite(data)
                if valid_mask.any():
                    self.stats[name] = {
                        'min': float(np.percentile(data[valid_mask], 2)),
                        'max': float(np.percentile(data[valid_mask], 98)),
                    }
                else:
                    self.stats[name] = {'min': 0.0, 'max': 1.0}
            
            # Apply normalization
            stats = self.stats.get(name, {'min': 0.0, 'max': 1.0})
            range_val = stats['max'] - stats['min']
            if range_val == 0:
                range_val = 1.0
            
            normalized[name] = np.clip(
                (data - stats['min']) / range_val,
                0, 1
            ).astype(np.float32)
        
        return normalized
    
    def create_feature_stack(
        self,
        spectral: Dict[str, np.ndarray],
        terrain: Dict[str, np.ndarray],
        geology: Dict[str, np.ndarray],
        hydrology: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create stacked feature array for ML input
        
        Returns:
            Tuple of (feature array, feature names)
        """
        all_features = {}
        all_features.update(spectral)
        all_features.update(terrain)
        all_features.update(geology)
        all_features.update(hydrology)
        
        # Normalize
        normalized = self.normalize_features(all_features)
        
        # Stack
        names = list(normalized.keys())
        stack = np.stack([normalized[n] for n in names], axis=-1)
        
        return stack, names


def prepare_training_data(
    features: np.ndarray,
    sinkhole_mask: np.ndarray,
    negative_sample_ratio: float = 5.0,
    buffer_size: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from features and sinkhole locations
    
    Args:
        features: Feature stack (H, W, C)
        sinkhole_mask: Binary mask of sinkhole locations
        negative_sample_ratio: Ratio of negative to positive samples
        buffer_size: Buffer around sinkholes to exclude from negatives
        random_state: Random seed
    
    Returns:
        Tuple of (X, y) training arrays
    """
    from scipy.ndimage import binary_dilation
    
    rng = np.random.RandomState(random_state)
    
    # Get positive samples
    positive_mask = sinkhole_mask > 0
    positive_indices = np.argwhere(positive_mask)
    
    # Create buffer around positives
    struct = np.ones((buffer_size * 2 + 1, buffer_size * 2 + 1))
    buffer_mask = binary_dilation(positive_mask, structure=struct)
    
    # Get negative sample candidates (outside buffer)
    negative_candidate_mask = ~buffer_mask & np.isfinite(features).all(axis=-1)
    negative_indices = np.argwhere(negative_candidate_mask)
    
    # Sample negatives
    n_positive = len(positive_indices)
    n_negative = int(n_positive * negative_sample_ratio)
    n_negative = min(n_negative, len(negative_indices))
    
    selected_neg_idx = rng.choice(len(negative_indices), size=n_negative, replace=False)
    selected_negatives = negative_indices[selected_neg_idx]
    
    # Extract features
    X_pos = features[positive_indices[:, 0], positive_indices[:, 1]]
    X_neg = features[selected_negatives[:, 0], selected_negatives[:, 1]]
    
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
    
    # Shuffle
    shuffle_idx = rng.permutation(len(y))
    
    return X[shuffle_idx], y[shuffle_idx]


def rasterize_points(
    points_gdf,
    bounds: Tuple[float, float, float, float],
    resolution: float = 10.0,
    buffer_m: float = 20.0,
) -> np.ndarray:
    """
    Rasterize point features to a binary mask
    
    Args:
        points_gdf: GeoDataFrame with point geometry
        bounds: (west, south, east, north)
        resolution: Pixel size in meters
        buffer_m: Buffer radius around points in meters
    
    Returns:
        Binary mask array
    """
    from rasterio.transform import from_bounds
    from rasterio.features import rasterize
    from shapely.geometry import Point
    
    west, south, east, north = bounds
    
    # Calculate dimensions
    width = int((east - west) * 111000 / resolution)
    height = int((north - south) * 111000 / resolution)
    
    transform = from_bounds(west, south, east, north, width, height)
    
    # Buffer points (convert buffer from meters to degrees approximately)
    buffer_deg = buffer_m / 111000
    
    shapes = []
    for geom in points_gdf.geometry:
        if geom is not None:
            buffered = geom.buffer(buffer_deg)
            shapes.append((buffered, 1))
    
    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)
    
    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    
    return mask

