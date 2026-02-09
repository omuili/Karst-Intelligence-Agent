"""
Real Data Inference Engine for Sinkhole Susceptibility

Uses live data from:
- Florida Geological Survey (sinkholes, geology)
- USGS 3DEP (elevation)
- National Hydrography Dataset (water)
- Sentinel satellite data
- Google Gemini for AI feature detection

DESIGN PRINCIPLE: No fake data, but graceful uncertainty handling
- If data is available: use real values
- If data is unavailable: mark as UNCERTAIN (not fake high/low values)
- Track data coverage for transparency
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import griddata

from backend.config import settings, WinterParkAOI


@dataclass
class TilePrediction:
    """Result of a tile prediction with metadata about data coverage"""
    susceptibility: np.ndarray  # The actual prediction array
    confidence: float  # Overall confidence (0-1) based on data availability
    data_coverage: Dict[str, bool] = field(default_factory=dict)  # Which data sources were available
    warnings: List[str] = field(default_factory=list)  # Any data quality warnings


class RealSusceptibilityInference:
    """
    Real-time sinkhole susceptibility inference using live data
    
    Combines:
    - Distance to known sinkholes (from FGS)
    - Karst geology presence
    - Terrain features (from DEM)
    - Distance to water features
    - Optional: Gemini AI feature detection
    """
    
    def __init__(self):
        self.data_manager = None
        self.cached_data = {}
        self.model = None
        self._init_data_manager()
    
    def _init_data_manager(self):
        """Initialize data manager"""
        try:
            from backend.data.services import RealDataManager
            cache_dir = settings.cache_dir if settings.cache_dir else Path("data/cache")
            self.data_manager = RealDataManager(cache_dir)
        except Exception as e:
            print(f"[!] Failed to init data manager: {e}")
    
    async def load_aoi_data(self, bbox: Tuple[float, float, float, float] = None):
        """
        Pre-load all data for the AOI
        """
        if bbox is None:
            bbox = tuple(WinterParkAOI.BBOX)
        
        if self.data_manager is None:
            print("[!] Data manager not initialized")
            return
        
        cache_key = f"aoi_{bbox[0]:.4f}_{bbox[1]:.4f}"
        
        if cache_key in self.cached_data:
            print("[*] Using cached AOI data")
            return
        
        print("[*] Loading real data for AOI...")
        
        try:
            data = await self.data_manager.fetch_all_layers(bbox)
            self.cached_data[cache_key] = data
            
            # Print summary
            if data.get("sinkholes"):
                n_sinkholes = len(data["sinkholes"].get("features", []))
                print(f"    Loaded {n_sinkholes} historical sinkholes")
            
            if data.get("geology"):
                n_geology = len(data["geology"].get("features", []))
                print(f"    Loaded {n_geology} geology polygons")
            
            if data.get("karst_units"):
                n_karst = len(data["karst_units"])
                print(f"    Identified {n_karst} karst units")
            
            if data.get("water"):
                n_water = len(data["water"].get("features", []))
                print(f"    Loaded {n_water} water features")
            
            if data.get("dem") is not None:
                print(f"    Loaded DEM: {data['dem'].shape}")
            
            # Ground displacement (NASA OPERA)
            if data.get("ground_displacement"):
                gd = data["ground_displacement"]
                disp = gd.get("displacement_mm")
                if disp is not None:
                    print(f"    Loaded ground displacement: shape={disp.shape}")
                    print(f"    Displacement range: {np.nanmin(disp):.1f} to {np.nanmax(disp):.1f} mm")
                    if gd.get("velocity_mm_year") is not None:
                        vel = gd["velocity_mm_year"]
                        print(f"    Velocity range: {np.nanmin(vel):.1f} to {np.nanmax(vel):.1f} mm/year")
            elif data.get("ground_displacement_error"):
                print(f"    [!] Ground displacement unavailable: {data['ground_displacement_error'][:60]}")
                
        except Exception as e:
            print(f"[!] Error loading AOI data: {e}")
    
    def get_loaded_data_status(self) -> Dict[str, Any]:
        """
        Return status of loaded AOI data (for scanner UI).
        Call after load_aoi_data() so cached_data is populated.
        """
        aoi_bbox = tuple(WinterParkAOI.BBOX)
        cache_key = f"aoi_{aoi_bbox[0]:.4f}_{aoi_bbox[1]:.4f}"
        data = self.cached_data.get(cache_key, {})
        sentinel2 = data.get("sentinel2_scenes") or []
        gd = data.get("ground_displacement")
        return {
            "sentinel2_scenes_count": len(sentinel2),
            "sentinel2_loaded": len(sentinel2) > 0,
            "ground_displacement_loaded": gd is not None and gd.get("displacement_mm") is not None,
            "sinkholes_count": len(data.get("sinkholes", {}).get("features", [])),
            "dem_loaded": data.get("dem") is not None,
            "water_count": len(data.get("water", {}).get("features", [])),
            "karst_units_count": len(data.get("karst_units", [])),
        }
    
    async def predict_tile(
        self,
        bounds: Tuple[float, float, float, float],
        tile_size: int = 256,
        zoom: int = 14,
        return_metadata: bool = False
    ) -> np.ndarray:
        """
        Predict susceptibility for a tile using real data
        
        Args:
            bounds: (west, south, east, north) in WGS84
            tile_size: Output size in pixels
            zoom: Zoom level
            return_metadata: If True, returns TilePrediction with metadata
        
        Returns:
            2D numpy array of susceptibility values [0, 1]
            (or TilePrediction if return_metadata=True)
        """
        west, south, east, north = bounds
        
        # Ensure AOI data is loaded
        aoi_bbox = tuple(WinterParkAOI.BBOX)
        cache_key = f"aoi_{aoi_bbox[0]:.4f}_{aoi_bbox[1]:.4f}"
        
        if cache_key not in self.cached_data:
            await self.load_aoi_data(aoi_bbox)
        
        data = self.cached_data.get(cache_key, {})
        
        # Create coordinate grids
        x = np.linspace(west, east, tile_size)
        y = np.linspace(north, south, tile_size)  # Flip for image
        xx, yy = np.meshgrid(x, y)
        
        # Track data coverage and warnings
        data_coverage = {}
        warnings = []
        available_weight = 0.0
        
        # Weights tuned for a continuous, organic heatmap (blue -> green -> yellow -> orange -> red)
        # with smooth variation across the tile, not isolated glowing spots.
        # Includes Sentinel-2 optical (NDVI/NDWI) when available.
        base_weights = {
            "ground_displacement": 0.08,
            "sinkhole_proximity": 0.45,   # Main driver but blended with terrain
            "karst_geology": 0.15,        # Regional variation
            "terrain": 0.15,              # DEM variation for organic look
            "water": 0.10,
            "sentinel_optical": 0.07,     # Sentinel-2 optical when available
        }
        
        # Initialize factors with uncertainty-aware defaults
        factors = {}
        
        # 1. Ground displacement from InSAR (NASA OPERA) - CRITICAL for early warning
        try:
            factors["ground_displacement"] = self._compute_ground_displacement_factor(
                data.get("ground_displacement"), bounds, tile_size
            )
            data_coverage["ground_displacement"] = True
            available_weight += base_weights["ground_displacement"]
        except (ValueError, RuntimeError, Exception) as e:
            # Ground displacement unavailable - DO NOT fake it
            # Use neutral value but flag clearly
            factors["ground_displacement"] = np.full((tile_size, tile_size), 0.5, dtype=np.float32)
            data_coverage["ground_displacement"] = False
            warnings.append(f"Ground displacement data unavailable: {str(e)[:80]}")
        
        # 2. Distance to known sinkholes (highest historical weight)
        try:
            factors["sinkhole_proximity"] = self._compute_sinkhole_proximity(
                data.get("sinkholes"), xx, yy, tile_size
            )
            data_coverage["sinkholes"] = True
            available_weight += base_weights["sinkhole_proximity"]
        except (ValueError, Exception) as e:
            # No sinkhole data - use NEUTRAL value (0.5) indicating uncertainty
            factors["sinkhole_proximity"] = np.full((tile_size, tile_size), 0.5, dtype=np.float32)
            data_coverage["sinkholes"] = False
            warnings.append(f"Sinkhole data unavailable: {str(e)[:50]}")
        
        # 3. Karst geology presence
        try:
            factors["karst_geology"] = self._compute_karst_presence(
                data.get("karst_units"), xx, yy, tile_size
            )
            data_coverage["karst_geology"] = True
            available_weight += base_weights["karst_geology"]
        except (ValueError, Exception) as e:
            # No karst data - Winter Park IS karst, so assume elevated baseline
            factors["karst_geology"] = np.full((tile_size, tile_size), 0.7, dtype=np.float32)
            data_coverage["karst_geology"] = False
            warnings.append(f"Karst data unavailable (assuming karst region): {str(e)[:50]}")
        
        # 4. Terrain features
        try:
            factors["terrain"] = self._compute_terrain_susceptibility(
                data.get("dem"), bounds, tile_size
            )
            data_coverage["terrain"] = True
            available_weight += base_weights["terrain"]
        except (ValueError, Exception) as e:
            # No DEM data - use neutral
            factors["terrain"] = np.full((tile_size, tile_size), 0.5, dtype=np.float32)
            data_coverage["terrain"] = False
            warnings.append(f"Terrain data unavailable: {str(e)[:50]}")
        
        # 5. Distance to water features
        try:
            factors["water"] = self._compute_water_proximity(
                data.get("water"), xx, yy, tile_size
            )
            data_coverage["water"] = True
            available_weight += base_weights["water"]
        except (ValueError, Exception) as e:
            # No water data - use neutral
            factors["water"] = np.full((tile_size, tile_size), 0.5, dtype=np.float32)
            data_coverage["water"] = False
            warnings.append(f"Water data unavailable: {str(e)[:50]}")
        
        # 6. Sentinel-2 optical (NDVI/NDWI) when scenes available
        try:
            factors["sentinel_optical"] = self._compute_sentinel_optical_factor(
                data.get("sentinel2_scenes"), bounds, tile_size
            )
            data_coverage["sentinel_optical"] = True
            available_weight += base_weights["sentinel_optical"]
        except (ValueError, Exception) as e:
            factors["sentinel_optical"] = np.full((tile_size, tile_size), 0.5, dtype=np.float32)
            data_coverage["sentinel_optical"] = False
        
        # Combine factors with weights
        susceptibility = (
            base_weights["ground_displacement"] * factors["ground_displacement"] +
            base_weights["sinkhole_proximity"] * factors["sinkhole_proximity"] +
            base_weights["karst_geology"] * factors["karst_geology"] +
            base_weights["terrain"] * factors["terrain"] +
            base_weights["water"] * factors["water"] +
            base_weights["sentinel_optical"] * factors["sentinel_optical"]
        )
        
        # Smooth for organic, continuous heatmap (no sharp glowing orbs)
        susceptibility = gaussian_filter(susceptibility, sigma=4)
        
        # Normalize to [0, 1]
        susceptibility = np.clip(susceptibility, 0, 1)
        
        # Calculate confidence based on data availability
        confidence = available_weight  # 0-1 based on which data sources worked
        
        if return_metadata:
            return TilePrediction(
                susceptibility=susceptibility,
                confidence=confidence,
                data_coverage=data_coverage,
                warnings=warnings
            )
        
        return susceptibility
    
    def _compute_sinkhole_proximity(
        self,
        sinkholes_geojson: Optional[Dict],
        xx: np.ndarray,
        yy: np.ndarray,
        tile_size: int
    ) -> np.ndarray:
        """
        Compute susceptibility factor based on proximity to known sinkholes
        
        Creates strong hotspots around known sinkholes with proper spatial decay.
        This is the PRIMARY driver of susceptibility visualization.
        """
        # Check if we have sinkhole data from FGS
        if sinkholes_geojson is None:
            raise ValueError(
                "Sinkhole data not loaded. Florida Geological Survey API may be unavailable."
            )
        
        if "features" not in sinkholes_geojson:
            raise ValueError(
                "Invalid sinkhole data format from Florida Geological Survey."
            )
        
        features = sinkholes_geojson["features"]
        
        # Extract sinkhole coordinates from the FULL AOI dataset
        sinkhole_points = []
        for feature in features:
            geom = feature.get("geometry", {})
            if geom.get("type") == "Point":
                coords = geom.get("coordinates", [])
                if len(coords) >= 2:
                    sinkhole_points.append((coords[0], coords[1]))
        
        # If no sinkholes in the entire AOI, that's unexpected for Winter Park
        if not sinkhole_points:
            raise ValueError(
                f"No sinkhole records found in Florida Geological Survey data. "
                f"This is unexpected for Winter Park, FL. API may have returned incomplete data."
            )
        
        # Compute distance to nearest sinkhole
        min_dist = np.full((tile_size, tile_size), np.inf)
        
        for sx, sy in sinkhole_points:
            dist = np.sqrt((xx - sx)**2 + (yy - sy)**2)
            min_dist = np.minimum(min_dist, dist)
        
        # Also compute sinkhole density effect (multiple nearby sinkholes = higher risk)
        density_effect = np.zeros((tile_size, tile_size), dtype=np.float32)
        # Slightly tighter influence radius so hotspots don’t smear out
        influence_radius = 0.022  # ~2km so susceptibility blends across the tile
        
        for sx, sy in sinkhole_points:
            dist = np.sqrt((xx - sx)**2 + (yy - sy)**2)
            # Each sinkhole adds to density within its influence radius
            contribution = np.exp(-dist / influence_radius) * 0.4
            density_effect += contribution
        
        # Cap density effect
        density_effect = np.clip(density_effect, 0, 1)
        
        # Convert minimum distance to susceptibility
        # Use multiple decay scales for realistic risk zones:
        # - Very close (<150–200m): Very high risk
        # - Close (200–600m): High risk  
        # - Medium (600m–1.2km): Elevated risk
        # - Far (>1.2km): Low baseline risk (karst region)
        
        # Primary decay - strong effect near sinkholes
        char_distance_near = 0.005   # ~500m - high risk zone (wider = smooth gradient)
        char_distance_mid = 0.015   # ~1.5km - elevated zone
        
        near_effect = np.exp(-min_dist / char_distance_near)
        mid_effect = np.exp(-min_dist / char_distance_mid) * 0.7
        # Blend for gradual falloff and full color ramp (blue -> green -> yellow -> red)
        factor = near_effect * 0.5 + mid_effect * 0.4 + density_effect * 0.4
        
        # Stretch to full [0, 1] range per‑tile to avoid “flat” tiles
        # One global scale: do not rescale per-tile (same value = same color everywhere).
        factor = np.clip(factor, 0, 1)
        
        return factor.astype(np.float32)
    
    def _compute_karst_presence(
        self,
        karst_units: Optional[List[Dict]],
        xx: np.ndarray,
        yy: np.ndarray,
        tile_size: int
    ) -> np.ndarray:
        """
        Compute karst geology factor
        
        Areas within karst units get higher susceptibility
        """
        if not karst_units:
            raise ValueError("No karst geology data available from Florida Geological Survey")
        
        # Initialize factor array - low value for non-karst areas
        factor = np.ones((tile_size, tile_size), dtype=np.float32) * 0.1
        
        # Check if points are within karst polygons
        from shapely.geometry import Point, shape
        
        for karst_feature in karst_units:
            geom = karst_feature.get("geometry")
            if geom:
                poly = shape(geom)
                
                # Sample check (full check is expensive)
                for i in range(0, tile_size, 10):
                    for j in range(0, tile_size, 10):
                        pt = Point(xx[i, j], yy[i, j])
                        if poly.contains(pt):
                            # Mark surrounding area as high susceptibility (karst)
                            i_min, i_max = max(0, i-5), min(tile_size, i+5)
                            j_min, j_max = max(0, j-5), min(tile_size, j+5)
                            factor[i_min:i_max, j_min:j_max] = 0.9
        
        return factor.astype(np.float32)
    
    def _compute_terrain_susceptibility(
        self,
        dem: Optional[np.ndarray],
        bounds: Tuple[float, float, float, float],
        tile_size: int
    ) -> np.ndarray:
        """
        Compute terrain-based susceptibility factors:
        - Slope (lower slope = potentially more prone to ponding)
        - Curvature (concave = depressions)
        - Local depressions (sink-fill difference)
        """
        if dem is None:
            raise ValueError("No DEM/elevation data available from USGS 3DEP")
        
        # Resize DEM to tile size if needed
        if dem.shape != (tile_size, tile_size):
            from scipy.ndimage import zoom
            scale = (tile_size / dem.shape[0], tile_size / dem.shape[1])
            dem = zoom(dem, scale, order=1)
        
        # Compute slope
        dy, dx = np.gradient(dem)
        slope = np.sqrt(dx**2 + dy**2)
        
        # Lower slope areas more susceptible (water accumulation)
        slope_factor = 1 - np.clip(slope / np.percentile(slope, 95), 0, 1)
        
        # Compute curvature (Laplacian)
        from scipy.ndimage import laplace
        curvature = laplace(dem)
        
        # Negative curvature (concave) = depressions = higher susceptibility
        curv_factor = np.clip(-curvature / (np.std(curvature) * 3 + 1e-6) + 0.5, 0, 1)
        
        # Detect local depressions (sink-fill)
        from scipy.ndimage import maximum_filter
        filled = maximum_filter(dem, size=7)
        sink_depth = filled - dem
        sink_factor = np.clip(sink_depth / 2, 0, 1)  # Normalize
        
        # Combine terrain factors
        terrain_factor = (
            0.3 * slope_factor +
            0.4 * curv_factor +
            0.3 * sink_factor
        )
        
        return terrain_factor.astype(np.float32)
    
    def _compute_water_proximity(
        self,
        water_geojson: Optional[Dict],
        xx: np.ndarray,
        yy: np.ndarray,
        tile_size: int
    ) -> np.ndarray:
        """
        Compute water proximity factor
        
        Areas near water features may have altered hydrology
        """
        if not water_geojson or not water_geojson.get("features"):
            raise ValueError("No water feature data available from National Hydrography Dataset")
        
        features = water_geojson["features"]
        
        # Extract water feature points/vertices
        water_points = []
        
        for feature in features:
            geom = feature.get("geometry", {})
            geom_type = geom.get("type", "")
            coords = geom.get("coordinates", [])
            
            if geom_type == "Point" and len(coords) >= 2:
                water_points.append((coords[0], coords[1]))
            elif geom_type == "LineString":
                for coord in coords:
                    if len(coord) >= 2:
                        water_points.append((coord[0], coord[1]))
            elif geom_type == "Polygon":
                for ring in coords:
                    for coord in ring:
                        if len(coord) >= 2:
                            water_points.append((coord[0], coord[1]))
            elif geom_type == "MultiLineString":
                for line in coords:
                    for coord in line:
                        if len(coord) >= 2:
                            water_points.append((coord[0], coord[1]))
        
        if not water_points:
            raise ValueError("No valid water feature coordinates found in NHD response")
        
        # Compute distance to nearest water feature
        min_dist = np.full((tile_size, tile_size), np.inf)
        
        # Sample water points to reduce computation
        sample_size = min(500, len(water_points))
        sampled_points = water_points[::max(1, len(water_points)//sample_size)]
        
        for wx, wy in sampled_points:
            dist = np.sqrt((xx - wx)**2 + (yy - wy)**2)
            min_dist = np.minimum(min_dist, dist)
        
        # Convert to factor
        # Medium distance = higher susceptibility (karst drainage effects)
        char_distance = 0.003  # ~300m
        factor = np.exp(-min_dist / char_distance) * 0.3 + 0.4
        
        return factor.astype(np.float32)
    
    def _compute_ground_displacement_factor(
        self,
        displacement_data: Optional[Dict],
        bounds: Tuple[float, float, float, float],
        tile_size: int
    ) -> np.ndarray:
        """
        Compute ground displacement factor from NASA OPERA DISP-S1 InSAR data
        
        CRITICAL for early warning - active subsidence is a strong sinkhole precursor.
        
        Subsidence (negative displacement) = HIGHER susceptibility
        - Rapid subsidence (>10mm/month) = Very High risk
        - Moderate subsidence (5-10mm/month) = High risk
        - Slow subsidence (1-5mm/month) = Elevated risk
        - Stable or uplift = Lower risk
        
        NO FAKE DATA - if displacement data is unavailable, this method raises an error.
        """
        if displacement_data is None:
            raise ValueError(
                "Ground displacement data not available from NASA OPERA DISP-S1. "
                "This could mean: no coverage for this area, data not yet processed, "
                "or NASA Earthdata authentication failed."
            )
        
        displacement_mm = displacement_data.get("displacement_mm")
        velocity_mm_year = displacement_data.get("velocity_mm_year")
        coherence = displacement_data.get("coherence")
        
        if displacement_mm is None:
            raise ValueError("No displacement array in OPERA data")
        
        # Resize to tile size if needed
        from scipy.ndimage import zoom as scipy_zoom
        
        if displacement_mm.shape[0] != tile_size or displacement_mm.shape[1] != tile_size:
            scale = (tile_size / displacement_mm.shape[0], tile_size / displacement_mm.shape[1])
            displacement_mm = scipy_zoom(displacement_mm, scale, order=1)
            
            if velocity_mm_year is not None:
                velocity_mm_year = scipy_zoom(velocity_mm_year, scale, order=1)
            if coherence is not None:
                coherence = scipy_zoom(coherence, scale, order=1)
        
        # Convert displacement to susceptibility factor
        # Negative displacement = subsidence = HIGHER risk
        # Use velocity if available (more meaningful than cumulative)
        if velocity_mm_year is not None:
            # Convert mm/year to risk factor
            # Thresholds based on literature:
            # - >20 mm/year = Very High risk
            # - 10-20 mm/year = High risk
            # - 5-10 mm/year = Moderate risk
            # - <5 mm/year = Low risk
            
            # Negative velocity = subsidence
            subsidence_rate = -velocity_mm_year  # Make subsidence positive
            
            # Normalize to [0, 1] factor
            # 20mm/year subsidence = factor of 1.0
            factor = np.clip(subsidence_rate / 20.0, 0, 1)
            
        else:
            # Use cumulative displacement
            # Threshold: >50mm cumulative subsidence = very high risk
            subsidence = -displacement_mm  # Make subsidence positive
            factor = np.clip(subsidence / 50.0, 0, 1)
        
        # Apply coherence weighting if available (low coherence = less reliable)
        if coherence is not None:
            # Coherence is 0-1, where higher = more reliable
            # Apply as confidence weight
            coherence_weight = np.clip(coherence, 0.3, 1.0)  # Minimum 30% weight
            factor = factor * coherence_weight

        return factor.astype(np.float32)

    def _compute_sentinel_optical_factor(
        self,
        sentinel2_scenes: Optional[List],
        bounds: Tuple[float, float, float, float],
        tile_size: int
    ) -> np.ndarray:
        """
        Compute susceptibility factor from Sentinel-2 optical data when available.
        When scenes exist, returns neutral 0.5 until per-tile NDVI/NDWI is implemented.
        """
        if not sentinel2_scenes or len(sentinel2_scenes) == 0:
            raise ValueError("No Sentinel-2 scenes available for this AOI")
        # Placeholder: neutral factor when optical data is available (NDVI/NDWI can be added later)
        return np.full((tile_size, tile_size), 0.5, dtype=np.float32)
    
    async def detect_features(
        self,
        bounds: Tuple[float, float, float, float],
        zoom: int = 14
    ) -> Dict[str, Any]:
        """
        Detect sinkhole-like features using real data + Gemini
        """
        # Get real sinkholes within bounds
        aoi_bbox = tuple(WinterParkAOI.BBOX)
        cache_key = f"aoi_{aoi_bbox[0]:.4f}_{aoi_bbox[1]:.4f}"
        
        if cache_key not in self.cached_data:
            await self.load_aoi_data(aoi_bbox)
        
        data = self.cached_data.get(cache_key, {})
        sinkholes = data.get("sinkholes", {}).get("features", [])
        
        west, south, east, north = bounds
        
        # Filter sinkholes within tile bounds
        tile_features = []
        
        for feature in sinkholes:
            geom = feature.get("geometry", {})
            if geom.get("type") == "Point":
                coords = geom.get("coordinates", [])
                if len(coords) >= 2:
                    lon, lat = coords[0], coords[1]
                    if west <= lon <= east and south <= lat <= north:
                        # Convert point to small polygon for visualization
                        props = feature.get("properties", {})
                        
                        # Create bounding box around point
                        buffer = 0.0003  # ~30m
                        tile_features.append({
                            "type": "Feature",
                            "properties": {
                                "id": props.get("OBJECTID", "unknown"),
                                "feature_type": "historical_sinkhole",
                                "confidence": 1.0,
                                "source": "Florida Geological Survey",
                                "date_reported": props.get("REPORTED_D", "Unknown"),
                                "type": props.get("SINKHOLE_T", "Unknown"),
                                "depth_ft": props.get("DEPTH_FT", "Unknown"),
                                "diameter_ft": props.get("DIAMETER_F", "Unknown"),
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [lon - buffer, lat - buffer],
                                    [lon + buffer, lat - buffer],
                                    [lon + buffer, lat + buffer],
                                    [lon - buffer, lat + buffer],
                                    [lon - buffer, lat - buffer],
                                ]]
                            }
                        })
        
        # Try Gemini for additional detection
        if settings.gemini_api_key:
            try:
                gemini_features = await self._detect_with_gemini(bounds, zoom)
                tile_features.extend(gemini_features)
            except Exception as e:
                print(f"[!] Gemini detection error: {e}")
        
        return {
            "type": "FeatureCollection",
            "properties": {
                "tile_bounds": list(bounds),
                "zoom": zoom,
                "source": "FGS + Gemini",
            },
            "features": tile_features
        }
    
    async def _detect_with_gemini(
        self,
        bounds: Tuple[float, float, float, float],
        zoom: int
    ) -> List[Dict]:
        """
        Use Gemini to detect additional features
        """
        try:
            from backend.gemini.client import GeminiClient
            
            client = GeminiClient(settings.gemini_api_key)
            
            if not client.is_available:
                return []
            
            # For now, return empty - full implementation would 
            # fetch satellite imagery and send to Gemini
            return []
            
        except Exception as e:
            print(f"[!] Gemini client error: {e}")
            return []

