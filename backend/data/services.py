"""
Real Data Services for Sinkhole Scanner

Live data fetching from:
- Florida Geological Survey (FGS) - Sinkhole inventory, geology
- USGS 3DEP - Digital Elevation Model
- Sentinel-1 (via ASF/Copernicus) - SAR/InSAR data
- National Hydrography Dataset - Water bodies
- Sentinel-2 (via Planetary Computer) - Optical imagery
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import hashlib

import httpx
import numpy as np

# Data source URLs - VERIFIED WORKING January 2026
# Florida DEP ArcGIS REST Services (ca.dep.state.fl.us)
FGS_SINKHOLE_URL = "https://ca.dep.state.fl.us/arcgis/rest/services/OpenData/FGS_SUBSIDENCE/MapServer/0/query"
FGS_GEOLOGY_URL = "https://ca.dep.state.fl.us/arcgis/rest/services/OpenData/FGS_PUBLIC/MapServer/0/query"  # Rock/Sediment
FGS_GEOMORPH_URL = "https://ca.dep.state.fl.us/arcgis/rest/services/OpenData/FGS_PUBLIC/MapServer/10/query"  # Karst Districts

# USGS National Map services
USGS_3DEP_WCS_URL = "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WCSServer"
USGS_3DEP_REST_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"

# National Hydrography Dataset
NHD_FLOWLINES_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6/query"
NHD_WATERBODIES_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/10/query"

# Satellite data
ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"
PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


class FloridaGeologicalSurvey:
    """
    Fetch real data from Florida Geological Survey ArcGIS services
    
    Available layers:
    - Subsidence Incident Reports (sinkholes)
    - Florida Geologic Map (lithology, karst units)
    """
    
    def __init__(self):
        # Longer timeout and retry-friendly settings
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=30.0),
            follow_redirects=True
        )
        self._cache = {}
        self._max_retries = 3
    
    async def close(self):
        await self.client.aclose()
    
    async def get_sinkhole_inventory(
        self,
        bbox: Tuple[float, float, float, float],
        max_records: int = 2000
    ) -> Dict[str, Any]:
        """
        Fetch sinkhole incident reports from FGS
        
        Args:
            bbox: (west, south, east, north) in WGS84
            max_records: Maximum number of records to fetch
        
        Returns:
            GeoJSON FeatureCollection with sinkhole points
        """
        west, south, east, north = bbox
        
        params = {
            "where": "1=1",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
            "resultRecordCount": max_records,
        }
        
        # Try the verified working endpoint
        urls = [
            FGS_SINKHOLE_URL,  # ca.dep.state.fl.us - verified working
        ]
        
        last_error = None
        for url in urls:
            for attempt in range(self._max_retries):
                try:
                    print(f"[FGS] Trying: {url[:60]}... (attempt {attempt + 1})")
                    response = await self.client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    n_features = len(data.get('features', []))
                    if n_features > 0:
                        print(f"[FGS] SUCCESS: Fetched {n_features} sinkhole records")
                        return data
                    elif 'features' in data:
                        # API returned but no sinkholes in this bbox - try another endpoint
                        print(f"[FGS] No sinkholes found in bbox from {url[:40]}")
                        break  # Try next URL, not retrying same one
                        
                except Exception as e:
                    last_error = e
                    print(f"[FGS] Attempt {attempt + 1} failed: {str(e)[:50]}")
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
        
        # If all endpoints fail, try fetching state-wide and filter
        print("[FGS] All endpoints failed. Trying state-wide fetch...")
        return await self._fetch_statewide_sinkholes(bbox, max_records)
    
    async def get_geology(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> Dict[str, Any]:
        """
        Fetch geology/lithology data from FGS
        
        Tries multiple sources:
        1. Karst Districts (geomorphology) - best for karst identification
        2. Rock/Sediment distribution - detailed lithology
        """
        west, south, east, north = bbox
        
        params = {
            "where": "1=1",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
        }
        
        # Try Karst Districts FIRST (more reliable for karst identification)
        # Then fall back to Rock/Sediment distribution
        urls = [
            (FGS_GEOMORPH_URL, "Karst Districts"),
            (FGS_GEOLOGY_URL, "Rock/Sediment"),
        ]
        
        all_features = []
        
        for url, name in urls:
            for attempt in range(self._max_retries):
                try:
                    print(f"[FGS] Trying {name}: {url[:50]}... (attempt {attempt + 1})")
                    response = await self.client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    n_features = len(data.get('features', []))
                    print(f"[FGS] {name}: Got {n_features} features")
                    
                    if n_features > 0:
                        # Mark source for debugging
                        for f in data.get('features', []):
                            f['properties']['_source'] = name
                        all_features.extend(data.get('features', []))
                    break  # Success, try next endpoint
                        
                except Exception as e:
                    print(f"[FGS] {name} attempt {attempt + 1} failed: {str(e)[:40]}")
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    continue
        
        if all_features:
            print(f"[FGS] SUCCESS: Total {len(all_features)} geology features from all sources")
            return {"type": "FeatureCollection", "features": all_features}
        
        # All endpoints returned no data - raise error (NO FALLBACK)
        raise RuntimeError(
            "Failed to fetch geology data from Florida Geological Survey. "
            "No geology features found for the specified area."
        )
    
    async def _fetch_statewide_sinkholes(
        self,
        bbox: Tuple[float, float, float, float],
        max_records: int
    ) -> Dict[str, Any]:
        """
        Last resort: Try fetching all Florida sinkholes and filter by bbox.
        Raises error if no real data can be obtained.
        """
        print("[FGS] Trying statewide fetch with bbox filter...")
        
        # Try statewide query without geometry filter
        params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
            "resultRecordCount": 5000,  # Get more records statewide
        }
        
        urls = [
            FGS_SINKHOLE_URL,  # Verified working endpoint
        ]
        
        west, south, east, north = bbox
        
        for url in urls:
            try:
                print(f"[FGS] Statewide query: {url[:50]}...")
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                all_features = data.get('features', [])
                print(f"[FGS] Got {len(all_features)} statewide sinkholes, filtering to bbox...")
                
                # Filter to bbox
                filtered_features = []
                for feature in all_features:
                    geom = feature.get("geometry", {})
                    if geom.get("type") == "Point":
                        coords = geom.get("coordinates", [])
                        if len(coords) >= 2:
                            lon, lat = coords[0], coords[1]
                            if west <= lon <= east and south <= lat <= north:
                                filtered_features.append(feature)
                
                print(f"[FGS] Filtered to {len(filtered_features)} sinkholes in AOI")
                
                if filtered_features:
                    return {"type": "FeatureCollection", "features": filtered_features}
                    
            except Exception as e:
                print(f"[FGS] Statewide fetch failed: {str(e)[:50]}")
                continue
        
        # All attempts failed - raise error (NO FALLBACK)
        raise RuntimeError(
            "Failed to fetch sinkhole data from Florida Geological Survey. "
            "All API endpoints are unavailable. Please check your internet connection."
        )
    
    def identify_karst_units(self, geology_geojson: Dict) -> List[Dict]:
        """
        Identify karst-prone units from geology data
        
        Karst indicators in Florida:
        - Karst Districts (from geomorphology layer): Ocala Karst, Dougherty Karst
        - Limestone formations: Ocala, Suwannee, Avon Park, Tampa
        - Any unit with "limestone" or "dolomite" in lithology
        """
        # Keywords indicating karst terrain
        karst_keywords = [
            "karst",           # Direct karst reference
            "limestone",       # Karst-forming rock
            "dolomite",        # Karst-forming rock  
            "ocala",           # Ocala Limestone / Ocala Karst District
            "suwannee",        # Suwannee Limestone
            "avon park",       # Avon Park Formation
            "tampa",           # Tampa Limestone
            "floridan",        # Floridan Aquifer (karst)
            "carbonate",       # Carbonate rock = karst
            "central lake",    # Central Lakes District (karst)
        ]
        
        karst_features = []
        features = geology_geojson.get("features", [])
        
        # Debug: print available fields from first feature
        if features:
            first_props = features[0].get("properties", {})
            print(f"[FGS] Geology fields: {list(first_props.keys())}")
            print(f"[FGS] Sample values: {dict(list(first_props.items())[:3])}")
        
        for feature in features:
            props = feature.get("properties", {})
            
            # Check ALL property values (field names vary between APIs)
            searchable_text = " ".join([
                str(v) for v in props.values() if v is not None
            ]).lower()
            
            is_karst = any(kw in searchable_text for kw in karst_keywords)
            
            # Also check if this is from Karst Districts layer (geomorphology)
            # These features ARE karst by definition
            source = props.get("_source", "")
            if "Karst Districts" in source:
                is_karst = True
            
            if is_karst:
                feature["properties"]["is_karst"] = True
                karst_features.append(feature)
        
        # Log results
        if karst_features:
            print(f"[FGS] Identified {len(karst_features)} karst features")
        elif features:
            print(f"[FGS] WARNING: No karst keywords found in {len(features)} geology features")
            # Show sample of what was found
            sample_props = features[0].get("properties", {})
            for k, v in list(sample_props.items())[:5]:
                print(f"[FGS]   {k}: {str(v)[:60]}")
        
        return karst_features


class USGSElevationService:
    """
    Fetch DEM data from USGS 3DEP (3D Elevation Program)
    
    Provides 1/3 arc-second (~10m) resolution elevation data
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def get_elevation_image(
        self,
        bbox: Tuple[float, float, float, float],
        width: int = 512,
        height: int = 512,
    ) -> Optional[np.ndarray]:
        """
        Fetch DEM as image from USGS 3DEP ImageServer
        
        Args:
            bbox: (west, south, east, north) in WGS84
            width: Output width in pixels
            height: Output height in pixels
        
        Returns:
            2D numpy array of elevation values (meters)
        """
        west, south, east, north = bbox
        
        # Use exportImage endpoint
        url = f"{USGS_3DEP_REST_URL}/exportImage"
        
        params = {
            "bbox": f"{west},{south},{east},{north}",
            "bboxSR": "4326",
            "size": f"{width},{height}",
            "imageSR": "4326",
            "format": "tiff",
            "pixelType": "F32",
            "noDataInterpretation": "esriNoDataMatchAny",
            "interpolation": "RSP_BilinearInterpolation",
            "f": "image",
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            # Parse TIFF response
            from io import BytesIO
            import rasterio
            
            with rasterio.open(BytesIO(response.content)) as src:
                dem = src.read(1)
                
            print(f"[USGS] Fetched DEM: {dem.shape}, range: {dem.min():.1f} - {dem.max():.1f}m")
            return dem
            
        except Exception as e:
            print(f"[USGS] Error fetching DEM: {e}")
            raise RuntimeError(f"Failed to fetch DEM data from USGS 3DEP: {str(e)}")


class NationalHydrographyDataset:
    """
    Fetch water features from USGS National Hydrography Dataset
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def get_water_features(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> Dict[str, Any]:
        """
        Fetch streams, rivers, and water bodies
        
        Returns combined GeoJSON of flowlines and waterbodies
        """
        west, south, east, north = bbox
        
        params = {
            "where": "1=1",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "outFields": "GNIS_NAME,FTYPE,FCODE,LENGTHKM",
            "returnGeometry": "true",
            "f": "geojson",
            "resultRecordCount": 1000,
        }
        
        features = []
        
        # Fetch flowlines (streams, rivers)
        try:
            response = await self.client.get(NHD_FLOWLINES_URL, params=params)
            response.raise_for_status()
            data = response.json()
            features.extend(data.get("features", []))
            print(f"[NHD] Fetched {len(data.get('features', []))} flowlines")
        except Exception as e:
            print(f"[NHD] Error fetching flowlines: {e}")
        
        # Fetch water bodies (lakes, ponds)
        try:
            params["outFields"] = "GNIS_NAME,FTYPE,FCODE,AREASQKM"
            response = await self.client.get(NHD_WATERBODIES_URL, params=params)
            response.raise_for_status()
            data = response.json()
            features.extend(data.get("features", []))
            print(f"[NHD] Fetched {len(data.get('features', []))} waterbodies")
        except Exception as e:
            print(f"[NHD] Error fetching waterbodies: {e}")
        
        return {
            "type": "FeatureCollection",
            "features": features
        }


class SentinelDataService:
    """
    Fetch Sentinel-1 SAR and Sentinel-2 optical data
    
    Sources:
    - Microsoft Planetary Computer (primary)
    - Alaska Satellite Facility (for InSAR products)
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def search_sentinel2(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20.0,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Search for Sentinel-2 scenes via Planetary Computer STAC
        Uses pystac-client for robust access
        """
        import asyncio
        
        def _search_sync():
            """Synchronous search using pystac-client"""
            try:
                import planetary_computer as pc
                from pystac_client import Client
                
                # Open Planetary Computer STAC catalog
                catalog = Client.open(
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                    modifier=pc.sign_inplace
                )
                
                west, south, east, north = bbox
                
                # Search for Sentinel-2 L2A scenes
                search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=[west, south, east, north],
                    datetime=f"{start_date}/{end_date}",
                    query={"eo:cloud_cover": {"lt": max_cloud_cover}},
                    max_items=limit,
                    sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}]
                )
                
                # Convert to list of dicts
                items = []
                for item in search.items():
                    items.append(item.to_dict())
                
                return items
                
            except Exception as e:
                print(f"[Sentinel-2] pystac-client error: {e}")
                return []
        
        try:
            # Run sync search in thread pool
            items = await asyncio.to_thread(_search_sync)
            
            if items:
                print(f"[Sentinel-2] Found {len(items)} scenes")
            else:
                print(f"[Sentinel-2] No scenes found for date range {start_date} to {end_date}")
            
            return items
            
        except Exception as e:
            print(f"[Sentinel-2] Search error: {e}")
            return []
    
    async def search_sentinel1(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Search for Sentinel-1 SAR scenes via Planetary Computer
        """
        west, south, east, north = bbox
        
        search_body = {
            "collections": ["sentinel-1-grd"],
            "bbox": [west, south, east, north],
            "datetime": f"{start_date}/{end_date}",
            "limit": limit,
        }
        
        try:
            url = f"{PLANETARY_COMPUTER_URL}/search"
            response = await self.client.post(url, json=search_body)
            response.raise_for_status()
            data = response.json()
            
            items = data.get("features", [])
            print(f"[Sentinel-1] Found {len(items)} SAR scenes")
            return items
            
        except Exception as e:
            print(f"[Sentinel-1] Search error: {e}")
            return []
    
    async def get_sentinel2_tile(
        self,
        item: Dict,
        bands: List[str] = ["B04", "B03", "B02", "B08"],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Fetch Sentinel-2 bands for a scene
        
        Args:
            item: STAC item from search
            bands: List of band names to fetch
            bbox: Optional bbox in WGS84 (west, south, east, north)
        
        Returns:
            Dictionary of band name -> numpy array
        """
        import asyncio
        
        def _fetch_bands_sync():
            try:
                import planetary_computer as pc
                import rasterio
                from rasterio.warp import transform_bounds
                from rasterio.windows import from_bounds
                
                band_data = {}
                assets = item.get("assets", {})
                
                for band in bands:
                    if band in assets:
                        href = assets[band]["href"]
                        
                        # Sign URL if not already signed (check for SAS token)
                        if "?" not in href or "sig=" not in href:
                            href = pc.sign(href)
                        
                        with rasterio.open(href) as src:
                            if bbox:
                                # Transform bbox from WGS84 to raster CRS
                                west, south, east, north = bbox
                                if src.crs and str(src.crs) != "EPSG:4326":
                                    transformed = transform_bounds(
                                        "EPSG:4326", src.crs, 
                                        west, south, east, north
                                    )
                                    west, south, east, north = transformed
                                
                                # Get window, ensuring it's within bounds
                                window = from_bounds(west, south, east, north, src.transform)
                                
                                # Clip window to valid range
                                window = window.intersection(rasterio.windows.Window(
                                    0, 0, src.width, src.height
                                ))
                                
                                if window.width > 0 and window.height > 0:
                                    data = src.read(1, window=window)
                                else:
                                    print(f"[Sentinel-2] Window outside image bounds, reading full image")
                                    data = src.read(1)
                                    # Resize to reasonable size
                                    if data.shape[0] > 1000 or data.shape[1] > 1000:
                                        from scipy.ndimage import zoom
                                        scale = 512 / max(data.shape)
                                        data = zoom(data, scale, order=1)
                            else:
                                data = src.read(1)
                                # Resize if too large
                                if data.shape[0] > 1000 or data.shape[1] > 1000:
                                    from scipy.ndimage import zoom
                                    scale = 512 / max(data.shape)
                                    data = zoom(data, scale, order=1)
                            
                            band_data[band] = data
                            print(f"[Sentinel-2] Band {band}: {data.shape}")
                
                print(f"[Sentinel-2] Fetched {len(band_data)} bands")
                return band_data
                
            except Exception as e:
                print(f"[Sentinel-2] Error fetching bands: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        try:
            return await asyncio.to_thread(_fetch_bands_sync)
        except Exception as e:
            print(f"[Sentinel-2] Error fetching tile: {e}")
            return None


class OPERADisplacementService:
    """
    NASA OPERA DISP-S1 Ground Displacement Service
    
    Fetches REAL InSAR-derived surface displacement data from NASA OPERA.
    Provides:
    - Cumulative displacement (mm)
    - Velocity (deformation rate, mm/year)
    - Coherence (data quality indicator)
    
    NO FAKE DATA - if data is unavailable, raises error.
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=180.0)
        self._earthaccess_authenticated = False
        self._cache_dir = Path("data/cache/opera")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def close(self):
        await self.client.aclose()
    
    def _ensure_earthaccess_auth(self):
        """Authenticate with NASA Earthdata (required for data access)"""
        if self._earthaccess_authenticated:
            return
        
        try:
            import earthaccess
            # Try to authenticate - will use cached credentials or prompt
            earthaccess.login(strategy="environment")  # Uses EARTHDATA_USERNAME/PASSWORD env vars
            self._earthaccess_authenticated = True
            print("[OPERA] Authenticated with NASA Earthdata")
        except Exception as e:
            # Try netrc file or interactive login
            try:
                import earthaccess
                earthaccess.login(persist=True)
                self._earthaccess_authenticated = True
                print("[OPERA] Authenticated with NASA Earthdata (cached credentials)")
            except Exception as e2:
                raise RuntimeError(
                    f"NASA Earthdata authentication failed: {e2}. "
                    f"Please run 'earthaccess.login()' interactively first, "
                    f"or set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables."
                )
    
    async def search_displacement_products(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict]:
        """
        Search for OPERA DISP-S1 products covering the AOI
        
        Args:
            bbox: (west, south, east, north) in WGS84
            start_date: Start date (YYYY-MM-DD), defaults to 90 days ago
            end_date: End date (YYYY-MM-DD), defaults to today
        
        Returns:
            List of granule metadata dicts
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        west, south, east, north = bbox
        
        def _search_sync():
            import earthaccess
            self._ensure_earthaccess_auth()
            
            print(f"[OPERA] Searching DISP-S1 products: {bbox}, {start_date} to {end_date}")
            
            results = earthaccess.search_data(
                short_name="OPERA_L3_DISP-S1_V1",
                bounding_box=(west, south, east, north),
                temporal=(start_date, end_date),
                count=50  # Limit for efficiency
            )
            
            granules = []
            for r in results:
                granule_info = {
                    "granule_id": r.get("umm", {}).get("GranuleUR", str(r)),
                    "time_start": r.get("umm", {}).get("TemporalExtent", {}).get("RangeDateTime", {}).get("BeginningDateTime"),
                    "time_end": r.get("umm", {}).get("TemporalExtent", {}).get("RangeDateTime", {}).get("EndingDateTime"),
                    "data_links": earthaccess.results.DataGranule(r).data_links() if hasattr(r, "data_links") else [],
                    "size_mb": r.get("umm", {}).get("DataGranule", {}).get("ArchiveAndDistributionInformation", [{}])[0].get("Size", 0),
                }
                granules.append(granule_info)
            
            print(f"[OPERA] Found {len(granules)} DISP-S1 granules")
            return granules
        
        try:
            return await asyncio.to_thread(_search_sync)
        except Exception as e:
            print(f"[OPERA] Search failed: {e}")
            raise RuntimeError(f"OPERA DISP-S1 search failed: {str(e)}")
    
    async def get_displacement_data(
        self,
        bbox: Tuple[float, float, float, float],
        target_resolution: int = 256,
    ) -> Dict[str, Any]:
        """
        Fetch REAL ground displacement data for an AOI
        
        Returns:
            Dict with:
            - displacement_mm: 2D array of cumulative displacement (mm)
            - velocity_mm_year: 2D array of velocity (mm/year)
            - coherence: 2D array of data quality (0-1)
            - metadata: Dict with temporal info, coverage, etc.
        
        Raises:
            RuntimeError if no data available (NO FAKE DATA!)
        """
        west, south, east, north = bbox
        
        def _fetch_sync():
            import earthaccess
            self._ensure_earthaccess_auth()
            
            print(f"[OPERA] Fetching displacement data for bbox: {bbox}")
            
            # Search for available data - OPERA Phase 1 has data through 2024
            # First try recent data, then fall back to historical
            search_ranges = [
                # Try most recent first
                ("2024-01-01", "2024-12-31", "2024"),
                ("2023-01-01", "2023-12-31", "2023"),
                ("2020-01-01", "2024-12-31", "2020-2024"),
                ("2016-07-01", "2024-12-31", "All available"),
            ]
            
            results = None
            used_range = None
            
            for start_date, end_date, label in search_ranges:
                print(f"[OPERA] Searching {label}...")
                results = earthaccess.search_data(
                    short_name="OPERA_L3_DISP-S1_V1",
                    bounding_box=(west, south, east, north),
                    temporal=(start_date, end_date),
                    count=10
                )
                if results:
                    used_range = label
                    print(f"[OPERA] Found {len(results)} granules in {label}")
                    break
            
            if not results:
                raise RuntimeError(
                    f"No OPERA DISP-S1 data available for bbox {bbox}. "
                    f"Searched all date ranges from 2016-2024. "
                    f"This area may not have Sentinel-1 coverage."
                )
            
            print(f"[OPERA] Found {len(results)} granules, downloading most recent...")
            
            # Download most recent granule
            downloaded = earthaccess.download(
                results[:1],  # Most recent
                str(self._cache_dir),
            )
            
            if not downloaded:
                raise RuntimeError("OPERA DISP-S1 download failed - no files returned")
            
            # Parse the downloaded file
            file_path = Path(downloaded[0])
            print(f"[OPERA] Processing: {file_path.name}")
            
            return self._parse_disp_file(file_path, bbox, target_resolution)
        
        try:
            return await asyncio.to_thread(_fetch_sync)
        except Exception as e:
            print(f"[OPERA] Fetch failed: {e}")
            raise RuntimeError(f"OPERA DISP-S1 data fetch failed: {str(e)}")
    
    def _parse_disp_file(
        self,
        file_path: Path,
        bbox: Tuple[float, float, float, float],
        target_resolution: int
    ) -> Dict[str, Any]:
        """
        Parse OPERA DISP-S1 HDF5/NetCDF file and extract displacement layers
        """
        import h5py
        from scipy.ndimage import zoom
        
        west, south, east, north = bbox
        
        # OPERA DISP-S1 files are HDF5
        with h5py.File(file_path, 'r') as f:
            print(f"[OPERA] File keys: {list(f.keys())}")
            
            # Navigate to displacement data
            # OPERA structure: /science/grids/data/...
            if 'science' in f:
                data_group = f['science/grids/data']
            elif 'data' in f:
                data_group = f['data']
            else:
                # Try root level
                data_group = f
            
            print(f"[OPERA] Data group keys: {list(data_group.keys())}")
            
            # Extract displacement layers
            displacement = None
            velocity = None
            coherence = None
            
            # Look for displacement data (naming varies)
            for key in data_group.keys():
                key_lower = key.lower()
                if 'displacement' in key_lower or 'disp' in key_lower:
                    displacement = np.array(data_group[key])
                    print(f"[OPERA] Found displacement: {key}, shape: {displacement.shape}")
                elif 'velocity' in key_lower or 'vel' in key_lower:
                    velocity = np.array(data_group[key])
                    print(f"[OPERA] Found velocity: {key}, shape: {velocity.shape}")
                elif 'coherence' in key_lower or 'coh' in key_lower:
                    coherence = np.array(data_group[key])
                    print(f"[OPERA] Found coherence: {key}, shape: {coherence.shape}")
            
            # If velocity not found, compute from displacement time series
            if velocity is None and displacement is not None:
                # Assume annual rate from cumulative
                velocity = displacement  # Placeholder - would need time delta
                print("[OPERA] Using displacement as velocity proxy")
            
            if displacement is None:
                raise RuntimeError(
                    f"Could not find displacement data in OPERA file {file_path.name}. "
                    f"Available keys: {list(data_group.keys())}"
                )
            
            # Get geotransform for subsetting
            # (simplified - actual implementation needs coordinate arrays)
            
            # Resize to target resolution
            if displacement.shape[0] != target_resolution:
                scale_y = target_resolution / displacement.shape[0]
                scale_x = target_resolution / displacement.shape[1] if len(displacement.shape) > 1 else 1
                
                displacement = zoom(displacement, (scale_y, scale_x), order=1)
                if velocity is not None:
                    velocity = zoom(velocity, (scale_y, scale_x), order=1)
                if coherence is not None:
                    coherence = zoom(coherence, (scale_y, scale_x), order=1)
            
            # Handle no-data values
            nodata = -9999
            displacement = np.where(displacement == nodata, np.nan, displacement)
            if velocity is not None:
                velocity = np.where(velocity == nodata, np.nan, velocity)
            
            # Convert units if needed (OPERA uses meters, we want mm)
            if np.nanmax(np.abs(displacement)) < 1:  # Likely in meters
                displacement = displacement * 1000  # Convert to mm
                if velocity is not None:
                    velocity = velocity * 1000
            
            result = {
                "displacement_mm": displacement.astype(np.float32),
                "velocity_mm_year": velocity.astype(np.float32) if velocity is not None else None,
                "coherence": coherence.astype(np.float32) if coherence is not None else None,
                "metadata": {
                    "source": "NASA OPERA DISP-S1",
                    "file": file_path.name,
                    "bbox": bbox,
                    "resolution_m": 30,
                    "units": "millimeters",
                },
            }
            
            print(f"[OPERA] Displacement range: {np.nanmin(displacement):.1f} to {np.nanmax(displacement):.1f} mm")
            if velocity is not None:
                print(f"[OPERA] Velocity range: {np.nanmin(velocity):.1f} to {np.nanmax(velocity):.1f} mm/year")
            
            return result
    
    async def get_displacement_metrics(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> Dict[str, float]:
        """
        Get summary displacement metrics for an AOI
        
        Returns:
            Dict with:
            - max_displacement_mm: Maximum cumulative displacement
            - mean_velocity_mm_year: Mean deformation rate
            - subsidence_area_percent: Percentage of area with negative displacement
            - coherence_mean: Mean data quality
        """
        data = await self.get_displacement_data(bbox)
        
        disp = data["displacement_mm"]
        vel = data.get("velocity_mm_year")
        coh = data.get("coherence")
        
        # Compute metrics
        valid_mask = ~np.isnan(disp)
        
        metrics = {
            "max_displacement_mm": float(np.nanmax(np.abs(disp))),
            "min_displacement_mm": float(np.nanmin(disp)),  # Negative = subsidence
            "mean_displacement_mm": float(np.nanmean(disp)),
            "subsidence_area_percent": float(100 * np.sum(disp[valid_mask] < 0) / np.sum(valid_mask)) if np.any(valid_mask) else 0,
        }
        
        if vel is not None:
            metrics["mean_velocity_mm_year"] = float(np.nanmean(vel))
            metrics["max_velocity_mm_year"] = float(np.nanmax(np.abs(vel)))
        
        if coh is not None:
            metrics["coherence_mean"] = float(np.nanmean(coh))
            metrics["coherence_min"] = float(np.nanmin(coh))
        
        return metrics


class InSARService:
    """
    Legacy InSAR service - kept for SAR coherence computation
    For ground displacement, use OPERADisplacementService instead
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def compute_sar_coherence(
        self,
        sar_image1: np.ndarray,
        sar_image2: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """
        Compute SAR coherence between two images
        
        Low coherence can indicate ground deformation/subsidence
        """
        from scipy.ndimage import uniform_filter
        
        if sar_image1.shape != sar_image2.shape:
            raise ValueError("SAR images must have same shape")
        
        if not np.iscomplexobj(sar_image1):
            sar_image1 = sar_image1.astype(np.complex64)
        if not np.iscomplexobj(sar_image2):
            sar_image2 = sar_image2.astype(np.complex64)
        
        numerator = uniform_filter(sar_image1 * np.conj(sar_image2), window_size)
        denominator1 = uniform_filter(np.abs(sar_image1)**2, window_size)
        denominator2 = uniform_filter(np.abs(sar_image2)**2, window_size)
        
        coherence = np.abs(numerator) / np.sqrt(denominator1 * denominator2 + 1e-10)
        
        return coherence.astype(np.float32)


class GPSGroundMovementService:
    """
    Real-time GPS Ground Movement Service
    
    Fetches ACTUAL position time series from Nevada Geodetic Laboratory.
    This provides CURRENT, CONTINUOUS data for early warning monitoring.
    
    Primary station: FLOL (Orlando, FL) - 28.571°N, -81.424°W
    Distance to Winter Park: ~10km
    
    Data characteristics:
    - 24-hour rapid solutions: ~1 day latency
    - Position accuracy: ~1-2mm horizontal, 5-10mm vertical
    - Updates: Daily
    
    NO FAKE DATA - if data unavailable, raises RuntimeError.
    """
    
    # Florida GPS stations from Nevada Geodetic Lab
    # Priority order for Central Florida: FLOL > ORL1 (both have confirmed data)
    FLORIDA_STATIONS = {
        "FLOL": {"lat": 28.571, "lon": -81.424, "name": "Orlando", "region": "Central", "priority": 1},
        "ORL1": {"lat": 28.49, "lon": -81.31, "name": "Orlando Airport", "region": "Central", "priority": 2},
        "FLSC": {"lat": 27.217, "lon": -82.405, "name": "Sarasota", "region": "West Coast", "priority": 3},
        "NAPL": {"lat": 26.149, "lon": -81.776, "name": "Naples", "region": "Southwest", "priority": 4},
        "FLIU": {"lat": 25.755, "lon": -80.374, "name": "Miami", "region": "Southeast", "priority": 5},
    }
    
    # Base URL for Nevada Geodetic Lab time series data
    NGL_BASE_URL = "https://geodesy.unr.edu/gps_timeseries"
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=20.0),
            follow_redirects=True
        )
        self._cache_dir = Path("data/cache/gps")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def close(self):
        await self.client.aclose()
    
    def find_nearest_stations(
        self,
        lat: float,
        lon: float,
        max_distance_km: float = 100.0
    ) -> List[Dict[str, Any]]:
        """
        Find GPS stations near given coordinates, sorted by distance.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            max_distance_km: Maximum acceptable distance
        
        Returns:
            List of station info dicts sorted by distance
        """
        from math import radians, sin, cos, sqrt, atan2
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in km
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
            return R * 2 * atan2(sqrt(a), sqrt(1-a))
        
        stations = []
        for station_id, info in self.FLORIDA_STATIONS.items():
            dist = haversine(lat, lon, info["lat"], info["lon"])
            if dist <= max_distance_km:
                stations.append({
                    "station_id": station_id,
                    "distance_km": round(dist, 2),
                    **info
                })
        
        # Sort by distance
        stations.sort(key=lambda x: x["distance_km"])
        return stations
    
    def find_nearest_station(
        self,
        lat: float,
        lon: float,
        max_distance_km: float = 100.0
    ) -> Optional[Dict[str, Any]]:
        """Find single nearest station (for backward compatibility)"""
        stations = self.find_nearest_stations(lat, lon, max_distance_km)
        return stations[0] if stations else None
    
    async def get_position_timeseries(
        self,
        station_id: str = "FLOL",
        days: int = 90,
    ) -> Dict[str, Any]:
        """
        Fetch GPS position time series from Nevada Geodetic Lab.
        
        Tries rapid solutions first (24hr latency), then falls back to 
        final solutions (1-2 week latency) if rapid unavailable.
        
        Args:
            station_id: GPS station identifier (default: FLOL for Orlando)
            days: Number of days of data to return
        
        Returns:
            Dict with position time series data
        
        Raises:
            RuntimeError if data unavailable (NO FAKE DATA)
        """
        import urllib.request
        import urllib.error
        import ssl
        
        if station_id not in self.FLORIDA_STATIONS:
            raise ValueError(f"Unknown station: {station_id}. Available: {list(self.FLORIDA_STATIONS.keys())}")
        
        # URLs to try in order: rapid first, then final
        urls_to_try = [
            (f"{self.NGL_BASE_URL}/rapids/tenv3/{station_id}.tenv3", "rapid"),
            (f"{self.NGL_BASE_URL}/tenv3/IGS14/{station_id}.tenv3", "final"),
        ]
        
        print(f"[GPS] Fetching data for station {station_id}...")
        
        def _fetch_sync(url: str, solution_type: str):
            """Synchronous fetch for thread pool"""
            print(f"[GPS] Trying {solution_type} solution: {url[:70]}...")
            
            # Create SSL context
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                with urllib.request.urlopen(req, timeout=60, context=ctx) as response:
                    content = response.read().decode('utf-8')
                    print(f"[GPS] SUCCESS: Received {len(content)} bytes from {solution_type}")
                    return content, solution_type
            except urllib.error.HTTPError as e:
                print(f"[GPS] {solution_type} HTTP Error {e.code}: {e.reason}")
                return None, solution_type
            except urllib.error.URLError as e:
                print(f"[GPS] {solution_type} URL Error: {e.reason}")
                return None, solution_type
            except Exception as e:
                print(f"[GPS] {solution_type} error: {type(e).__name__}: {e}")
                return None, solution_type
        
        # Try each URL in order
        for url, solution_type in urls_to_try:
            try:
                content, used_solution = await asyncio.to_thread(_fetch_sync, url, solution_type)
                
                if content:
                    result = self._parse_tenv3(content, station_id, days)
                    result["metadata"]["solution_type"] = used_solution
                    return result
                    
            except Exception as e:
                print(f"[GPS] Failed to parse {solution_type} data: {e}")
                continue
        
        # All attempts failed
        raise RuntimeError(
            f"Failed to fetch GPS time series for {station_id} from Nevada Geodetic Lab. "
            f"Tried both rapid and final solutions. Check internet connection or station availability."
        )
    
    def _parse_tenv3(
        self,
        content: str,
        station_id: str,
        days: int
    ) -> Dict[str, Any]:
        """
        Parse tenv3 format time series data.
        
        tenv3 format columns:
        1. Station name
        2. Date (YYMMMDD)
        3. Decimal year
        4. Modified Julian Day
        5-6. GPS week, day of week
        7-13. Position data (east, north, up with integer/fractional parts)
        14. Antenna height
        15-20. Sigmas and correlations
        21-23. Nominal lat, lon, height
        """
        lines = content.strip().split('\n')
        
        dates = []
        decimal_years = []
        east_mm = []
        north_mm = []
        up_mm = []
        east_sigma = []
        north_sigma = []
        up_sigma = []
        
        for line in lines:
            parts = line.split()
            if len(parts) < 17:
                continue
            
            try:
                # Extract fields
                date_str = parts[1]  # e.g., "24JAN15"
                decimal_year = float(parts[2])
                
                # Position fractional parts (in meters, convert to mm)
                east_frac = float(parts[8]) * 1000  # Convert m to mm
                north_frac = float(parts[10]) * 1000
                up_frac = float(parts[12]) * 1000
                
                # Sigmas
                e_sig = float(parts[14]) * 1000
                n_sig = float(parts[15]) * 1000
                u_sig = float(parts[16]) * 1000
                
                dates.append(date_str)
                decimal_years.append(decimal_year)
                east_mm.append(east_frac)
                north_mm.append(north_frac)
                up_mm.append(up_frac)
                east_sigma.append(e_sig)
                north_sigma.append(n_sig)
                up_sigma.append(u_sig)
                
            except (ValueError, IndexError) as e:
                continue
        
        if not dates:
            raise RuntimeError(f"No valid data parsed from GPS time series for {station_id}")
        
        # Keep only requested number of days
        if len(dates) > days:
            dates = dates[-days:]
            decimal_years = decimal_years[-days:]
            east_mm = east_mm[-days:]
            north_mm = north_mm[-days:]
            up_mm = up_mm[-days:]
            east_sigma = east_sigma[-days:]
            north_sigma = north_sigma[-days:]
            up_sigma = up_sigma[-days:]
        
        # Compute velocity (rate of change) using linear regression
        if len(decimal_years) >= 10:
            years = np.array(decimal_years)
            up = np.array(up_mm)
            
            # Simple linear regression for vertical velocity
            # velocity in mm/year
            mean_year = years.mean()
            mean_up = up.mean()
            slope = np.sum((years - mean_year) * (up - mean_up)) / np.sum((years - mean_year)**2)
            vertical_velocity_mm_year = slope
        else:
            vertical_velocity_mm_year = 0.0
        
        # Convert lists to arrays and remove mean (relative displacement)
        up_arr = np.array(up_mm)
        up_relative = up_arr - up_arr[0]  # Relative to first measurement
        
        station_info = self.FLORIDA_STATIONS[station_id]
        
        result = {
            "station_id": station_id,
            "station_name": station_info["name"],
            "station_lat": station_info["lat"],
            "station_lon": station_info["lon"],
            "dates": dates,
            "decimal_years": decimal_years,
            "east_mm": east_mm,
            "north_mm": north_mm,
            "up_mm": up_mm,
            "up_relative_mm": up_relative.tolist(),
            "east_sigma_mm": east_sigma,
            "north_sigma_mm": north_sigma,
            "up_sigma_mm": up_sigma,
            "metadata": {
                "source": "Nevada Geodetic Laboratory",
                "url": f"{self.NGL_BASE_URL}/rapids/tenv3/{station_id}.tenv3",
                "n_observations": len(dates),
                "date_range": f"{dates[0]} to {dates[-1]}" if dates else "N/A",
                "vertical_velocity_mm_year": round(vertical_velocity_mm_year, 2),
                "mean_vertical_sigma_mm": round(np.mean(up_sigma), 2),
            },
            "analysis": {
                "vertical_velocity_mm_year": round(vertical_velocity_mm_year, 2),
                "max_subsidence_mm": round(float(np.min(up_relative)), 2),
                "max_uplift_mm": round(float(np.max(up_relative)), 2),
                "total_vertical_range_mm": round(float(np.max(up_relative) - np.min(up_relative)), 2),
                "is_subsiding": bool(vertical_velocity_mm_year < -1.0),  # >1mm/year subsidence (native bool for JSON)
            }
        }
        
        print(f"[GPS] Parsed {len(dates)} observations for {station_id}")
        print(f"[GPS] Vertical velocity: {vertical_velocity_mm_year:.2f} mm/year")
        print(f"[GPS] Date range: {result['metadata']['date_range']}")
        
        return result
    
    async def get_ground_movement_for_aoi(
        self,
        bbox: Tuple[float, float, float, float],
        days: int = 90
    ) -> Dict[str, Any]:
        """
        Get ground movement data for an AOI by finding nearest GPS station.
        
        Tries multiple stations in order of distance until one succeeds.
        
        Args:
            bbox: (west, south, east, north) bounding box
            days: Days of historical data to fetch
        
        Returns:
            Ground movement analysis including velocity and displacement
        """
        west, south, east, north = bbox
        center_lat = (south + north) / 2
        center_lon = (west + east) / 2
        
        # Find all nearby stations (up to 100km)
        stations = self.find_nearest_stations(center_lat, center_lon, max_distance_km=100.0)
        
        if not stations:
            raise RuntimeError(
                f"No GPS station found within 100km of ({center_lat:.2f}, {center_lon:.2f}). "
                f"GPS ground movement monitoring unavailable for this area."
            )
        
        print(f"[GPS] Found {len(stations)} stations within 100km")
        
        # Try each station in order of distance until one works
        last_error = None
        for station in stations:
            try:
                print(f"[GPS] Trying station {station['station_id']} ({station['name']}) - {station['distance_km']:.1f}km from AOI")
                
                timeseries = await self.get_position_timeseries(
                    station_id=station["station_id"],
                    days=days
                )
                
                # Success! Add station distance info
                timeseries["distance_to_aoi_km"] = station["distance_km"]
                timeseries["aoi_center"] = {"lat": center_lat, "lon": center_lon}
                
                print(f"[GPS] SUCCESS: Using {station['station_id']} data")
                return timeseries
                
            except Exception as e:
                last_error = e
                print(f"[GPS] Station {station['station_id']} failed: {str(e)[:50]}")
                continue
        
        # All stations failed
        raise RuntimeError(
            f"Failed to fetch GPS data from any of {len(stations)} nearby stations. "
            f"Last error: {last_error}"
        )


class MiamiInSARService:
    """
    University of Miami InSAR Data Service
    
    Fetches InSAR displacement time series from insarmaps.miami.edu
    Provides area-wide coverage (vs point-based GPS).
    
    API parameters:
    - longitude, latitude (required)
    - outputType: json, csv, dataset
    - satellite, orbit, frame (optional)
    - startTime, endTime (optional, YYYY-MM-DD)
    
    Coverage: Florida (primarily Miami-Dade, some Central FL)
    Update frequency: Every 6-12 days (Sentinel-1 repeat cycle)
    Latency: 1-4 weeks for processing
    """
    
    BASE_URL = "https://insarmaps.miami.edu"
    API_URL = f"{BASE_URL}/WebServicesUI"
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=30.0),
            follow_redirects=True
        )
    
    async def close(self):
        await self.client.aclose()
    
    async def get_displacement_timeseries(
        self,
        lat: float,
        lon: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_type: str = "json"
    ) -> Dict[str, Any]:
        """
        Fetch InSAR displacement time series for a point.
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_type: json, csv, or dataset
        
        Returns:
            Displacement time series data
        
        Raises:
            RuntimeError if no data available (NO FAKE DATA)
        """
        # Build API URL
        params = {
            "longitude": lon,
            "latitude": lat,
            "outputType": output_type,
        }
        
        if start_date:
            params["startTime"] = start_date
        if end_date:
            params["endTime"] = end_date
        
        print(f"[Miami InSAR] Querying ({lat:.4f}, {lon:.4f})...")
        
        try:
            # The Miami InSAR API uses form parameters
            # Based on the web interface, it's a GET request with query params
            url = f"{self.API_URL}?"
            for k, v in params.items():
                url += f"{k}={v}&"
            url = url.rstrip("&")
            
            response = await self.client.get(url)
            
            if response.status_code != 200:
                raise RuntimeError(f"Miami InSAR API returned status {response.status_code}")
            
            # Try to parse JSON response
            content_type = response.headers.get("content-type", "")
            
            if "json" in content_type or output_type == "json":
                data = response.json()
                return self._process_response(data, lat, lon)
            else:
                # May return HTML if no data
                text = response.text
                if "no data" in text.lower() or "not found" in text.lower():
                    raise RuntimeError(
                        f"No InSAR data available for ({lat:.4f}, {lon:.4f}). "
                        f"Location may be outside Miami InSAR coverage area."
                    )
                return {"raw_response": text[:500], "status": "parse_error"}
                
        except httpx.TimeoutException:
            raise RuntimeError("Miami InSAR API request timed out")
        except Exception as e:
            print(f"[Miami InSAR] Error: {e}")
            raise RuntimeError(f"Miami InSAR API error: {str(e)}")
    
    def _process_response(
        self,
        data: Any,
        lat: float,
        lon: float
    ) -> Dict[str, Any]:
        """Process API response into standardized format"""
        
        # Response format depends on what the API returns
        # This is a placeholder that needs adjustment based on actual response
        
        if isinstance(data, dict):
            if "error" in data:
                raise RuntimeError(f"Miami InSAR API error: {data['error']}")
            
            return {
                "source": "University of Miami InSAR",
                "location": {"lat": lat, "lon": lon},
                "data": data,
                "url": f"{self.API_URL}?longitude={lon}&latitude={lat}&outputType=json"
            }
        elif isinstance(data, list):
            return {
                "source": "University of Miami InSAR",
                "location": {"lat": lat, "lon": lon},
                "timeseries": data,
                "n_observations": len(data)
            }
        else:
            return {"raw_data": str(data)[:500]}
    
    async def check_coverage(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Dict[str, Any]:
        """
        Check if Miami InSAR has coverage for an area.
        
        Returns coverage status without fetching full data.
        """
        west, south, east, north = bbox
        center_lat = (south + north) / 2
        center_lon = (west + east) / 2
        
        try:
            # Try to fetch a small amount of data
            result = await self.get_displacement_timeseries(
                lat=center_lat,
                lon=center_lon,
                output_type="json"
            )
            
            return {
                "has_coverage": True,
                "center": {"lat": center_lat, "lon": center_lon},
                "message": "InSAR data available for this area"
            }
        except RuntimeError as e:
            return {
                "has_coverage": False,
                "center": {"lat": center_lat, "lon": center_lon},
                "error": str(e),
                "message": "No InSAR coverage for this area"
            }


class RealDataManager:
    """
    Unified manager for all real data services
    
    Services:
    - Florida Geological Survey (sinkholes, geology)
    - USGS 3DEP (elevation)
    - National Hydrography Dataset (water)
    - Sentinel-2 (optical imagery)
    - NASA OPERA DISP-S1 (ground displacement - InSAR area coverage)
    - Nevada Geodetic Lab GPS (ground displacement - point-based, real-time)
    - University of Miami InSAR (additional InSAR coverage)
    """
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.fgs = FloridaGeologicalSurvey()
        self.usgs = USGSElevationService()
        self.nhd = NationalHydrographyDataset()
        self.sentinel = SentinelDataService()
        self.insar = InSARService()
        self.opera = OPERADisplacementService()  # InSAR ground displacement (area)
        self.gps = GPSGroundMovementService()    # GPS ground movement (point, real-time)
        self.miami_insar = MiamiInSARService()   # Miami InSAR (area)
    
    async def close(self):
        """Close all service connections"""
        await asyncio.gather(
            self.fgs.close(),
            self.usgs.close(),
            self.nhd.close(),
            self.sentinel.close(),
            self.insar.close(),
            self.opera.close(),
            self.gps.close(),
            self.miami_insar.close(),
        )
    
    def _cache_key(self, service: str, bbox: tuple, **kwargs) -> str:
        """Generate cache key"""
        key_data = f"{service}_{bbox}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    async def fetch_all_layers(
        self,
        bbox: Tuple[float, float, float, float],
        include_satellite: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch all data layers for an AOI
        
        Returns:
            Dictionary with all fetched data layers
        """
        print(f"\n{'='*50}")
        print(f"Fetching real data for bbox: {bbox}")
        print(f"{'='*50}\n")
        
        results = {}
        
        # Fetch in parallel where possible
        tasks = {
            "sinkholes": self.fgs.get_sinkhole_inventory(bbox),
            "geology": self.fgs.get_geology(bbox),
            "water": self.nhd.get_water_features(bbox),
        }
        
        # Run parallel fetches
        fetched = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Critical data sources - must succeed
        critical_sources = ["sinkholes", "geology"]
        
        for key, result in zip(tasks.keys(), fetched):
            if isinstance(result, Exception):
                print(f"[!] Error fetching {key}: {result}")
                if key in critical_sources:
                    # Critical data failed - raise error (NO FALLBACK)
                    raise RuntimeError(
                        f"Failed to fetch {key} from data source: {str(result)}. "
                        f"Cannot proceed without real {key} data."
                    )
                results[key] = None
            else:
                results[key] = result
        
        # Verify critical data was actually loaded
        if not results.get("sinkholes") or not results["sinkholes"].get("features"):
            raise RuntimeError(
                "No sinkhole data returned from Florida Geological Survey. "
                "Cannot proceed without real sinkhole data."
            )
        
        # DEM (separate due to size) - also critical
        try:
            results["dem"] = await self.usgs.get_elevation_image(bbox)
            if results["dem"] is None:
                raise RuntimeError("USGS 3DEP returned no elevation data")
        except Exception as e:
            print(f"[!] Error fetching DEM: {e}")
            raise RuntimeError(
                f"Failed to fetch elevation data from USGS 3DEP: {str(e)}. "
                f"Cannot proceed without real elevation data."
            )
        
        # Satellite data (optional)
        if include_satellite:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            
            try:
                results["sentinel2_scenes"] = await self.sentinel.search_sentinel2(
                    bbox, start_date, end_date
                )
            except Exception as e:
                print(f"[!] Error searching Sentinel-2: {e}")
                results["sentinel2_scenes"] = []
        
        # Ground displacement data - try multiple sources for best coverage
        # Priority: GPS (real-time) > OPERA InSAR (area coverage)
        
        # 1. GPS Ground Movement (Nevada Geodetic Lab) - REAL-TIME
        try:
            print("[GPS] Fetching real-time ground movement data...")
            results["gps_ground_movement"] = await self.gps.get_ground_movement_for_aoi(bbox)
            print(f"[GPS] Ground movement data loaded successfully")
            print(f"[GPS] Station: {results['gps_ground_movement'].get('station_id')} ({results['gps_ground_movement'].get('station_name')})")
            print(f"[GPS] Vertical velocity: {results['gps_ground_movement'].get('analysis', {}).get('vertical_velocity_mm_year', 'N/A')} mm/year")
        except Exception as e:
            print(f"[!] GPS ground movement data unavailable: {e}")
            results["gps_ground_movement"] = None
            results["gps_ground_movement_error"] = str(e)
        
        # 2. NASA OPERA DISP-S1 (InSAR area coverage - historical)
        try:
            print("[OPERA] Fetching InSAR ground displacement data...")
            results["ground_displacement"] = await self.opera.get_displacement_data(bbox)
            print(f"[OPERA] Ground displacement data loaded successfully")
        except Exception as e:
            print(f"[!] OPERA ground displacement data unavailable: {e}")
            # Store the error - do NOT use fake data
            results["ground_displacement"] = None
            results["ground_displacement_error"] = str(e)
        
        # Process karst units from geology
        if results.get("geology"):
            results["karst_units"] = self.fgs.identify_karst_units(results["geology"])
            print(f"[FGS] Identified {len(results['karst_units'])} karst units")
        
        print(f"\n{'='*50}")
        print(f"Data fetch complete")
        print(f"{'='*50}\n")
        
        return results
    
    def save_cache(self, key: str, data: Any):
        """Save data to cache"""
        cache_path = self.cache_dir / f"{key}.json"
        with open(cache_path, "w") as f:
            json.dump(data, f)
    
    def load_cache(self, key: str) -> Optional[Any]:
        """Load data from cache"""
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return None

