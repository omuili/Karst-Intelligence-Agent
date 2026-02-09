"""
Data downloading utilities for Sinkhole Scanner

Downloads and caches:
- Sentinel-2 imagery from Microsoft Planetary Computer
- USGS 3DEP elevation data
- Florida Geological Survey geology layers
- FGS Sinkhole inventory
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime, timedelta

import httpx
import numpy as np
from tqdm import tqdm

# Lazy imports for heavy libraries
rasterio = None
geopandas = None


def _lazy_import():
    """Lazy import heavy libraries"""
    global rasterio, geopandas
    if rasterio is None:
        import rasterio as _rasterio
        rasterio = _rasterio
    if geopandas is None:
        import geopandas as _geopandas
        geopandas = _geopandas


class DataDownloader:
    """
    Downloads geospatial data for sinkhole analysis
    
    Data sources:
    - Sentinel-2: Microsoft Planetary Computer (free, API key optional)
    - DEM: USGS 3DEP via AWS/OpenTopography
    - Geology: Florida Geological Survey ArcGIS REST services
    - Sinkholes: FGS Subsidence Incident Reports
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.sentinel_dir = self.data_dir / "sentinel"
        self.dem_dir = self.data_dir / "dem"
        self.geology_dir = self.data_dir / "geology"
        self.sinkhole_dir = self.data_dir / "sinkholes"
        
        for d in [self.sentinel_dir, self.dem_dir, self.geology_dir, self.sinkhole_dir]:
            d.mkdir(exist_ok=True)
        
        # HTTP client
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
    
    async def download_all(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        Download all data layers for an AOI
        
        Args:
            bbox: (west, south, east, north) in WGS84
            start_date: Start date for imagery (YYYY-MM-DD)
            end_date: End date for imagery (YYYY-MM-DD)
        """
        print(f"📥 Downloading data for bbox: {bbox}")
        
        # Default to last 3 months
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        # Download in parallel where possible
        tasks = [
            self.download_sinkhole_inventory(bbox),
            self.download_geology(bbox),
        ]
        
        # DEM and Sentinel may need sequential download
        await asyncio.gather(*tasks)
        
        print("✓ Data download complete")
    
    async def download_sentinel(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20.0,
    ) -> Optional[Path]:
        """
        Download Sentinel-2 imagery from Planetary Computer
        
        Args:
            bbox: AOI bounding box
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_cloud_cover: Maximum cloud cover percentage
        
        Returns:
            Path to downloaded GeoTIFF or None
        """
        print("📡 Searching for Sentinel-2 imagery...")
        
        try:
            from pystac_client import Client
            import planetary_computer as pc
        except ImportError:
            print("⚠ pystac-client or planetary-computer not installed")
            return None
        
        try:
            # Connect to Planetary Computer
            catalog = Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=pc.sign_inplace,
            )
            
            # Search for items
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{start_date}/{end_date}",
                query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            )
            
            items = list(search.items())
            
            if not items:
                print("⚠ No Sentinel-2 imagery found matching criteria")
                return None
            
            # Get the least cloudy item
            items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
            best_item = items[0]
            
            print(f"  Found {len(items)} scenes, using: {best_item.id}")
            print(f"  Cloud cover: {best_item.properties.get('eo:cloud_cover', 'N/A')}%")
            print(f"  Date: {best_item.datetime}")
            
            # Download specific bands (B02, B03, B04, B08 for RGB + NIR)
            bands_to_download = ["B02", "B03", "B04", "B08"]
            
            _lazy_import()
            
            for band in bands_to_download:
                if band in best_item.assets:
                    asset = best_item.assets[band]
                    href = pc.sign(asset.href)
                    
                    output_path = self.sentinel_dir / f"{best_item.id}_{band}.tif"
                    
                    if not output_path.exists():
                        print(f"  Downloading {band}...")
                        await self._download_file(href, output_path)
            
            return self.sentinel_dir
            
        except Exception as e:
            print(f"⚠ Failed to download Sentinel-2: {e}")
            return None
    
    async def download_dem(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[Path]:
        """
        Download DEM from USGS 3DEP
        
        For Florida, we can use the 3DEP 1/3 arc-second (~10m) DEM
        """
        print("🏔️ Downloading DEM...")
        
        # USGS 3DEP WCS endpoint
        # Note: This is a simplified version - production would use proper WCS client
        
        west, south, east, north = bbox
        
        # Use OpenTopography Global DEM API as fallback
        dem_url = (
            f"https://portal.opentopography.org/API/globaldem"
            f"?demtype=SRTMGL1"
            f"&south={south}&north={north}&west={west}&east={east}"
            f"&outputFormat=GTiff"
        )
        
        output_path = self.dem_dir / "dem.tif"
        
        if output_path.exists():
            print("  DEM already downloaded")
            return output_path
        
        try:
            print("  Downloading from OpenTopography...")
            # Note: OpenTopography requires API key for production use
            # Real implementation would use proper authentication
            print("⚠ DEM download requires OpenTopography API key")
            return None
            
        except Exception as e:
            print(f"⚠ Failed to download DEM: {e}")
            return None
    
    async def download_geology(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[Path]:
        """
        Download geology layer from Florida Geological Survey
        
        Uses ArcGIS REST API for Florida Geologic Map
        """
        print("🪨 Downloading geology data...")
        
        output_path = self.geology_dir / "geology.geojson"
        
        if output_path.exists():
            print("  Geology data already downloaded")
            return output_path
        
        # FGS Geologic Map ArcGIS REST endpoint
        base_url = "https://ca.dep.state.fl.us/arcgis/rest/services/OpenData/FGS_GEOLOGY/MapServer/0/query"
        
        west, south, east, north = bbox
        
        params = {
            "where": "1=1",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "outFields": "*",
            "f": "geojson",
        }
        
        try:
            response = await self.client.get(base_url, params=params)
            response.raise_for_status()
            
            geojson = response.json()
            
            if geojson.get("features"):
                output_path.write_text(response.text)
                print(f"  Downloaded {len(geojson['features'])} geology features")
                return output_path
            else:
                print("  No geology features found in the specified area")
                return None
                
        except Exception as e:
            print(f"⚠ Failed to download geology: {e}")
            return None
    
    async def download_sinkhole_inventory(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[Path]:
        """
        Download sinkhole inventory from Florida Geological Survey
        
        Subsidence Incident Reports from FGS
        """
        print("🕳️ Downloading sinkhole inventory...")
        
        output_path = self.sinkhole_dir / "sinkholes.geojson"
        
        if output_path.exists():
            print("  Sinkhole inventory already downloaded")
            return output_path
        
        # FGS Subsidence Incident Reports endpoint
        base_url = "https://ca.dep.state.fl.us/arcgis/rest/services/OpenData/FGS_SUBSIDENCE_INCIDENT_REPORTS/MapServer/0/query"
        
        west, south, east, north = bbox
        
        params = {
            "where": "1=1",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "outFields": "*",
            "f": "geojson",
        }
        
        try:
            response = await self.client.get(base_url, params=params)
            response.raise_for_status()
            
            geojson = response.json()
            
            if geojson.get("features"):
                output_path.write_text(response.text)
                print(f"  Downloaded {len(geojson['features'])} sinkhole records")
                return output_path
            else:
                print("  No sinkhole records found in the specified area")
                return None
                
        except Exception as e:
            print(f"⚠ Failed to download sinkholes: {e}")
            return None
    
    async def _download_file(self, url: str, output_path: Path):
        """Download a file with progress bar"""
        async with self.client.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            
            with open(output_path, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc=output_path.name) as pbar:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
                        pbar.update(len(chunk))


async def main():
    """Download data for Winter Park AOI"""
    from backend.config import WinterParkAOI, settings
    
    downloader = DataDownloader(settings.data_dir)
    
    try:
        await downloader.download_all(
            bbox=WinterParkAOI.BBOX,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
    finally:
        await downloader.close()


if __name__ == "__main__":
    asyncio.run(main())

