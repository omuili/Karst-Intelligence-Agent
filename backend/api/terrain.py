"""
3D Terrain (DEM) API - OpenTopography
Fetches elevation data for 3D visualization in the sidebar viewer.
"""

import os
import numpy as np
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import settings, WinterParkAOI


router = APIRouter()


def _get_opentopography_key() -> str:
    key = (settings.opentopography_api_key or "") if settings else ""
    if not key:
        key = os.environ.get("OPENTOPOGRAPHY_API_KEY", "")
    return key


class TerrainRequest(BaseModel):
    lat: float
    lng: float
    size_km: float = 2.0
    resolution: str = "10m"


@router.post("/terrain")
async def get_terrain_data(request: TerrainRequest):
    """Fetch 3D terrain elevation from OpenTopography; returns elevation grid for sidebar 3D viewer."""
    api_key = _get_opentopography_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenTopography API key not configured. Set OPENTOPOGRAPHY_API_KEY in .env",
        )

    lat_offset = request.size_km / 111.0 / 2
    lng_offset = request.size_km / (111.0 * np.cos(np.radians(request.lat))) / 2
    south = request.lat - lat_offset
    north = request.lat + lat_offset
    west = request.lng - lng_offset
    east = request.lng + lng_offset

    dataset = "USGS10m" if request.resolution == "10m" else "SRTMGL1"
    api_endpoint = "usgsdem" if request.resolution == "10m" else "globaldem"
    url = f"https://portal.opentopography.org/API/{api_endpoint}"
    params = {
        "datasetName" if api_endpoint == "usgsdem" else "demtype": dataset,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "AAIGrid",
        "API_Key": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
            response = await client.get(url, params=params)

            if response.status_code == 204:
                if request.resolution == "10m":
                    fallback = TerrainRequest(
                        lat=request.lat,
                        lng=request.lng,
                        size_km=request.size_km,
                        resolution="30m",
                    )
                    return await get_terrain_data(fallback)
                raise HTTPException(status_code=404, detail="No terrain data available for this location")

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"OpenTopography API error: {response.text[:200]}")

            content = response.text
            lines = content.strip().split("\n")
            header = {}
            data_start = 0
            for i, line in enumerate(lines):
                if line and line[0].isalpha():
                    parts = line.split()
                    key = parts[0].lower()
                    val = parts[1] if len(parts) > 1 else ""
                    try:
                        header[key] = float(val) if "." in val or val.lstrip("-").replace(".", "").isdigit() else val
                    except ValueError:
                        header[key] = val
                    data_start = i + 1
                else:
                    break

            ncols = int(header.get("ncols", 0))
            nrows = int(header.get("nrows", 0))
            nodata = float(header.get("nodata_value", -9999))
            cellsize = float(header.get("cellsize", 1))

            elevation_data = []
            for line in lines[data_start:]:
                if line.strip():
                    row = [float(x) if float(x) != nodata else None for x in line.split()]
                    elevation_data.append(row)

            elevation_array = np.array(elevation_data, dtype=float)
            mask = np.isnan(elevation_array) | (elevation_array == nodata)
            if mask.any():
                valid = ~mask
                elevation_array[mask] = np.nanmin(elevation_array[valid]) if valid.any() else 0

            max_size = 200
            if elevation_array.shape[0] > max_size or elevation_array.shape[1] > max_size:
                step = max(elevation_array.shape[0] // max_size, elevation_array.shape[1] // max_size, 1)
                elevation_array = elevation_array[::step, ::step]

            elev_min = float(np.nanmin(elevation_array))
            elev_max = float(np.nanmax(elevation_array))
            elev_mean = float(np.nanmean(elevation_array))

            return {
                "success": True,
                "metadata": {
                    "center": {"lat": request.lat, "lng": request.lng},
                    "bounds": {"south": south, "north": north, "west": west, "east": east},
                    "resolution": request.resolution,
                    "actual_resolution_m": cellsize,
                    "grid_size": {"rows": elevation_array.shape[0], "cols": elevation_array.shape[1]},
                    "elevation_stats": {
                        "min_m": elev_min,
                        "max_m": elev_max,
                        "mean_m": elev_mean,
                        "range_m": elev_max - elev_min,
                    },
                },
                "elevation": elevation_array.tolist(),
            }
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Terrain data request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching terrain data: {str(e)[:200]}")


@router.get("/terrain/regions")
async def get_terrain_regions():
    """Predefined regions with known good DEM coverage for 3D terrain."""
    return {
        "regions": [
            {
                "id": "winter_park",
                "name": "Winter Park, FL",
                "description": "Site of 1981 sinkhole – focus area for susceptibility",
                "center": {"lat": WinterParkAOI.CENTER_LAT, "lng": WinterParkAOI.CENTER_LON},
            },
            {
                "id": "orlando",
                "name": "Orlando, FL",
                "description": "Orlando metropolitan area",
                "center": {"lat": 28.5383, "lng": -81.3792},
            },
            {
                "id": "ocala",
                "name": "Ocala, FL",
                "description": "Karst region with sinkhole lakes",
                "center": {"lat": 29.1872, "lng": -82.1401},
            },
        ]
    }
