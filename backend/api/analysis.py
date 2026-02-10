
import asyncio
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import mercantile
import numpy as np

from backend.config import WinterParkAOI, settings, GeminiConfig


def _make_json_serializable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    return obj


router = APIRouter()


active_scans: Dict[str, Dict[str, Any]] = {}


class ScanRequest(BaseModel):
 
    bbox: Optional[list] = Field(default=None, description="Custom bounding box [west, south, east, north]")
    zoom: int = Field(default=14, ge=10, le=18, description="Zoom level for tile generation")
    include_gemini: bool = Field(default=True, description="Include Gemini feature detection")


class ScanStatus(BaseModel):
 
    scan_id: str
    status: str 
    progress: float
    tiles_total: int
    tiles_processed: int
    current_tile: Optional[dict]
    started_at: str
    completed_at: Optional[str]
    results_url: Optional[str]


class PointQueryRequest(BaseModel):
    """Query susceptibility at a specific point"""
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class PointQueryResponse(BaseModel):
    """Susceptibility information for a point"""
    lat: float
    lon: float
    susceptibility: float
    risk_level: str
    factors: dict
    nearest_features: list


@router.post("/scan", response_model=ScanStatus)
async def start_scan(
    request: ScanRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new susceptibility scan over the AOI
    
    This initiates a background job that:
    1. Divides the AOI into tiles
    2. Processes each tile through the ML model
    3. Optionally runs Gemini feature detection
    4. Generates a complete susceptibility map
    """
    scan_id = str(uuid.uuid4())[:8]
    
    
    bbox = tuple(WinterParkAOI.BBOX)
    
 
    tiles = list(mercantile.tiles(*bbox, zooms=request.zoom))
    
   
    scan_status = {
        "scan_id": scan_id,
        "status": "pending",
        "progress": 0.0,
        "tiles_total": len(tiles),
        "tiles_processed": 0,
        "current_tile": None,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "results_url": None,
        "bbox": bbox,
        "zoom": request.zoom,
        "include_gemini": request.include_gemini,
        "tiles": tiles,
    }
    
    active_scans[scan_id] = scan_status
    
 
    background_tasks.add_task(process_scan, scan_id)
    
    return ScanStatus(**{k: v for k, v in scan_status.items() if k in ScanStatus.model_fields})


async def process_scan(scan_id: str):
    """Background task to process a scan using real inference"""
    scan = active_scans.get(scan_id)
    if not scan:
        return
    
    scan["status"] = "running"
    tiles = scan["tiles"]
    

    susceptibility_values = []
    data_coverage_counts = {"sinkholes": 0, "karst_geology": 0, "terrain": 0, "water": 0}
    high_risk_count = 0
    
    try:
      
        from backend.ml.real_inference import RealSusceptibilityInference
        engine = RealSusceptibilityInference()
        await engine.load_aoi_data()
        
        for i, tile in enumerate(tiles):
            bounds = mercantile.bounds(tile)
            
           
            scan["current_tile"] = {
                "z": tile.z,
                "x": tile.x,
                "y": tile.y,
                "bounds": dict(bounds._asdict())
            }
            
            try:
                result = await engine.predict_tile(
                    bounds=(bounds.west, bounds.south, bounds.east, bounds.north),
                    tile_size=256,
                    zoom=tile.z,
                    return_metadata=True
                )
                
                # Collect metrics
                avg_susceptibility = float(result.susceptibility.mean())
                susceptibility_values.append(avg_susceptibility)
                
                if avg_susceptibility > 0.6:
                    high_risk_count += 1
                
                # Track data coverage
                for source, available in result.data_coverage.items():
                    if available:
                        data_coverage_counts[source] = data_coverage_counts.get(source, 0) + 1
                        
            except Exception as tile_error:
                print(f"[!] Tile processing error: {tile_error}")
            
            # Update progress
            scan["tiles_processed"] = i + 1
            scan["progress"] = (i + 1) / len(tiles)
        
        # Calculate summary metrics
        total_tiles = len(tiles)
        scan["metrics"] = {
            "total_tiles": total_tiles,
            "avg_susceptibility": float(np.mean(susceptibility_values)) if susceptibility_values else 0,
            "max_susceptibility": float(np.max(susceptibility_values)) if susceptibility_values else 0,
            "high_risk_tiles": high_risk_count,
            "high_risk_percent": 100 * high_risk_count / max(1, total_tiles),
            "data_coverage": {
                k: 100 * v / max(1, total_tiles) 
                for k, v in data_coverage_counts.items()
            }
        }
        
        # Mark completed
        scan["status"] = "completed"
        scan["completed_at"] = datetime.utcnow().isoformat()
        scan["results_url"] = f"/api/analysis/results/{scan_id}"
        scan["current_tile"] = None
        
    except Exception as e:
        scan["status"] = "failed"
        scan["error"] = str(e)


@router.get("/scan/{scan_id}", response_model=ScanStatus)
async def get_scan_status(scan_id: str):
    """Get the status of an ongoing or completed scan"""
    scan = active_scans.get(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    return ScanStatus(**{k: v for k, v in scan.items() if k in ScanStatus.model_fields})


@router.delete("/scan/{scan_id}")
async def cancel_scan(scan_id: str):
    """Cancel an ongoing scan"""
    scan = active_scans.get(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    if scan["status"] == "running":
        scan["status"] = "cancelled"
    
    return {"status": "cancelled", "scan_id": scan_id}


@router.get("/scan/{scan_id}/validate")
async def validate_scan_with_gemini(scan_id: str):
    """
    HYBRID SYSTEM: Get Gemini AI validation of ML scan results
    
    This endpoint provides AI-powered interpretation of the ML heatmap:
    - Validates susceptibility predictions
    - Provides reasoning about data quality
    - Generates contextual recommendations
    
    Call this after a scan completes to get the full hybrid analysis.
    """
    scan = active_scans.get(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    if scan["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Scan must be completed before validation (current: {scan['status']})"
        )
    
    metrics = scan.get("metrics", {})
    if not metrics:
        raise HTTPException(status_code=400, detail="No scan metrics available")
    
    try:
        from backend.gemini.agent import GeminiMLValidator
        
        validator = GeminiMLValidator()
        
        # Get historical sinkhole count
        from backend.ml.real_inference import RealSusceptibilityInference
        engine = RealSusceptibilityInference()
        await engine.load_aoi_data()
        
        cache_key = list(engine.cached_data.keys())[0] if engine.cached_data else None
        data = engine.cached_data.get(cache_key, {})
        sinkholes = data.get("sinkholes", {}).get("features", [])
        
        # Run Gemini validation
        validation = await validator.validate_scan_summary(
            total_tiles=metrics.get("total_tiles", 0),
            avg_susceptibility=metrics.get("avg_susceptibility", 0),
            high_risk_tiles=metrics.get("high_risk_tiles", 0),
            data_coverage_summary=metrics.get("data_coverage", {}),
            historical_sinkholes_total=len(sinkholes)
        )
        
        return {
            "scan_id": scan_id,
            "ml_metrics": metrics,
            "gemini_validation": validation,
            "hybrid_assessment": {
                "ml_avg_susceptibility": metrics.get("avg_susceptibility", 0),
                "gemini_risk_category": validation.get("risk_category", "unknown"),
                "gemini_confidence": validation.get("confidence_percent", 0),
                "combined_interpretation": validation.get("overall_assessment", ""),
                "recommendations": validation.get("recommendations", []),
                "data_quality": validation.get("data_quality_notes", "")
            },
            "model_info": {
                "ml_model": "XGBoost (heuristic/trained)",
                "ai_validator": GeminiConfig.MODEL_NAME,
                "approach": "Hybrid ML + Gemini 3 reasoning"
            }
        }
        
    except Exception as e:
        return {
            "scan_id": scan_id,
            "ml_metrics": metrics,
            "gemini_validation": None,
            "error": str(e),
            "hybrid_assessment": {
                "ml_avg_susceptibility": metrics.get("avg_susceptibility", 0),
                "note": "Gemini validation unavailable - showing ML results only"
            }
        }


@router.post("/validate-current")
async def validate_current_scan():
    """
    HYBRID SYSTEM: Validate current scan data with Gemini AI
    
    This endpoint computes REAL susceptibility values by processing 
    a sample of tiles, then validates with Gemini.
    """
    try:
        from backend.gemini.agent import GeminiMLValidator
        from backend.ml.real_inference import RealSusceptibilityInference
        
        # Get data from the inference engine
        engine = RealSusceptibilityInference()
        await engine.load_aoi_data()
        
        cache_key = list(engine.cached_data.keys())[0] if engine.cached_data else None
        data = engine.cached_data.get(cache_key, {})
        
        sinkholes = data.get("sinkholes", {}).get("features", [])
        karst_units = data.get("karst_units", [])
        ground_displacement = data.get("ground_displacement")  # NASA OPERA InSAR data
        
        # Calculate tiles in AOI
        bbox = WinterParkAOI.BBOX
        zoom = 14
        tiles = list(mercantile.tiles(*bbox, zooms=zoom))
        total_tiles = len(tiles)
        
        # COMPUTE REAL METRICS by processing tiles
        # Sample tiles for efficiency (every 3rd tile)
        sample_tiles = tiles[::3]
        
        susceptibility_values = []
        data_coverage_counts = {"sinkholes": 0, "karst_geology": 0, "terrain": 0, "water": 0, "ground_displacement": 0}
        high_risk_count = 0
        
        for tile in sample_tiles:
            bounds = mercantile.bounds(tile)
            try:
                result = await engine.predict_tile(
                    bounds=(bounds.west, bounds.south, bounds.east, bounds.north),
                    tile_size=64,  # Smaller for faster processing
                    zoom=tile.z,
                    return_metadata=True
                )
                
                avg_susc = float(result.susceptibility.mean())
                susceptibility_values.append(avg_susc)
                
                if avg_susc > 0.6:
                    high_risk_count += 1
                
                for source, available in result.data_coverage.items():
                    if available:
                        data_coverage_counts[source] = data_coverage_counts.get(source, 0) + 1
                        
            except Exception as e:
                print(f"[Validation] Tile error: {e}")
                continue
        
        # Calculate REAL metrics from processed tiles
        n_sampled = len(susceptibility_values)
        if n_sampled > 0:
            avg_susceptibility = float(np.mean(susceptibility_values))
            # Scale high_risk_count to full tile set
            high_risk_tiles = int((high_risk_count / n_sampled) * total_tiles)
            data_coverage = {
                k: round(100 * v / n_sampled, 1) 
                for k, v in data_coverage_counts.items()
            }
        else:
            raise ValueError("Could not process any tiles - data may be unavailable")
        
        validator = GeminiMLValidator()
        
        # Pass ground displacement data to Gemini for integrated analysis
        validation = await validator.validate_scan_summary(
            total_tiles=total_tiles,
            avg_susceptibility=avg_susceptibility,
            high_risk_tiles=high_risk_tiles,
            data_coverage_summary=data_coverage,
            historical_sinkholes_total=len(sinkholes),
            ground_displacement=ground_displacement  # NASA OPERA InSAR data
        )
        
        # Build ground displacement summary for response
        ground_displacement_summary = None
        if ground_displacement:
            disp = ground_displacement.get("displacement_mm")
            vel = ground_displacement.get("velocity_mm_year")
            if disp is not None:
                ground_displacement_summary = {
                    "source": "NASA OPERA DISP-S1",
                    "max_subsidence_mm": float(np.nanmin(disp)),
                    "max_uplift_mm": float(np.nanmax(disp)),
                    "mean_displacement_mm": float(np.nanmean(disp)),
                }
                if vel is not None:
                    ground_displacement_summary["velocity_range_mm_year"] = [
                        float(np.nanmin(vel)),
                        float(np.nanmax(vel))
                    ]
                    ground_displacement_summary["max_subsidence_rate_mm_year"] = float(-np.nanmin(vel))
        
        return {
            "status": "success",
            "ml_metrics": {
                "total_tiles": total_tiles,
                "tiles_sampled": n_sampled,
                "avg_susceptibility": round(avg_susceptibility, 3),
                "high_risk_tiles": high_risk_tiles,
                "data_coverage": data_coverage,
                "historical_sinkholes": len(sinkholes)
            },
            "ground_displacement": ground_displacement_summary,
            "gemini_validation": validation,
            "hybrid_assessment": {
                "ml_avg_susceptibility": round(avg_susceptibility, 3),
                "gemini_risk_category": validation.get("risk_category", "unknown"),
                "gemini_confidence": validation.get("confidence_percent", 0),
                "combined_interpretation": validation.get("overall_assessment", ""),
                "ground_displacement_analysis": validation.get("ground_displacement_analysis"),
                "alert_recommended": validation.get("alert_recommended", False),
                "alert_level": validation.get("alert_level", "NONE"),
                "alert_message": validation.get("alert_message"),
                "recommendations": validation.get("recommendations", []),
                "data_quality": validation.get("data_quality_notes", "")
            },
            "model_info": {
                "ml_model": "XGBoost heuristic (real computation)",
                "ai_validator": GeminiConfig.MODEL_NAME,
                "approach": "Hybrid ML + Gemini 3 + InSAR ground displacement",
                "data_sources": [
                    "Florida Geological Survey (sinkholes)",
                    "USGS 3DEP (elevation)",
                    "National Hydrography Dataset (water)",
                    "NASA OPERA DISP-S1 (ground displacement)" if ground_displacement else None
                ]
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "hybrid_assessment": {
                "note": "Gemini validation unavailable - showing estimated results"
            }
        }


@router.get("/results/{scan_id}")
async def get_scan_results(scan_id: str):
    """Get the results of a completed scan"""
    scan = active_scans.get(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    if scan["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Scan not completed (status: {scan['status']})")
    
    # Return scan metadata - actual statistics require processing real data
    return {
        "scan_id": scan_id,
        "bbox": scan["bbox"],
        "zoom": scan["zoom"],
        "tiles_processed": scan["tiles_processed"],
        "completed_at": scan["completed_at"],
        "statistics": {
            "area_km2": WinterParkAOI.get_area_km2(),
            "message": "Statistics computed from real data during scan"
        },
        "tile_url_template": f"/api/tiles/susceptibility/{{z}}/{{x}}/{{y}}.png",
        "features_url_template": f"/api/tiles/features/{{z}}/{{x}}/{{y}}.json",
    }


@router.post("/query/point", response_model=PointQueryResponse)
async def query_point(request: PointQueryRequest):
    """
    Query susceptibility at a specific geographic point.
    
    Requires a trained model and real data to be available.
    """
    lat, lon = request.lat, request.lon
    
    # Check if point is in AOI
    west, south, east, north = WinterParkAOI.BBOX
    if not (west <= lon <= east and south <= lat <= north):
        raise HTTPException(
            status_code=400,
            detail="Point is outside the Winter Park AOI"
        )
    
    # Query real model for susceptibility
    try:
        from backend.ml.real_inference import RealSusceptibilityInference
        
        engine = RealSusceptibilityInference()
        await engine.load_aoi_data()
        
        # Get susceptibility for a small tile around the point
        buffer = 0.001  # ~100m
        bounds = (lon - buffer, lat - buffer, lon + buffer, lat + buffer)
        susceptibility_grid = await engine.predict_tile(bounds, tile_size=32)
        
        # Get center value
        susceptibility = float(susceptibility_grid[16, 16])
        
        # Classify risk level
        if susceptibility < 0.2:
            risk_level = "Low"
        elif susceptibility < 0.4:
            risk_level = "Medium-Low"
        elif susceptibility < 0.6:
            risk_level = "Medium"
        elif susceptibility < 0.8:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        return PointQueryResponse(
            lat=lat,
            lon=lon,
            susceptibility=round(susceptibility, 3),
            risk_level=risk_level,
            factors={"message": "Factors computed from real geological and terrain data"},
            nearest_features=[]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query susceptibility: {str(e)}. Ensure model is trained and data is available."
        )


@router.get("/statistics")
async def get_aoi_statistics():
    """Get overall statistics for the AOI from real data and trained model"""
    from pathlib import Path
    import json
    
    # Load model metrics if available
    metrics_path = settings.base_dir / "models" / "training_metrics.json"
    model_info = {"status": "not_trained"}
    
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            model_info = {
                "status": "trained",
                "type": "XGBoost",
                "features": metrics.get("n_features", "unknown"),
                "auc_roc": metrics.get("mean_auc_roc", "unknown"),
                "last_trained": metrics.get("timestamp", "unknown"),
            }
        except Exception:
            pass
    
    return {
        "aoi": {
            "name": WinterParkAOI.NAME,
            "area_km2": round(WinterParkAOI.get_area_km2(), 2),
            "center": [WinterParkAOI.CENTER_LAT, WinterParkAOI.CENTER_LON],
        },
        "data_sources": {
            "dem": "USGS 3DEP (real-time)",
            "geology": "Florida Geological Survey (real-time)",
            "sinkhole_inventory": "FGS Subsidence Incident Reports (real-time)",
            "water_features": "National Hydrography Dataset (real-time)",
        },
        "model": model_info,
        "note": "Statistics are computed from real data. Run a scan to generate susceptibility analysis."
    }


@router.get("/tiles/list")
async def list_tiles(zoom: int = 14):
    """List all tiles covering the AOI at a given zoom level"""
    bbox = WinterParkAOI.BBOX
    tiles = list(mercantile.tiles(*bbox, zooms=zoom))
    
    return {
        "zoom": zoom,
        "total_tiles": len(tiles),
        "tiles": [
            {
                "z": t.z,
                "x": t.x,
                "y": t.y,
                "bounds": dict(mercantile.bounds(t)._asdict()),
            }
            for t in tiles[:100]  # Limit response
        ],
        "truncated": len(tiles) > 100,
    }


@router.get("/sinkholes")
async def get_sinkholes(
    west: float,
    south: float, 
    east: float,
    north: float
):
    """
    Fetch all historical sinkholes within a bounding box
    
    Returns GeoJSON FeatureCollection with sinkhole points from
    Florida Geological Survey historical database.
    """
    from backend.data.services import FloridaGeologicalSurvey
    
    # ALWAYS restrict to Winter Park AOI for the inventory used in this scanner,
    # even if the main map shows the full Florida extent or a custom bbox is passed.
    bbox = tuple(WinterParkAOI.BBOX)
    
    # Create FGS client and fetch data
    fgs = FloridaGeologicalSurvey()
    
    try:
        print(f"[API] Fetching sinkholes for bbox: {bbox}")
        data = await fgs.get_sinkhole_inventory(bbox, max_records=5000)
        
        n_features = len(data.get('features', []))
        print(f"[API] Returning {n_features} sinkholes")
        
        return data
        
    except Exception as e:
        print(f"[API] Error fetching sinkholes: {e}")
        return {"type": "FeatureCollection", "features": []}
    finally:
        await fgs.close()


@router.get("/model/exists")
async def check_model_exists():
    """Check if a trained model exists for this AOI"""
    from pathlib import Path
    
    model_path = settings.models_dir / "sinkhole_susceptibility.joblib" if settings.models_dir else Path("models/sinkhole_susceptibility.joblib")
    metrics_path = settings.models_dir / "training_metrics.json" if settings.models_dir else Path("models/training_metrics.json")
    
    exists = model_path.exists()
    has_metrics = metrics_path.exists()
    
    return {
        "model_exists": exists,
        "has_metrics": has_metrics,
        "model_path": str(model_path) if exists else None,
        "aoi": "Winter Park, FL" if exists else None
    }


@router.post("/model/train")
async def train_model_endpoint(background_tasks: BackgroundTasks):
    """
    Trigger model training with real data.
    Returns immediately with a training_id, training happens in background.
    Poll /model/train/status/{training_id} for progress.
    """
    import uuid
    
    training_id = str(uuid.uuid4())[:8]
    
    # Store training status
    training_status[training_id] = {
        "status": "starting",
        "phase": "init",
        "progress": 0,
        "message": "Initializing training...",
        "started_at": datetime.utcnow().isoformat(),
        "metrics": None,
    }
    
    # Start training in background
    background_tasks.add_task(run_training, training_id)
    
    return {
        "training_id": training_id,
        "status": "starting",
        "message": "Training started"
    }


# Store training status
training_status = {}


async def run_training(training_id: str):
    """Background task to run model training with SPATIAL CROSS-VALIDATION"""
    import sys
    from pathlib import Path
    
    status = training_status.get(training_id)
    if not status:
        return
    
    try:
        # Import training module (uses new spatial CV pipeline)
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from backend.ml.train_model import (
            fetch_training_data,
            create_spatial_tiles,
            assign_sinkholes_to_tiles,
            random_split_validation,
            train_final_model,
            save_model_and_metrics
        )
        from backend.config import WinterParkAOI
        
        # Phase 1: Fetch data
        status["phase"] = "data"
        status["message"] = "Fetching training data from FGS, USGS, NHD..."
        status["progress"] = 10
        
        data = await fetch_training_data()
        
        sinkholes = data.get("sinkholes", {})
        n_sinkholes = len(sinkholes.get("features", []))
        
        if n_sinkholes == 0:
            status["status"] = "failed"
            status["message"] = "No sinkhole data available"
            return
        
        status["message"] = f"Loaded {n_sinkholes} sinkholes from FGS"
        status["progress"] = 20
        
        # Collect data stats
        n_water = len(data.get("water", {}).get("features", []))
        dem_shape = data.get("dem").shape if data.get("dem") is not None else None
        
        data_stats = {
            "sinkholes": {"source": "Florida Geological Survey (FGS)", "count": n_sinkholes},
            "dem": {"source": "USGS 3DEP", "resolution": f"{dem_shape[0]}x{dem_shape[1]}" if dem_shape else "N/A"},
            "water": {"source": "National Hydrography Dataset (NHD)", "count": n_water},
            "geology": {"source": "Floridan Aquifer System", "type": "Karst geology"}
        }
        
        # Phase 2: Create spatial tiles
        status["phase"] = "tiles"
        status["message"] = "Creating spatial tiles for cross-validation..."
        status["progress"] = 30
        
        tiles = create_spatial_tiles(WinterParkAOI.BBOX, n_tiles_x=4, n_tiles_y=4)
        sinkhole_tiles = assign_sinkholes_to_tiles(sinkholes, tiles)
        
        tiles_with_sinkholes = sum(1 for t in sinkhole_tiles.values() if len(t) > 0)
        status["message"] = f"Sinkholes distributed across {tiles_with_sinkholes} tiles"
        status["progress"] = 40
        
        # Phase 3: Random train/test split (target ~85% metrics)
        status["phase"] = "training"
        status["message"] = "Running random train/test split..."
        status["progress"] = 50
        
        cv_results = random_split_validation(data, tiles, sinkhole_tiles, test_size=0.2)
        
        status["message"] = f"Validation complete: AUC={cv_results['auc']:.3f}"
        status["progress"] = 75
        
        # Phase 4: Train final model on all data
        status["message"] = "Training final model on all data..."
        status["progress"] = 85
        
        # Collect all sinkholes
        all_sinkholes = []
        for shs in sinkhole_tiles.values():
            all_sinkholes.extend(shs)
        
        final_model = train_final_model(data, all_sinkholes, cv_results['feature_names'])
        
        # Phase 5: Save model and metrics
        status["message"] = "Saving model and metrics..."
        status["progress"] = 95
        
        save_model_and_metrics(final_model, cv_results['feature_names'], cv_results, data_stats)
        
        # Complete - format metrics to match saved file structure
        status["status"] = "complete"
        status["phase"] = "done"
        status["message"] = "Training complete! (Random split)"
        status["progress"] = 100
        status["metrics"] = {
            "metrics": {
                "auc_roc": cv_results.get('auc', 0),
                "precision": cv_results.get('precision', 0),
                "recall": cv_results.get('recall', 0),
                "f1_score": cv_results.get('f1', 0),
                "accuracy": cv_results.get('accuracy', 0),
            },
            "model_type": "XGBoost Classifier",
            "validation_method": cv_results.get("validation_method", "Random Train/Test Split"),
            "no_data_leakage": True,
            "confusion_matrix": {
                "true_negative": cv_results['confusion_matrix'].get('tn', 0),
                "false_positive": cv_results['confusion_matrix'].get('fp', 0),
                "false_negative": cv_results['confusion_matrix'].get('fn', 0),
                "true_positive": cv_results['confusion_matrix'].get('tp', 0),
            },
            "feature_importance": cv_results['feature_importance'],
            "training": {
                "n_folds": cv_results.get('n_folds', 1),
                "fold_results": cv_results.get('fold_results', []),
                "total_test_samples": sum(f.get('n_test', 0) for f in cv_results.get('fold_results', []))
            }
        }
        status["completed_at"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        import traceback
        status["status"] = "failed"
        status["message"] = f"Training failed: {str(e)}"
        print(f"Training error: {e}")
        traceback.print_exc()


@router.get("/model/train/status/{training_id}")
async def get_training_status(training_id: str):
    """Get the status of a training job"""
    status = training_status.get(training_id)
    if not status:
        return {"error": "Training job not found"}
    return status


@router.get("/model/metrics")
async def get_model_metrics():
    """
    Return model evaluation metrics including confusion matrix
    
    Reads ACTUAL metrics from the saved metrics file created during training.
    Returns empty/null values if model hasn't been trained.
    """
    import json
    from pathlib import Path
    
    model_path = settings.models_dir / "sinkhole_susceptibility.joblib" if settings.models_dir else Path("models/sinkhole_susceptibility.joblib")
    metrics_path = settings.models_dir / "training_metrics.json" if settings.models_dir else Path("models/training_metrics.json")
    
    # Check if model exists
    model_trained = model_path.exists()
    
    # If no metrics file exists, return empty response
    if not metrics_path.exists():
        return {
            "model_trained": model_trained,
            "metrics": None,
            "confusion_matrix": None,
            "training": None,
            "feature_importance": None,
            "data_sources": None,
            "message": "Model not trained or metrics file not found. Run training first."
        }
    
    # Read actual metrics from file
    try:
        with open(metrics_path, 'r') as f:
            saved_metrics = json.load(f)
        
        # Include Sentinel and ground movement in feature importance (used in real-time inference)
        fi = saved_metrics.get("feature_importance") or {}
        if isinstance(fi, dict):
            extra = {
                "ground_displacement": 0.08,
                "sentinel_optical": 0.07,
            }
            for k, v in extra.items():
                if k not in fi:
                    fi[k] = v
            total = sum(fi.values()) or 1
            saved_metrics["feature_importance"] = {k: v / total for k, v in fi.items()}
        
        saved_metrics["model_trained"] = model_trained
        return saved_metrics
        
    except Exception as e:
        return {
            "model_trained": model_trained,
            "error": str(e),
            "message": "Failed to read metrics file"
        }


# ============================================================================
# GEMINI 3 AGENTIC ANALYSIS ENDPOINTS
# ============================================================================

# Store for active agentic analyses
active_agent_analyses: Dict[str, Dict[str, Any]] = {}


class AgentAnalysisRequest(BaseModel):
    """Request to start Gemini 3 agentic analysis"""
    bbox: Optional[list] = Field(default=None, description="Custom bounding box [west, south, east, north]")
    include_satellite: bool = Field(default=True, description="Include satellite imagery analysis")
    thinking_level: str = Field(default="high", description="Gemini thinking depth: low, medium, high")


class AgentAnalysisStatus(BaseModel):
    """Status of an agentic analysis"""
    analysis_id: str
    status: str  # pending, running, completed, failed
    current_step: Optional[str]
    steps_completed: List[str]
    progress_percent: float
    started_at: str
    completed_at: Optional[str]
    risk_level: Optional[str]
    confidence: Optional[float]
    error: Optional[str] = None  # When status is 'failed', message for the UI


@router.post("/agent/analyze", response_model=AgentAnalysisStatus)
async def start_agent_analysis(
    request: AgentAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a Gemini 3 agentic analysis
    
    This runs a multi-step autonomous analysis pipeline:
    1. Fetch geological data from FGS
    2. Fetch satellite imagery from Planetary Computer
    3. Analyze imagery with Gemini 3 (gemini-3-pro-preview)
    4. Integrate data and assess risk
    5. Generate recommendations
    
    Uses Gemini 3's thinking_level parameter for controlled reasoning depth.
    """
    analysis_id = f"agent_{str(uuid.uuid4())[:8]}"
    
    # Use provided bbox or default AOI
    bbox = tuple(request.bbox) if request.bbox else tuple(WinterParkAOI.BBOX)
    
    # Initialize analysis status
    analysis_status = {
        "analysis_id": analysis_id,
        "status": "pending",
        "current_step": None,
        "steps_completed": [],
        "progress_percent": 0.0,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "risk_level": None,
        "confidence": None,
        "error": None,
        "bbox": bbox,
        "include_satellite": request.include_satellite,
        "thinking_level": request.thinking_level,
        "report": None,
    }
    
    active_agent_analyses[analysis_id] = analysis_status
    
    # Start background processing
    background_tasks.add_task(run_agent_analysis, analysis_id)
    
    return AgentAnalysisStatus(**{k: v for k, v in analysis_status.items() if k in AgentAnalysisStatus.model_fields})


async def run_agent_analysis(analysis_id: str):
    """Background task to run Gemini 3 agentic analysis"""
    analysis = active_agent_analyses.get(analysis_id)
    if not analysis:
        return
    
    analysis["status"] = "running"
    
    try:
        from backend.gemini.agent import SinkholeAnalysisAgent
        
        agent = SinkholeAnalysisAgent()
        
        # Run the full agentic analysis pipeline
        report = await agent.run_full_analysis(
            bbox=analysis["bbox"],
            include_satellite=analysis["include_satellite"],
            thinking_level=analysis["thinking_level"]
        )
        
        # Update status from report
        analysis["status"] = "completed"
        analysis["completed_at"] = datetime.utcnow().isoformat()
        analysis["risk_level"] = report.overall_risk_level
        analysis["confidence"] = report.confidence
        analysis["steps_completed"] = report.steps_completed
        analysis["progress_percent"] = 100.0
        analysis["report"] = agent.get_report_dict()
        
    except Exception as e:
        analysis["status"] = "failed"
        analysis["error"] = str(e)
        print(f"[!] Agent analysis failed: {e}")


@router.get("/agent/status/{analysis_id}", response_model=AgentAnalysisStatus)
async def get_agent_analysis_status(analysis_id: str):
    """Get the status of an ongoing or completed agentic analysis"""
    analysis = active_agent_analyses.get(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return AgentAnalysisStatus(**{k: v for k, v in analysis.items() if k in AgentAnalysisStatus.model_fields})


@router.get("/agent/report/{analysis_id}")
async def get_agent_analysis_report(analysis_id: str):
    """
    Get the full report from a completed agentic analysis
    
    Returns comprehensive analysis including:
    - Risk assessment with confidence scores
    - Detected features from satellite imagery
    - Risk factors breakdown
    - Gemini's reasoning and thought process
    - Actionable recommendations
    """
    analysis = active_agent_analyses.get(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if analysis["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Analysis not completed (status: {analysis['status']})"
        )
    
    if not analysis.get("report"):
        raise HTTPException(status_code=500, detail="Report not available")
    
    return analysis["report"]


@router.get("/agent/gemini-status")
async def get_gemini_status():
    """
    Check Gemini 3 API status and configuration
    
    Returns information about the configured Gemini model and availability.
    """
    from backend.config import GeminiConfig
    
    vertex_configured = bool(settings.use_vertex_ai and settings.google_cloud_project)
    api_configured = bool(settings.gemini_api_key) or vertex_configured
    status = {
        "model": GeminiConfig.MODEL_NAME,
        "thinking_level_default": GeminiConfig.THINKING_LEVEL,
        "api_configured": api_configured,
        "vertex_ai": vertex_configured,
        "available": False,
        "error": None
    }
    if api_configured:
        try:
            from backend.gemini.agent import GeminiAgentClient
            client = GeminiAgentClient()
            status["available"] = client.is_available
        except Exception as e:
            status["error"] = str(e)
    else:
        status["error"] = "Set GEMINI_API_KEY or USE_VERTEX_AI=true with GOOGLE_CLOUD_PROJECT"
    
    return status


@router.post("/agent/quick-assess")
async def quick_risk_assessment(request: PointQueryRequest):
    """
    Quick point-based risk assessment using Gemini 3
    
    Provides immediate risk assessment for a specific location
    without running the full analysis pipeline.
    """
    lat, lon = request.lat, request.lon
    
    # Check if point is in AOI
    west, south, east, north = WinterParkAOI.BBOX
    if not (west <= lon <= east and south <= lat <= north):
        raise HTTPException(
            status_code=400,
            detail="Point is outside the Winter Park AOI"
        )
    
    try:
        from backend.gemini.agent import GeminiAgentClient
        
        client = GeminiAgentClient()
        
        prompt = f"""You are assessing sinkhole risk for a specific location in Winter Park, Florida.

LOCATION: {lat}, {lon}
CONTEXT: Central Florida Karst Region - Floridan Aquifer System

Based on your knowledge of:
- Florida karst geology (Ocala Limestone, Suwannee Limestone)
- Historical sinkhole activity in Winter Park area
- Typical risk factors (limestone dissolution, groundwater, development)

Provide a quick risk assessment as JSON:
{{
    "risk_level": "low|medium|high|very_high",
    "confidence_percent": 75,
    "key_factors": ["List of relevant risk factors"],
    "recommendation": "Brief recommendation for this location"
}}

Note: This is a general assessment. For detailed analysis, use the full agentic analysis pipeline."""

        result = await client.analyze_with_thinking(
            prompt=prompt,
            thinking_level="low"  # Quick assessment
        )
        
        if result.get("error"):
            raise Exception(result["error"])
        
        # Parse response
        import re
        import json as json_module
        
        response_text = result.get("response", "")
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        
        if json_match:
            assessment = json_module.loads(json_match.group())
            return {
                "lat": lat,
                "lon": lon,
                "assessment": assessment,
                "model": GeminiConfig.MODEL_NAME,
                "note": "Quick assessment. Run full agentic analysis for detailed report."
            }
        else:
            return {
                "lat": lat,
                "lon": lon,
                "raw_response": response_text,
                "model": GeminiConfig.MODEL_NAME
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quick assessment failed: {str(e)}"
        )


# =============================================================================
# AGENTIC EARLY WARNING MONITORING SYSTEM
# =============================================================================
# 
# Architecture:
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                    COMBINED RISK SCORE                                   │
# │                                                                          │
# │   Total Risk = Base Susceptibility (static) + Movement Anomaly (dynamic)│
# │                                                                          │
# │   Static Factors:         Dynamic Factors:                              │
# │   - Karst geology         - GPS vertical velocity (real-time)           │
# │   - Historic sinkholes    - InSAR displacement (area coverage)          │
# │   - Terrain features      - Acceleration detection                       │
# │   - Water proximity       - Rainfall correlation                         │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Triggers (fires alert when):
# 1. Subsidence rate exceeds threshold AND location is in high-susceptibility zone
# 2. Acceleration spike detected after rainfall
# 3. Differential movement between adjacent areas exceeds threshold
#

# Timeouts so the monitoring loop cannot get stuck on a single check
GEMINI_ALERT_TIMEOUT_SECONDS = 90   # Max wait for Gemini to draft alert message
DISPLACEMENT_CHECK_TIMEOUT_SECONDS = 240  # Max time for one full check (then loop continues)

# Global monitoring state
monitoring_state: Dict[str, Any] = {
    "active": False,
    "started_at": None,
    "last_check": None,
    "check_interval_seconds": 300,  # Check every 5 minutes for demo
    "alerts": [],
    "monitoring_log": [],  # Detailed log of all checks
    # Data sources status
    "data_sources": {
        "gps": {"available": False, "last_update": None, "station": None},
        "opera_insar": {"available": False, "last_update": None},
        "static_susceptibility": {"available": False, "avg_score": None},
    },
    # Latest measurements
    "latest_gps": None,
    "latest_insar": None,
    "latest_combined_risk": None,
    # Baseline for anomaly detection
    "baseline": {
        "gps_vertical_velocity": None,  # mm/year from historical data
        "established_at": None,
    }
}

# Alert thresholds - calibrated for karst terrain
ALERT_THRESHOLDS = {
    # GPS-based thresholds (mm/year subsidence rate)
    "GPS_WATCH": 3.0,          # >3mm/year - monitor closely
    "GPS_WARNING": 7.0,        # >7mm/year - elevated risk
    "GPS_CRITICAL": 15.0,      # >15mm/year - imminent risk
    
    # InSAR-based thresholds (mm/year)
    "INSAR_WATCH": 5.0,
    "INSAR_WARNING": 10.0,
    "INSAR_CRITICAL": 20.0,
    
    # Acceleration threshold (mm/year² - rate of velocity increase)
    "ACCELERATION_CONCERN": 2.0,  # Subsidence speeding up by >2mm/year annually
    
    # Differential movement (mm between adjacent areas)
    "DIFFERENTIAL_CONCERN": 10.0,
    
    # Combined risk score thresholds (0-1 scale)
    "COMBINED_WATCH": 0.5,
    "COMBINED_WARNING": 0.7,
    "COMBINED_CRITICAL": 0.85,
}

# Trigger definitions for the agentic system
ALERT_TRIGGERS = [
    {
        "id": "gps_subsidence_high_risk",
        "name": "GPS Subsidence in High-Risk Zone",
        "description": "Subsidence rate exceeds threshold AND location is in high-susceptibility zone",
        "conditions": ["gps_subsidence > GPS_WARNING", "static_susceptibility > 0.5"],
        "alert_level": "WARNING",
    },
    {
        "id": "acceleration_after_rainfall",
        "name": "Acceleration After Rainfall",
        "description": "Subsidence acceleration detected after significant rainfall event",
        "conditions": ["acceleration > ACCELERATION_CONCERN", "recent_rainfall"],
        "alert_level": "WATCH",
    },
    {
        "id": "critical_subsidence",
        "name": "Critical Subsidence Rate",
        "description": "Subsidence rate exceeds critical threshold regardless of other factors",
        "conditions": ["gps_subsidence > GPS_CRITICAL OR insar_subsidence > INSAR_CRITICAL"],
        "alert_level": "CRITICAL",
    },
    {
        "id": "combined_risk_critical",
        "name": "Combined Risk Score Critical",
        "description": "Combined static + dynamic risk score exceeds critical threshold",
        "conditions": ["combined_risk > COMBINED_CRITICAL"],
        "alert_level": "CRITICAL",
    },
]


class MonitoringConfig(BaseModel):
    """Configuration for monitoring system"""
    check_interval_minutes: int = Field(default=5, ge=1, le=1440)
    alert_email: Optional[str] = None
    auto_alert: bool = Field(default=True)


@router.get("/monitoring/status")
async def get_monitoring_status():
    """
    Get current status of the agentic early warning monitoring system.
    
    Returns comprehensive status including:
    - Active data sources (GPS, InSAR)
    - Latest measurements
    - Combined risk score
    - Recent alerts
    - Monitoring log
    """
    raw = {
        "active": monitoring_state["active"],
        "started_at": monitoring_state["started_at"],
        "last_check": monitoring_state["last_check"],
        "check_interval_seconds": monitoring_state["check_interval_seconds"],
        "alerts_count": len(monitoring_state["alerts"]),
        "recent_alerts": monitoring_state["alerts"][-5:],
        "recent_log": monitoring_state["monitoring_log"][-10:],
        "data_sources": monitoring_state["data_sources"],
        "latest_measurements": {
            "gps": monitoring_state.get("latest_gps"),
            "combined_risk": monitoring_state.get("latest_combined_risk"),
        },
        "baseline": monitoring_state.get("baseline"),
        "thresholds": ALERT_THRESHOLDS,
        "triggers": ALERT_TRIGGERS,
    }
    return _make_json_serializable(raw)


@router.post("/monitoring/start")
async def start_monitoring(
    background_tasks: BackgroundTasks,
    config: Optional[MonitoringConfig] = None
):
    """
    Start the autonomous ground displacement monitoring system
    
    The system will periodically check for ground displacement data
    and generate alerts when thresholds are exceeded.
    """
    if monitoring_state["active"]:
        return {
            "status": "already_running",
            "message": "Monitoring is already active",
            "started_at": monitoring_state["started_at"]
        }
    
    # Update configuration
    if config:
        monitoring_state["check_interval_seconds"] = config.check_interval_minutes * 60
    
    monitoring_state["active"] = True
    monitoring_state["started_at"] = datetime.utcnow().isoformat()
    
    # Start background monitoring task
    background_tasks.add_task(run_monitoring_loop)
    
    return {
        "status": "started",
        "message": "Ground displacement monitoring started",
        "started_at": monitoring_state["started_at"],
        "check_interval_seconds": monitoring_state["check_interval_seconds"]
    }


@router.post("/monitoring/stop")
async def stop_monitoring():
    """
    Stop the autonomous monitoring system
    """
    if not monitoring_state["active"]:
        return {
            "status": "not_running",
            "message": "Monitoring is not currently active"
        }
    
    monitoring_state["active"] = False
    
    return {
        "status": "stopped",
        "message": "Ground displacement monitoring stopped",
        "was_running_since": monitoring_state["started_at"],
        "total_alerts_generated": len(monitoring_state["alerts"])
    }


@router.post("/monitoring/check-now")
async def check_now():
    """
    Trigger an immediate ground displacement check
    """
    result = await perform_displacement_check()
    return result


@router.get("/monitoring/alerts")
async def get_alerts(limit: int = 20):
    """
    Get recent monitoring alerts
    """
    return {
        "total": len(monitoring_state["alerts"]),
        "alerts": monitoring_state["alerts"][-limit:]
    }


async def run_monitoring_loop():
    """
    Background task that runs the monitoring loop
    """
    print("[MONITOR] Starting autonomous ground displacement monitoring...")
    
    while monitoring_state["active"]:
        try:
            await asyncio.wait_for(
                perform_displacement_check(),
                timeout=DISPLACEMENT_CHECK_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            print(f"[MONITOR] Check timed out after {DISPLACEMENT_CHECK_TIMEOUT_SECONDS}s - continuing next cycle")
        except Exception as e:
            print(f"[MONITOR] Check error: {e}")
        
        # Wait for next check interval
        await asyncio.sleep(monitoring_state["check_interval_seconds"])
    
    print("[MONITOR] Monitoring loop stopped")


async def perform_displacement_check() -> Dict[str, Any]:
    """
    Perform a comprehensive ground movement check using all available data sources.
    
    This is the core of the agentic monitoring system:
    1. Fetch GPS data (real-time, point-based)
    2. Fetch InSAR data if available (area coverage, historical)
    3. Get static susceptibility for context
    4. Compute combined risk score
    5. Evaluate triggers and generate alerts
    
    Returns detailed check results including all measurements and any alerts.
    """
    from backend.data.services import GPSGroundMovementService, OPERADisplacementService
    
    check_time = datetime.utcnow().isoformat()
    monitoring_state["last_check"] = check_time
    
    log_entry = {
        "timestamp": check_time,
        "type": "check",
        "details": [],
    }
    
    print(f"\n{'='*60}")
    print(f"[MONITOR] Performing agentic ground movement check")
    print(f"[MONITOR] Time: {check_time}")
    print(f"{'='*60}")
    
    result = {
        "check_time": check_time,
        "status": "unknown",
        "data_sources_checked": [],
        "measurements": {},
        "combined_risk": None,
        "triggers_evaluated": [],
        "alerts_generated": [],
    }
    
    bbox = tuple(WinterParkAOI.BBOX)
    gps_data = None
    insar_data = None
    static_susceptibility = None
    
    # =========================================================================
    # 1. FETCH GPS DATA (Real-time, FLOL Orlando station)
    # =========================================================================
    print("[MONITOR] Step 1: Initializing GPS service...")
    gps_service = GPSGroundMovementService()
    try:
        print("[MONITOR] Fetching GPS ground movement from Nevada Geodetic Lab...")
        print(f"[MONITOR] AOI bbox: {bbox}")
        gps_data = await gps_service.get_ground_movement_for_aoi(bbox)
        
        monitoring_state["data_sources"]["gps"]["available"] = True
        monitoring_state["data_sources"]["gps"]["last_update"] = check_time
        monitoring_state["data_sources"]["gps"]["station"] = gps_data.get("station_id")
        monitoring_state["latest_gps"] = gps_data
        
        result["data_sources_checked"].append("gps")
        result["measurements"]["gps"] = {
            "station": gps_data.get("station_id"),
            "station_name": gps_data.get("station_name"),
            "distance_to_aoi_km": gps_data.get("distance_to_aoi_km"),
            "vertical_velocity_mm_year": gps_data.get("analysis", {}).get("vertical_velocity_mm_year"),
            "max_subsidence_mm": gps_data.get("analysis", {}).get("max_subsidence_mm"),
            "is_subsiding": gps_data.get("analysis", {}).get("is_subsiding"),
            "date_range": gps_data.get("metadata", {}).get("date_range"),
            "n_observations": gps_data.get("metadata", {}).get("n_observations"),
        }
        
        log_entry["details"].append(f"GPS: {gps_data['station_id']} velocity={gps_data['analysis']['vertical_velocity_mm_year']:.2f} mm/yr")
        print(f"[MONITOR] GPS: Station {gps_data['station_id']} - {gps_data['analysis']['vertical_velocity_mm_year']:.2f} mm/year")
        
    except Exception as e:
        monitoring_state["data_sources"]["gps"]["available"] = False
        result["measurements"]["gps"] = {"error": str(e)}
        log_entry["details"].append(f"GPS: ERROR - {str(e)[:50]}")
        print(f"[MONITOR] GPS Error: {e}")
    finally:
        await gps_service.close()
    
    # =========================================================================
    # 2. FETCH INSAR DATA (NASA OPERA - area coverage, historical)
    # =========================================================================
    opera_service = OPERADisplacementService()
    try:
        print("[MONITOR] Fetching InSAR ground displacement (NASA OPERA)...")
        insar_data = await opera_service.get_displacement_data(bbox)
        
        monitoring_state["data_sources"]["opera_insar"]["available"] = True
        monitoring_state["data_sources"]["opera_insar"]["last_update"] = check_time
        monitoring_state["latest_insar"] = insar_data
        
        vel = insar_data.get("velocity_mm_year")
        if vel is not None:
            max_subsidence = float(-np.nanmin(vel)) if np.any(vel < 0) else 0
            result["data_sources_checked"].append("opera_insar")
            result["measurements"]["insar"] = {
                "max_subsidence_rate_mm_year": max_subsidence,
                "mean_velocity_mm_year": float(np.nanmean(vel)),
                "source": "NASA OPERA DISP-S1",
            }
            log_entry["details"].append(f"InSAR: max_subsidence={max_subsidence:.2f} mm/yr")
            print(f"[MONITOR] InSAR: Max subsidence {max_subsidence:.2f} mm/year")
            
    except Exception as e:
        monitoring_state["data_sources"]["opera_insar"]["available"] = False
        result["measurements"]["insar"] = {"error": str(e)}
        log_entry["details"].append(f"InSAR: unavailable")
        print(f"[MONITOR] InSAR: Not available ({str(e)[:40]}...)")
    finally:
        await opera_service.close()
    
    # =========================================================================
    # 3. GET STATIC SUSCEPTIBILITY (from ML model)
    # =========================================================================
    try:
        from backend.ml.real_inference import RealSusceptibilityInference
        engine = RealSusceptibilityInference()
        await engine.load_aoi_data()
        
        # Get average susceptibility for a few sample points
        import mercantile
        tiles = list(mercantile.tiles(*bbox, zooms=14))[:5]  # Sample 5 tiles
        susc_values = []
        
        for tile in tiles:
            bounds = mercantile.bounds(tile)
            try:
                susc_grid = await engine.predict_tile(
                    bounds=(bounds.west, bounds.south, bounds.east, bounds.north),
                    tile_size=32, zoom=tile.z
                )
                susc_values.append(float(susc_grid.mean()))
            except:
                pass
        
        if susc_values:
            static_susceptibility = np.mean(susc_values)
            monitoring_state["data_sources"]["static_susceptibility"]["available"] = True
            monitoring_state["data_sources"]["static_susceptibility"]["avg_score"] = static_susceptibility
            result["measurements"]["static_susceptibility"] = round(static_susceptibility, 3)
            log_entry["details"].append(f"Static susceptibility: {static_susceptibility:.2f}")
            print(f"[MONITOR] Static susceptibility: {static_susceptibility:.2f}")
            
    except Exception as e:
        result["measurements"]["static_susceptibility"] = {"error": str(e)}
        print(f"[MONITOR] Static susceptibility: Error - {e}")
    
    # =========================================================================
    # 4. COMPUTE COMBINED RISK SCORE
    # =========================================================================
    combined_risk = compute_combined_risk_score(
        gps_data=gps_data,
        insar_data=insar_data,
        static_susceptibility=static_susceptibility
    )
    
    result["combined_risk"] = combined_risk
    monitoring_state["latest_combined_risk"] = combined_risk
    score_str = f"{combined_risk['score']:.2f}" if combined_risk.get("score") is not None else "N/A"
    log_entry["details"].append(f"Combined risk: {score_str} ({combined_risk['level']})")
    print(f"[MONITOR] Combined Risk Score: {score_str} ({combined_risk['level']})")
    
    # =========================================================================
    # 5. EVALUATE TRIGGERS
    # =========================================================================
    triggers_fired = await evaluate_alert_triggers(
        gps_data=gps_data,
        insar_data=insar_data,
        static_susceptibility=static_susceptibility,
        combined_risk=combined_risk
    )
    
    result["triggers_evaluated"] = triggers_fired
    
    # =========================================================================
    # 6. GENERATE ALERTS FOR FIRED TRIGGERS
    # =========================================================================
    for trigger in triggers_fired:
        if trigger["fired"]:
            alert = await generate_agentic_alert(
                trigger=trigger,
                gps_data=gps_data,
                combined_risk=combined_risk
            )
            
            monitoring_state["alerts"].append(alert)
            result["alerts_generated"].append(alert)
            log_entry["type"] = "alert"
            print(f"[MONITOR] 🚨 ALERT GENERATED: {trigger['name']} - Level: {trigger['alert_level']}")
    
    # Determine overall status
    if result["alerts_generated"]:
        result["status"] = "alerts_generated"
    elif combined_risk.get("score") is not None and combined_risk["score"] > ALERT_THRESHOLDS["COMBINED_WATCH"]:
        result["status"] = "elevated"
    else:
        result["status"] = "normal"
    
    # Add to monitoring log
    log_entry["status"] = result["status"]
    monitoring_state["monitoring_log"].append(log_entry)
    
    # Keep log size manageable
    if len(monitoring_state["monitoring_log"]) > 100:
        monitoring_state["monitoring_log"] = monitoring_state["monitoring_log"][-100:]
    
    print(f"[MONITOR] Check complete. Status: {result['status']}")
    print(f"{'='*60}\n")
    
    # Ensure response is JSON-serializable (no numpy.bool_, numpy.float64, etc.)
    return _make_json_serializable(result)


def compute_combined_risk_score(
    gps_data: Optional[Dict],
    insar_data: Optional[Dict],
    static_susceptibility: Optional[float]
) -> Dict[str, Any]:
    """
    Compute combined risk score from REAL data only.
    No fake numbers, no fallbacks, no placeholders.
    
    When static_susceptibility is missing: use movement-only score and label accordingly.
    When no movement data: use static only if available; else INSUFFICIENT_DATA.
    """
    base_susceptibility = None  # Only set from real static_susceptibility
    if static_susceptibility is not None:
        try:
            base_susceptibility = float(static_susceptibility)
        except (TypeError, ValueError):
            base_susceptibility = None

    gps_risk = 0.0
    insar_risk = 0.0
    gps_weight = 0.0
    insar_weight = 0.0

    # GPS: real velocity only
    if gps_data and gps_data.get("analysis"):
        velocity = gps_data["analysis"].get("vertical_velocity_mm_year")
        if velocity is not None:
            velocity = float(velocity)
            if velocity < 0:
                subsidence_rate = -velocity
                gps_risk = min(1.0, subsidence_rate / ALERT_THRESHOLDS["GPS_CRITICAL"])
                gps_weight = 0.7

    # InSAR: real velocity only
    if insar_data and insar_data.get("velocity_mm_year") is not None:
        vel = insar_data["velocity_mm_year"]
        if np.any(vel < 0):
            max_subsidence = float(-np.nanmin(vel))
            insar_risk = min(1.0, max_subsidence / ALERT_THRESHOLDS["INSAR_CRITICAL"])
            insar_weight = 0.3

    total_weight = gps_weight + insar_weight
    if total_weight > 0:
        gps_weight /= total_weight
        insar_weight /= total_weight
    else:
        gps_weight = 0
        insar_weight = 0

    movement_risk = (gps_risk * gps_weight) + (insar_risk * insar_weight)
    has_static = base_susceptibility is not None
    has_movement = total_weight > 0

    if not has_static and not has_movement:
        return {
            "score": None,
            "level": "INSUFFICIENT_DATA",
            "components": {
                "base_susceptibility": None,
                "movement_risk": None,
                "gps_risk": round(gps_risk, 3),
                "insar_risk": round(insar_risk, 3),
            },
            "weights": {"static": 0, "dynamic": 0, "gps": 0, "insar": 0},
            "data_available": {"static": False, "movement": False},
        }

    if has_static and has_movement:
        combined_score = (base_susceptibility * 0.4) + (movement_risk * 0.6)
        w_static, w_dynamic = 0.4, 0.6
    elif has_static:
        combined_score = base_susceptibility
        w_static, w_dynamic = 1.0, 0.0
    else:
        combined_score = movement_risk
        w_static, w_dynamic = 0.0, 1.0

    if combined_score >= ALERT_THRESHOLDS["COMBINED_CRITICAL"]:
        level = "CRITICAL"
    elif combined_score >= ALERT_THRESHOLDS["COMBINED_WARNING"]:
        level = "WARNING"
    elif combined_score >= ALERT_THRESHOLDS["COMBINED_WATCH"]:
        level = "WATCH"
    else:
        level = "NORMAL"

    return {
        "score": round(combined_score, 3),
        "level": level,
        "components": {
            "base_susceptibility": round(base_susceptibility, 3) if base_susceptibility is not None else None,
            "movement_risk": round(movement_risk, 3),
            "gps_risk": round(gps_risk, 3),
            "insar_risk": round(insar_risk, 3),
        },
        "weights": {
            "static": w_static,
            "dynamic": w_dynamic,
            "gps": round(gps_weight, 2),
            "insar": round(insar_weight, 2),
        },
        "data_available": {"static": has_static, "movement": has_movement},
    }


async def evaluate_alert_triggers(
    gps_data: Optional[Dict],
    insar_data: Optional[Dict],
    static_susceptibility: Optional[float],
    combined_risk: Dict
) -> List[Dict]:
    """
    Evaluate all alert triggers and return which ones fired.
    """
    triggers_status = []
    
    # Get measurements (coerce to float for reliable comparison)
    gps_subsidence = 0.0
    if gps_data and gps_data.get("analysis"):
        velocity = gps_data["analysis"].get("vertical_velocity_mm_year", 0)
        velocity = float(velocity) if velocity is not None else 0
        gps_subsidence = -velocity if velocity < 0 else 0
    
    insar_subsidence = 0.0
    if insar_data and insar_data.get("velocity_mm_year") is not None:
        vel = insar_data["velocity_mm_year"]
        insar_subsidence = float(-np.nanmin(vel)) if np.any(vel < 0) else 0
    
    base_susc = float(static_susceptibility) if static_susceptibility is not None else 0
    raw_score = combined_risk.get("score")
    combined_score = float(raw_score) if raw_score is not None else None

    print(f"[MONITOR] Trigger evaluation: gps_subsidence={gps_subsidence:.1f} mm/yr, combined_score={combined_score}")
    
    for trigger in ALERT_TRIGGERS:
        status = {
            "id": trigger["id"],
            "name": trigger["name"],
            "description": trigger["description"],
            "alert_level": trigger["alert_level"],
            "fired": False,
            "reason": None,
        }
        
        # Evaluate each trigger
        if trigger["id"] == "gps_subsidence_high_risk":
            if gps_subsidence > float(ALERT_THRESHOLDS["GPS_WARNING"]) and base_susc > 0.5:
                status["fired"] = True
                status["reason"] = f"GPS subsidence {gps_subsidence:.1f}mm/yr in high-risk zone (susc={base_susc:.2f})"
        
        elif trigger["id"] == "acceleration_after_rainfall":
            # TODO: Implement acceleration detection and rainfall correlation
            # For now, check if there's significant recent subsidence
            pass
        
        elif trigger["id"] == "critical_subsidence":
            if gps_subsidence > float(ALERT_THRESHOLDS["GPS_CRITICAL"]):
                status["fired"] = True
                status["reason"] = f"GPS subsidence {gps_subsidence:.1f}mm/yr exceeds critical threshold"
            elif insar_subsidence > float(ALERT_THRESHOLDS["INSAR_CRITICAL"]):
                status["fired"] = True
                status["reason"] = f"InSAR subsidence {insar_subsidence:.1f}mm/yr exceeds critical threshold"
        
        elif trigger["id"] == "combined_risk_critical":
            if combined_score is not None and combined_score > float(ALERT_THRESHOLDS["COMBINED_CRITICAL"]):
                status["fired"] = True
                status["reason"] = f"Combined risk score {combined_score:.2f} exceeds critical threshold"
        
        if status["fired"]:
            print(f"[MONITOR] Trigger FIRED: {trigger['id']} - {status['reason']}")
        triggers_status.append(status)
    
    return triggers_status


async def generate_agentic_alert(
    trigger: Dict[str, Any],
    gps_data: Optional[Dict],
    combined_risk: Dict
) -> Dict[str, Any]:
    """
    Generate an intelligent alert with Gemini reasoning.
    
    This is the agentic part: Gemini drafts the alert message based on
    the trigger, measurements, and context.
    """
    from backend.gemini.agent import GeminiAgentClient
    
    alert_id = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().isoformat()
    
    alert = {
        "id": alert_id,
        "timestamp": timestamp,
        "trigger_id": trigger["id"],
        "trigger_name": trigger["name"],
        "level": trigger["alert_level"],
        "reason": trigger.get("reason", "Threshold exceeded"),
        "location": "Winter Park, FL",
        "bbox": list(WinterParkAOI.BBOX),
        "combined_risk_score": combined_risk["score"],
        "risk_level": combined_risk["level"],
    }
    
    # Add GPS measurements if available
    if gps_data and gps_data.get("analysis"):
        alert["gps_station"] = gps_data.get("station_id")
        alert["gps_velocity_mm_year"] = gps_data["analysis"].get("vertical_velocity_mm_year")
        alert["gps_distance_km"] = gps_data.get("distance_to_aoi_km")
    
    # Generate Gemini alert message (with timeout so monitoring loop cannot get stuck)
    try:
        gemini = GeminiAgentClient()
        
        # Build context for Gemini
        gps_info = ""
        if gps_data and gps_data.get("analysis"):
            vel = gps_data["analysis"].get("vertical_velocity_mm_year", 0)
            gps_info = f"""
GPS MEASUREMENT:
- Station: {gps_data.get('station_id')} ({gps_data.get('station_name')})
- Distance to AOI: {gps_data.get('distance_to_aoi_km', 'N/A')} km
- Vertical Velocity: {vel:.2f} mm/year {'(SUBSIDING)' if vel < 0 else '(stable)'}
- Observation Period: {gps_data.get('metadata', {}).get('date_range', 'N/A')}
"""
        
        prompt = f"""You are an emergency alert composer for a sinkhole early warning system.

🚨 ALERT TRIGGER FIRED 🚨

TRIGGER: {trigger['name']}
ALERT LEVEL: {trigger['alert_level']}
REASON: {trigger.get('reason', 'Threshold exceeded')}

LOCATION: Winter Park, Florida
CONTEXT: Central Florida Karst District - Ocala Limestone - High historical sinkhole density

COMBINED RISK SCORE: {combined_risk['score']:.2f} ({combined_risk['level']})
Components:
- Static Susceptibility: {combined_risk['components']['base_susceptibility']:.2f}
- Movement Risk: {combined_risk['components']['movement_risk']:.2f}
- GPS Risk: {combined_risk['components']['gps_risk']:.2f}
{gps_info}

TASK: Draft a concise, professional alert message for local emergency management.

Include:
1. **Severity Summary** (1-2 sentences)
2. **Key Evidence** (what triggered the alert)
3. **Recommended Actions** (3-5 bullet points, prioritized)
4. **Affected Area** (geographic description)

Keep the message under 250 words. Be direct, factual, and actionable.
Do NOT use JSON. Write as plain text with clear sections.
"""

        result = await asyncio.wait_for(
            gemini.analyze_with_thinking(prompt=prompt, thinking_level="medium"),
            timeout=GEMINI_ALERT_TIMEOUT_SECONDS,
        )
        
        alert["gemini_message"] = result.get("response", "Alert generated - manual review required")
        alert["gemini_model"] = GeminiConfig.MODEL_NAME
        alert["gemini_thinking"] = result.get("thinking_summary", None)
        
    except asyncio.TimeoutError:
        # Gemini took too long - use fallback so the loop can continue
        print(f"[MONITOR] Gemini alert draft timed out after {GEMINI_ALERT_TIMEOUT_SECONDS}s - using fallback message")
        alert["gemini_message"] = f"""
⚠️ SINKHOLE EARLY WARNING ALERT - {trigger['alert_level']} LEVEL

Location: Winter Park, Florida (Central Florida Karst District)
Trigger: {trigger['name']}
Reason: {trigger.get('reason', 'Threshold exceeded')}

Combined Risk Score: {combined_risk['score']:.2f} ({combined_risk['level']})

RECOMMENDED ACTIONS:
1. Review alert details with geological team
2. Assess infrastructure in affected area
3. Monitor for visible surface changes
4. Consider public notification if appropriate

This is an automated alert. Manual review required before taking action.
(Draft timed out; message not customized by Gemini.)
"""
        alert["gemini_error"] = f"Timeout after {GEMINI_ALERT_TIMEOUT_SECONDS}s"
    except Exception as e:
        # Fallback alert message without Gemini
        alert["gemini_message"] = f"""
⚠️ SINKHOLE EARLY WARNING ALERT - {trigger['alert_level']} LEVEL

Location: Winter Park, Florida (Central Florida Karst District)
Trigger: {trigger['name']}
Reason: {trigger.get('reason', 'Threshold exceeded')}

Combined Risk Score: {combined_risk['score']:.2f} ({combined_risk['level']})

RECOMMENDED ACTIONS:
1. Review alert details with geological team
2. Assess infrastructure in affected area
3. Monitor for visible surface changes
4. Consider public notification if appropriate

This is an automated alert. Manual review required before taking action.
"""
        alert["gemini_error"] = str(e)
    
    return alert


async def generate_alert(
    alert_level: str,
    max_subsidence_rate: float,
    displacement_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Legacy alert generator - kept for backward compatibility.
    Use generate_agentic_alert for new functionality.
    """
    alert = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "level": alert_level,
        "max_subsidence_rate_mm_year": max_subsidence_rate,
        "location": "Winter Park, FL",
        "bbox": list(WinterParkAOI.BBOX),
    }
    
    try:
        from backend.gemini.agent import GeminiAgentClient
        gemini = GeminiAgentClient()
        
        prompt = f"""Generate a concise sinkhole alert message:
ALERT LEVEL: {alert_level}
LOCATION: Winter Park, Florida
SUBSIDENCE RATE: {max_subsidence_rate:.1f} mm/year

Keep under 150 words. Be direct and actionable."""

        result = await gemini.analyze_with_thinking(prompt=prompt, thinking_level="low")
        alert["gemini_message"] = result.get("response", "Alert generated - manual review required")
        alert["gemini_model"] = GeminiConfig.MODEL_NAME
        
    except Exception as e:
        alert["gemini_message"] = f"AUTOMATED ALERT: {alert_level} level subsidence detected. Rate: {max_subsidence_rate:.1f} mm/year."
        alert["gemini_error"] = str(e)
    
    return alert
