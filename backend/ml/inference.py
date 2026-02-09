"""
Inference pipeline for sinkhole susceptibility prediction
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from backend.config import settings, WinterParkAOI, ModelConfig


class SusceptibilityInference:
    """
    Inference engine for sinkhole susceptibility prediction
    
    Handles:
    - Loading trained XGBoost model
    - Feature extraction from satellite/terrain data
    - Per-tile susceptibility prediction
    - Feature detection (sinkhole candidates)
    """
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.gemini_client = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained XGBoost model if available"""
        model_path = settings.base_dir / settings.ml_model_path
        
        if model_path.exists():
            try:
                import joblib
                self.model = joblib.load(model_path)
                print(f"✓ Loaded model from {model_path}")
            except Exception as e:
                print(f"⚠ Failed to load model: {e}")
                self.model = None
        else:
            print("⚠ No trained model found, using heuristic prediction")
            self.model = None
    
    async def predict_tile(
        self,
        bounds: Tuple[float, float, float, float],
        tile_size: int = 256,
        zoom: int = 14
    ) -> np.ndarray:
        """
        Predict susceptibility for a tile
        
        Args:
            bounds: (west, south, east, north) in WGS84
            tile_size: Output size in pixels
            zoom: Zoom level (affects resolution)
        
        Returns:
            2D numpy array of susceptibility values [0, 1]
        
        Raises:
            RuntimeError: If no trained model is available
        """
        if self.model is None:
            raise RuntimeError("No trained model available. Please train the model first.")
        
        return await self._predict_with_model(bounds, tile_size, zoom)
    
    async def _predict_with_model(
        self,
        bounds: Tuple[float, float, float, float],
        tile_size: int,
        zoom: int
    ) -> np.ndarray:
        """Predict using trained XGBoost model"""
        # Extract features for the tile
        features = await self._extract_features(bounds, tile_size)
        
        # Reshape for prediction (n_samples, n_features)
        n_features = features.shape[2]
        X = features.reshape(-1, n_features)
        
        # Predict probabilities
        proba = self.model.predict_proba(X)[:, 1]
        
        # Reshape back to image
        susceptibility = proba.reshape(tile_size, tile_size)
        
        return susceptibility.astype(np.float32)
    
    async def detect_features(
        self,
        bounds: Tuple[float, float, float, float],
        zoom: int = 14
    ) -> Dict[str, Any]:
        """
        Detect sinkhole-like features in a tile
        
        Uses Gemini for AI-powered detection when available.
        Returns empty collection if Gemini is not configured.
        
        Returns:
            GeoJSON FeatureCollection
        """
        if settings.use_gemini_features and settings.gemini_api_key:
            return await self._detect_with_gemini(bounds, zoom)
        
        # No detection available without Gemini
        return {
            "type": "FeatureCollection",
            "properties": {
                "tile_bounds": list(bounds),
                "zoom": zoom,
                "detection_method": "none",
                "message": "Feature detection requires Gemini API configuration"
            },
            "features": []
        }
    
    async def _detect_with_gemini(
        self,
        bounds: Tuple[float, float, float, float],
        zoom: int
    ) -> Dict[str, Any]:
        """Use Gemini for feature detection with bounding boxes"""
        try:
            from backend.gemini.client import GeminiClient
            
            if self.gemini_client is None:
                self.gemini_client = GeminiClient()
            
            if not self.gemini_client.is_available:
                return {
                    "type": "FeatureCollection",
                    "properties": {
                        "tile_bounds": list(bounds),
                        "zoom": zoom,
                        "error": "Gemini client not available"
                    },
                    "features": []
                }
            
            # TODO: Implement actual satellite imagery fetching and Gemini analysis
            # For now, return empty collection until satellite imagery integration is complete
            return {
                "type": "FeatureCollection",
                "properties": {
                    "tile_bounds": list(bounds),
                    "zoom": zoom,
                    "detection_method": "gemini",
                    "message": "Satellite imagery integration pending"
                },
                "features": []
            }
            
        except Exception as e:
            print(f"Gemini detection failed: {e}")
            return {
                "type": "FeatureCollection",
                "properties": {
                    "tile_bounds": list(bounds),
                    "zoom": zoom,
                    "error": str(e)
                },
                "features": []
            }

