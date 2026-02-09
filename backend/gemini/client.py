"""
Gemini API client for sinkhole feature detection

Uses Google's Gemini model for:
- Detecting sinkhole-like features in satellite imagery
- Extracting structured risk factors from imagery + context
- Quality control of ML predictions
"""

import asyncio
import base64
import io
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings, GeminiConfig
from backend.gemini.prompts import (
    FEATURE_DETECTION_PROMPT,
    RISK_FACTORS_PROMPT,
    QA_PROMPT,
)


class GeminiClient:
    """
    Client for Gemini API for sinkhole analysis.
    Uses Gemini 3 via Vertex AI when USE_VERTEX_AI=true, else AI Studio.
    
    Provides methods for:
    - detect_features(): Get bounding boxes of sinkhole-like features
    - extract_risk_factors(): Get structured JSON of risk indicators
    - quality_check(): Compare model output with visual cues
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        self._client = None
        self._model = None
        self._genai_client = None  # Vertex AI client (google.genai)
        self._model_name = GeminiConfig.MODEL_NAME
        self._is_vertex = False
        self._init_client()
    
    def _init_client(self):
        """Initialize Gemini 3 client - Vertex AI or AI Studio"""
        if settings.use_vertex_ai and settings.google_cloud_project:
            try:
                from google import genai
                self._genai_client = genai.Client(
                    vertexai=True,
                    project=settings.google_cloud_project,
                    location="global",
                )
                self._model_name = GeminiConfig.MODEL_NAME
                self._is_vertex = True
                print(f"✓ Gemini client initialized via Vertex AI ({self._model_name})")
                return
            except Exception as e:
                print(f"⚠ Vertex AI init failed: {e}, falling back to AI Studio")
        if not self.api_key:
            print("⚠ Gemini API key not configured")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(
                model_name=GeminiConfig.MODEL_NAME,
                generation_config={
                    "temperature": GeminiConfig.TEMPERATURE,
                    "max_output_tokens": GeminiConfig.MAX_OUTPUT_TOKENS,
                }
            )
            print(f"✓ Gemini client initialized via AI Studio ({GeminiConfig.MODEL_NAME})")
        except Exception as e:
            print(f"⚠ Failed to initialize Gemini: {e}")
            self._model = None
    
    @property
    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return self._model is not None or (self._is_vertex and self._genai_client is not None)
    
    async def _generate_content_text(self, prompt: str, image_b64: Optional[str] = None) -> str:
        """Call Gemini (Vertex or AI Studio) and return response text."""
        if self._is_vertex and self._genai_client:
            from google.genai import types
            content_parts = [prompt]
            if image_b64:
                image_bytes = base64.b64decode(image_b64)
                content_parts.insert(0, types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
            response = await asyncio.to_thread(
                self._genai_client.models.generate_content,
                model=self._model_name,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    temperature=GeminiConfig.TEMPERATURE,
                    max_output_tokens=GeminiConfig.MAX_OUTPUT_TOKENS,
                ),
            )
            return response.text
        content = [{"mime_type": "image/png", "data": image_b64}, prompt] if image_b64 else [prompt]
        response = await asyncio.to_thread(self._model.generate_content, content)
        return response.text
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array to base64 encoded PNG"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    @retry(
        stop=stop_after_attempt(GeminiConfig.MAX_RETRIES),
        wait=wait_exponential(multiplier=GeminiConfig.RETRY_DELAY)
    )
    async def detect_features(
        self,
        image: np.ndarray,
        bounds: tuple,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect sinkhole-like features in an image
        
        Uses Gemini's bounding box detection capability to identify:
        - Circular depressions
        - Dolines
        - Collapse features
        - Subsidence areas
        
        Args:
            image: RGB image array (H, W, 3) or grayscale (H, W)
            bounds: Geographic bounds (west, south, east, north)
            context: Optional context (geology, nearby features, etc.)
        
        Returns:
            GeoJSON FeatureCollection with detected features
        """
        if not self.is_available:
            return {"type": "FeatureCollection", "features": [], "error": "Gemini not available"}
        
        # Prepare image
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        image_b64 = self._image_to_base64(image)
        
        # Build prompt with context
        prompt = FEATURE_DETECTION_PROMPT.format(
            bounds=bounds,
            context=context or {},
            image_width=image.shape[1],
            image_height=image.shape[0],
        )
        
        try:
            response_text = await self._generate_content_text(prompt, image_b64)
            return self._parse_feature_response(response_text, bounds, image.shape)
            
        except Exception as e:
            print(f"Gemini feature detection error: {e}")
            return {"type": "FeatureCollection", "features": [], "error": str(e)}
    
    def _parse_feature_response(
        self,
        response_text: str,
        bounds: tuple,
        image_shape: tuple
    ) -> Dict[str, Any]:
        """Parse Gemini response into GeoJSON"""
        import json
        import re
        
        features = []
        west, south, east, north = bounds
        height, width = image_shape[:2]
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                
                # Handle different response formats
                detections = data.get('features', data.get('detections', []))
                
                for det in detections:
                    # Get bounding box (normalized or pixel coordinates)
                    bbox = det.get('bbox', det.get('bounding_box', det.get('box_2d', [])))
                    
                    if len(bbox) >= 4:
                        # Convert to geographic coordinates
                        # Assuming bbox is [x1, y1, x2, y2] in pixel coords
                        x1, y1, x2, y2 = bbox[:4]
                        
                        # Normalize if needed
                        if max(bbox) > 1:
                            x1, x2 = x1 / width, x2 / width
                            y1, y2 = y1 / height, y2 / height
                        
                        # Convert to geographic
                        lon1 = west + x1 * (east - west)
                        lon2 = west + x2 * (east - west)
                        lat1 = north - y1 * (north - south)  # Y is flipped
                        lat2 = north - y2 * (north - south)
                        
                        features.append({
                            "type": "Feature",
                            "properties": {
                                "feature_type": det.get('type', det.get('label', 'unknown')),
                                "confidence": det.get('confidence', det.get('score', 0.5)),
                                "source": "gemini",
                                "description": det.get('description', ''),
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [lon1, lat1],
                                    [lon2, lat1],
                                    [lon2, lat2],
                                    [lon1, lat2],
                                    [lon1, lat1],
                                ]]
                            }
                        })
                
            except json.JSONDecodeError:
                pass
        
        return {
            "type": "FeatureCollection",
            "properties": {
                "source": "gemini",
                "model": GeminiConfig.MODEL_NAME,
            },
            "features": features
        }
    
    @retry(
        stop=stop_after_attempt(GeminiConfig.MAX_RETRIES),
        wait=wait_exponential(multiplier=GeminiConfig.RETRY_DELAY)
    )
    async def extract_risk_factors(
        self,
        image: np.ndarray,
        geology_context: Dict[str, Any],
        bounds: tuple
    ) -> Dict[str, Any]:
        """
        Extract structured risk factors from imagery and context
        
        Returns JSON with:
        - karst_indicators: presence of karst features
        - drainage_anomalies: unusual drainage patterns
        - vegetation_stress: vegetation health indicators
        - lineaments: linear features that may indicate faults
        - confidence scores for each factor
        
        Args:
            image: RGB satellite image
            geology_context: Local geology information
            bounds: Geographic bounds
        
        Returns:
            Structured risk factor dictionary
        """
        if not self.is_available:
            return {"error": "Gemini not available", "factors": {}}
        
        image_b64 = self._image_to_base64(image)
        
        prompt = RISK_FACTORS_PROMPT.format(
            geology=geology_context,
            bounds=bounds,
        )
        
        try:
            response_text = await self._generate_content_text(prompt, image_b64)
            return self._parse_risk_factors(response_text)
        except Exception as e:
            print(f"Gemini risk factor extraction error: {e}")
            return {"error": str(e), "factors": {}}
    
    def _parse_risk_factors(self, response_text: str) -> Dict[str, Any]:
        """Parse risk factors response"""
        import json
        import re
        
        # Default structure
        factors = {
            "karst_indicators": {"present": False, "confidence": 0.0, "details": []},
            "drainage_anomalies": {"present": False, "confidence": 0.0, "details": []},
            "vegetation_stress": {"present": False, "confidence": 0.0, "details": []},
            "lineaments": {"present": False, "confidence": 0.0, "details": []},
            "overall_risk": "low",
            "summary": "",
        }
        
        # Try to extract JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                # Merge with defaults
                for key in factors:
                    if key in parsed:
                        factors[key] = parsed[key]
            except json.JSONDecodeError:
                factors["summary"] = response_text[:500]
        else:
            factors["summary"] = response_text[:500]
        
        return {"factors": factors}
    
    @retry(
        stop=stop_after_attempt(GeminiConfig.MAX_RETRIES),
        wait=wait_exponential(multiplier=GeminiConfig.RETRY_DELAY)
    )
    async def quality_check(
        self,
        image: np.ndarray,
        model_prediction: np.ndarray,
        bounds: tuple
    ) -> Dict[str, Any]:
        """
        Quality check ML model predictions against visual cues
        
        Flags tiles where model predictions conflict with obvious visual features
        
        Args:
            image: RGB satellite image
            model_prediction: ML model susceptibility output
            bounds: Geographic bounds
        
        Returns:
            QA results with flags and recommendations
        """
        if not self.is_available:
            return {"error": "Gemini not available", "qa_passed": True}
        
        # Create visualization of model output
        prediction_viz = self._create_prediction_overlay(image, model_prediction)
        
        image_b64 = self._image_to_base64(prediction_viz)
        
        prompt = QA_PROMPT.format(bounds=bounds)
        
        try:
            response_text = await self._generate_content_text(prompt, image_b64)
            return self._parse_qa_response(response_text)
        except Exception as e:
            print(f"Gemini QA error: {e}")
            return {"error": str(e), "qa_passed": True}
    
    def _create_prediction_overlay(
        self,
        image: np.ndarray,
        prediction: np.ndarray
    ) -> np.ndarray:
        """Create side-by-side or overlay visualization"""
        # Ensure image is RGB uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Create heatmap from prediction
        prediction = np.clip(prediction, 0, 1)
        heatmap = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        
        # Color mapping
        heatmap[:, :, 0] = (prediction * 255).astype(np.uint8)  # Red channel
        heatmap[:, :, 1] = ((1 - prediction) * 255).astype(np.uint8)  # Green channel
        
        # Resize if needed
        if image.shape[:2] != prediction.shape:
            from PIL import Image as PILImage
            heatmap_pil = PILImage.fromarray(heatmap)
            heatmap_pil = heatmap_pil.resize((image.shape[1], image.shape[0]))
            heatmap = np.array(heatmap_pil)
        
        # Blend
        alpha = 0.5
        overlay = (image * (1 - alpha) + heatmap * alpha).astype(np.uint8)
        
        return overlay
    
    def _parse_qa_response(self, response_text: str) -> Dict[str, Any]:
        """Parse QA response"""
        import json
        import re
        
        result = {
            "qa_passed": True,
            "flags": [],
            "recommendations": [],
            "summary": "",
        }
        
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                result.update(parsed)
            except json.JSONDecodeError:
                result["summary"] = response_text[:500]
        else:
            result["summary"] = response_text[:500]
            # Simple heuristic: if "conflict" or "inconsistent" in response, flag it
            lower_text = response_text.lower()
            if "conflict" in lower_text or "inconsistent" in lower_text:
                result["qa_passed"] = False
                result["flags"].append("potential_conflict_detected")
        
        return result


async def batch_detect_features(
    tiles: List[Dict[str, Any]],
    client: Optional[GeminiClient] = None
) -> List[Dict[str, Any]]:
    """
    Batch process multiple tiles for feature detection
    
    Respects rate limits and batches requests efficiently
    
    Args:
        tiles: List of dicts with 'image', 'bounds', 'context'
        client: Optional GeminiClient instance
    
    Returns:
        List of GeoJSON FeatureCollections
    """
    if client is None:
        client = GeminiClient()
    
    if not client.is_available:
        return [{"type": "FeatureCollection", "features": [], "error": "Gemini not available"}
                for _ in tiles]
    
    results = []
    
    # Process in batches
    batch_size = GeminiConfig.MAX_TILES_PER_BATCH
    
    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i + batch_size]
        
        # Process batch concurrently
        tasks = [
            client.detect_features(
                tile['image'],
                tile['bounds'],
                tile.get('context')
            )
            for tile in batch
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                results.append({"type": "FeatureCollection", "features": [], "error": str(result)})
            else:
                results.append(result)
        
        # Rate limiting delay between batches
        if i + batch_size < len(tiles):
            await asyncio.sleep(1.0)
    
    return results

