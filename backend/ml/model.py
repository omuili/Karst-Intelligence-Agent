"""
XGBoost model for sinkhole susceptibility prediction
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import joblib

from backend.config import ModelConfig, settings


@dataclass
class ModelMetrics:
    """Metrics for model evaluation"""
    auc_roc: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    confusion_matrix: np.ndarray


class SinkholeModel:
    """
    XGBoost classifier for sinkhole susceptibility prediction
    
    Trained on labeled grid cells with features from:
    - Satellite imagery
    - Terrain/DEM derivatives
    - Geology layers
    - Hydrology data
    """
    
    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build list of feature names"""
        self.feature_names = (
            ModelConfig.SPECTRAL_FEATURES +
            ModelConfig.TERRAIN_FEATURES +
            ModelConfig.GEOLOGY_FEATURES +
            ModelConfig.HYDROLOGY_FEATURES
        )
    
    def build(self):
        """Build the XGBoost model"""
        from xgboost import XGBClassifier
        
        self.model = XGBClassifier(
            **ModelConfig.XGBOOST_PARAMS,
            scale_pos_weight=ModelConfig.POSITIVE_CLASS_WEIGHT,
            use_label_encoder=False,
            eval_metric='auc',
        )
        
        return self
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True
    ):
        """
        Train the model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0 = no sinkhole, 1 = sinkhole)
            eval_set: Optional validation set (X_val, y_val)
            verbose: Print training progress
        """
        if self.model is None:
            self.build()
        
        fit_params = {
            'verbose': verbose,
        }
        
        if eval_set is not None:
            X_val, y_val = eval_set
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['early_stopping_rounds'] = 20
        
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> ModelMetrics:
        """
        Evaluate model performance
        
        Returns:
            ModelMetrics with various evaluation metrics
        """
        from sklearn.metrics import (
            roc_auc_score,
            precision_score,
            recall_score,
            f1_score,
            accuracy_score,
            confusion_matrix
        )
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        return ModelMetrics(
            auc_roc=roc_auc_score(y, y_proba),
            precision=precision_score(y, y_pred),
            recall=recall_score(y, y_pred),
            f1_score=f1_score(y, y_pred),
            accuracy=accuracy_score(y, y_pred),
            confusion_matrix=confusion_matrix(y, y_pred),
        )
    
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def save(self, path: Optional[Path] = None):
        """Save model to disk"""
        if path is None:
            path = settings.base_dir / settings.ml_model_path
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load(self, path: Optional[Path] = None):
        """Load model from disk"""
        if path is None:
            path = settings.base_dir / settings.ml_model_path
        
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {path}")
        return self


def create_training_data(
    sinkhole_points: np.ndarray,
    feature_rasters: Dict[str, np.ndarray],
    transform,
    negative_sample_ratio: float = 5.0,
    buffer_cells: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training data from sinkhole locations and feature rasters
    
    Args:
        sinkhole_points: Array of (x, y) coordinates of known sinkholes
        feature_rasters: Dictionary of feature name -> raster array
        transform: Affine transform from coordinates to pixel indices
        negative_sample_ratio: Ratio of negative to positive samples
        buffer_cells: Buffer around sinkholes to exclude from negatives
        random_state: Random seed
    
    Returns:
        Tuple of (X, y) for training
    """
    rng = np.random.RandomState(random_state)
    
    # Get raster shape from first feature
    first_raster = list(feature_rasters.values())[0]
    height, width = first_raster.shape
    
    # Convert sinkhole points to pixel coordinates
    positive_pixels = set()
    buffer_pixels = set()
    
    for x, y in sinkhole_points:
        # Convert coordinate to pixel
        col = int((x - transform.c) / transform.a)
        row = int((y - transform.f) / transform.e)
        
        if 0 <= row < height and 0 <= col < width:
            positive_pixels.add((row, col))
            
            # Add buffer zone
            for dr in range(-buffer_cells, buffer_cells + 1):
                for dc in range(-buffer_cells, buffer_cells + 1):
                    r, c = row + dr, col + dc
                    if 0 <= r < height and 0 <= c < width:
                        buffer_pixels.add((r, c))
    
    # Create positive samples
    X_pos = []
    for row, col in positive_pixels:
        features = [feature_rasters[name][row, col] for name in feature_rasters]
        X_pos.append(features)
    
    X_pos = np.array(X_pos)
    y_pos = np.ones(len(X_pos))
    
    # Create negative samples (outside buffer zones)
    n_negative = int(len(positive_pixels) * negative_sample_ratio)
    valid_negative_pixels = []
    
    for row in range(height):
        for col in range(width):
            if (row, col) not in buffer_pixels:
                valid_negative_pixels.append((row, col))
    
    # Random sample of negatives
    neg_indices = rng.choice(len(valid_negative_pixels), size=n_negative, replace=False)
    negative_pixels = [valid_negative_pixels[i] for i in neg_indices]
    
    X_neg = []
    for row, col in negative_pixels:
        features = [feature_rasters[name][row, col] for name in feature_rasters]
        X_neg.append(features)
    
    X_neg = np.array(X_neg)
    y_neg = np.zeros(len(X_neg))
    
    # Combine
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])
    
    # Shuffle
    shuffle_idx = rng.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


def train_model_pipeline(
    sinkhole_gdf,
    feature_rasters: Dict[str, np.ndarray],
    transform,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[SinkholeModel, ModelMetrics]:
    """
    Complete training pipeline
    
    Args:
        sinkhole_gdf: GeoDataFrame with sinkhole point geometry
        feature_rasters: Dictionary of feature rasters
        transform: Raster affine transform
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        Trained model and test metrics
    """
    from sklearn.model_selection import train_test_split
    
    # Extract coordinates
    points = np.array([(p.x, p.y) for p in sinkhole_gdf.geometry])
    
    # Create training data
    X, y = create_training_data(
        points, feature_rasters, transform,
        random_state=random_state
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train model
    model = SinkholeModel()
    model.build()
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model.save()
    
    print(f"\nModel Training Complete")
    print(f"  AUC-ROC: {metrics.auc_roc:.3f}")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  F1 Score: {metrics.f1_score:.3f}")
    
    return model, metrics

