"""
Anomaly Detection Module
For identifying unusual transaction and behavioral patterns
"""

import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Anomaly detection for identifying suspicious patterns
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Anomaly Detector
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        self.model = IsolationForest(
            contamination=self.config.get('contamination', 0.1),
            random_state=42
        )
        self.is_trained = False
    
    def fit(self, X: np.ndarray) -> Dict:
        """
        Fit the anomaly detection model
        
        Args:
            X: Feature matrix
            
        Returns:
            Fitting metrics
        """
        logger.info("Fitting anomaly detection model...")
        
        self.model.fit(X)
        self.is_trained = True
        
        anomaly_count = sum(self.model.predict(X) == -1)
        logger.info(f"Detected {anomaly_count} anomalies in training data")
        
        return {'anomalies_detected': anomaly_count}
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (-1 for anomaly, 1 for normal) and anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        return predictions, scores
