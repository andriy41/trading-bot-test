# models/ensemble.py
from typing import List, Dict, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass

@dataclass
class EnsembleResult:
    predictions: np.ndarray
    probabilities: np.ndarray
    model_weights: Dict[str, float]
    confidence_scores: np.ndarray
    feature_importance: Dict[str, float]

class TradingEnsemble:
    def __init__(
        self,
        feature_columns: List[str],
        target_column: str,
        models_config: Dict = None
    ):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.models = self._initialize_models(models_config)
        self.scaler = StandardScaler()
        self.model_weights = {}
        self.feature_importance = {}
        
    def _initialize_models(self, config: Dict = None) -> Dict:
        """
        Initialize ensemble models with optimized configurations.
        """
        default_config = {
            'rf': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 10
            },
            'gb': {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 5
            },
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'probability': True
            },
            'nn': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'learning_rate': 'adaptive'
            }
        }
        
        config = config or default_config
        
        return {
            'rf': RandomForestClassifier(**config['rf']),
            'gb': GradientBoostingClassifier(**config['gb']),
            'svm': SVC(**config['svm']),
            'nn': MLPClassifier(**config['nn'])
        }
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train ensemble with dynamic weight adjustment.
        """
        X_scaled = self.scaler.fit_transform(X)
        
        # Train individual models and evaluate performance
        model_scores = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            model_scores[name] = np.mean(scores)
            model.fit(X_scaled, y)
            
        # Calculate model weights based on performance
        total_score = sum(model_scores.values())
        self.model_weights = {
            name: score/total_score for name, score in model_scores.items()
        }
        
        # Calculate feature importance
        self._calculate_feature_importance(X, y)
        
    def predict(self, X: pd.DataFrame) -> EnsembleResult:
        """
        Generate weighted ensemble predictions with confidence scores.
        """
        X_scaled = self.scaler.transform(X)
        predictions = []
        probabilities = []
        
        # Get predictions from each model
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            prob = model.predict_proba(X_scaled)
            
            predictions.append(pred)
            probabilities.append(prob)
            
        # Combine predictions using weighted voting
        weighted_predictions = np.zeros(len(X))
        weighted_probabilities = np.zeros((len(X), 2))
        
        for i, (name, weight) in enumerate(self.model_weights.items()):
            weighted_predictions += weight * predictions[i]
            weighted_probabilities += weight * probabilities[i]
            
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence(weighted_probabilities)
        
        return EnsembleResult(
            predictions=weighted_predictions,
            probabilities=weighted_probabilities,
            model_weights=self.model_weights,
            confidence_scores=confidence_scores,
            feature_importance=self.feature_importance
        )
        
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """
        Calculate aggregated feature importance across models.
        """
        importance_scores = {}
        
        # Get feature importance from tree-based models
        rf_importance = self.models['rf'].feature_importances_
        gb_importance = self.models['gb'].feature_importances_
        
        # Calculate SVM feature importance using weights
        svm_importance = np.abs(self.models['svm'].coef_[0])
        
        # Normalize and combine importance scores
        for i, feature in enumerate(self.feature_columns):
            importance_scores[feature] = (
                self.model_weights['rf'] * rf_importance[i] +
                self.model_weights['gb'] * gb_importance[i] +
                self.model_weights['svm'] * svm_importance[i]
            )
            
        self.feature_importance = importance_scores
        
    def _calculate_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate prediction confidence scores.
        """
        # Maximum probability difference as confidence measure
        confidence = np.max(probabilities, axis=1) - np.min(probabilities, axis=1)
        
        # Adjust confidence based on ensemble agreement
        model_agreement = np.std(probabilities, axis=0)
        confidence *= (1 - np.mean(model_agreement))
        
        return confidence
        
    def update_weights(self, performance_metrics: Dict[str, float]):
        """
        Dynamically update model weights based on recent performance.
        """
        total_performance = sum(performance_metrics.values())
        
        self.model_weights = {
            name: perf/total_performance 
            for name, perf in performance_metrics.items()
        }
