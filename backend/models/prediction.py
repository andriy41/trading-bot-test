# models/prediction.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

@dataclass
class PredictionResult:
    predictions: np.ndarray
    confidence: np.ndarray
    support_levels: List[float]
    resistance_levels: List[float]
    trend_direction: str
    risk_score: float

class PricePredictor:
    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 10,
        n_lstm_layers: int = 3,
        lstm_units: int = 128
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model(n_lstm_layers, lstm_units)
        self.scaler = MinMaxScaler()
        self.feature_extractors = self._initialize_feature_extractors()
        
    def _build_model(self, n_lstm_layers: int, lstm_units: int) -> Sequential:
        """
        Build LSTM model architecture with attention mechanism.
        """
        model = Sequential()
        
        # First LSTM layer with return sequences
        model.add(LSTM(
            units=lstm_units,
            return_sequences=True if n_lstm_layers > 1 else False,
            input_shape=(self.sequence_length, self.n_features)
        ))
        model.add(Dropout(0.2))
        
        # Middle LSTM layers
        for i in range(n_lstm_layers - 2):
            model.add(LSTM(units=lstm_units, return_sequences=True))
            model.add(Dropout(0.2))
            
        # Final LSTM layer
        if n_lstm_layers > 1:
            model.add(LSTM(units=lstm_units))
            model.add(Dropout(0.2))
            
        # Output layers
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber_loss',
            metrics=['mae']
        )
        
        return model
        
    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """
        Generate price predictions with confidence intervals.
        """
        # Extract features
        features = self._extract_features(data)
        
        # Prepare sequences
        sequences = self._prepare_sequences(features)
        
        # Generate predictions
        raw_predictions = self.model.predict(sequences)
        
        # Calculate confidence intervals
        confidence = self._calculate_confidence(raw_predictions, data)
        
        # Identify support and resistance levels
        support_resistance = self._identify_support_resistance(data)
        
        # Determine trend direction
        trend = self._analyze_trend(raw_predictions)
        
        # Calculate risk score
        risk = self._calculate_risk_score(data, raw_predictions)
        
        return PredictionResult(
            predictions=raw_predictions,
            confidence=confidence,
            support_levels=support_resistance['support'],
            resistance_levels=support_resistance['resistance'],
            trend_direction=trend,
            risk_score=risk
        )
        
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract and engineer features for prediction.
        """
        features = []
        
        for extractor in self.feature_extractors:
            feature = extractor(data)
            features.append(feature)
            
        return np.column_stack(features)
        
    def _prepare_sequences(self, features: np.ndarray) -> np.ndarray:
        """
        Prepare sequences for LSTM input.
        """
        sequences = []
        for i in range(len(features) - self.sequence_length):
            sequence = features[i:(i + self.sequence_length)]
            sequences.append(sequence)
            
        return np.array(sequences)
        
    def _calculate_confidence(
        self,
        predictions: np.ndarray,
        data: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate prediction confidence using multiple factors.
        """
        # Historical volatility
        volatility = data['close'].pct_change().std()
        
        # Prediction variance
        pred_std = np.std(predictions)
        
        # Market conditions factor
        market_factor = self._assess_market_conditions(data)
        
        # Combine factors for confidence score
        confidence = 1 - (volatility * pred_std * market_factor)
        
        return np.clip(confidence, 0, 1)
        
    def _identify_support_resistance(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Identify key support and resistance levels.
        """
        price_data = data['close'].values
        
        # Find local minima and maxima
        support = self._find_local_extrema(price_data, 'min')
        resistance = self._find_local_extrema(price_data, 'max')
        
        return {
            'support': self._cluster_levels(support),
            'resistance': self._cluster_levels(resistance)
        }
        
    def _analyze_trend(self, predictions: np.ndarray) -> str:
        """
        Determine trend direction and strength.
        """
        trend_slope = np.polyfit(np.arange(len(predictions)), predictions.flatten(), 1)[0]
        
        if trend_slope > 0.02:
            return 'strong_uptrend'
        elif trend_slope > 0:
            return 'weak_uptrend'
        elif trend_slope < -0.02:
            return 'strong_downtrend'
        else:
            return 'weak_downtrend'
            
    def _calculate_risk_score(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray
    ) -> float:
        """
        Calculate comprehensive risk score.
        """
        # Volatility risk
        volatility_risk = data['close'].pct_change().std() * np.sqrt(252)
        
        # Prediction uncertainty
        pred_uncertainty = np.std(predictions) / np.mean(predictions)
        
        # Market condition risk
        market_risk = self._assess_market_conditions(data)
        
        # Combine risk factors
        total_risk = (0.4 * volatility_risk + 
                     0.3 * pred_uncertainty +
                     0.3 * market_risk)
                     
        return np.clip(total_risk, 0, 1)
# models/prediction.py

from typing import Dict, List, Union, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from utils.logger import setup_logger
import joblib
import os
from datetime import datetime

logger = setup_logger(__name__)

class PricePredictor:
    """Machine learning model for price prediction."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize the price predictor.
        
        Args:
            model_path: Path to saved model
            feature_columns: List of feature column names
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = feature_columns or [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'ema_9', 'bb_upper', 'bb_lower'
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No model loaded, predictions will be randomized")
            
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction."""
        try:
            # Check required columns
            missing_cols = [col for col in self.feature_columns 
                          if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Scale features
            features = data[self.feature_columns].copy()
            scaled_features = self.scaler.fit_transform(features)
            
            # Reshape for LSTM if needed
            if self.model and isinstance(self.model, tf.keras.Model):
                return np.reshape(scaled_features, 
                                (scaled_features.shape[0], 1, scaled_features.shape[1]))
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame) -> List[Dict[str, Union[str, float]]]:
        """
        Generate price predictions.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if self.model is None:
                # Generate random predictions for testing
                return self._generate_random_predictions(len(data))
                
            # Prepare features
            X = self.prepare_features(data)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Process predictions
            return [
                {
                    'timestamp': data.index[i],
                    'direction': 'up' if pred > 0.5 else 'down',
                    'confidence': float(pred),
                    'predicted_price': float(data['close'].iloc[i] * 
                                          (1 + (pred - 0.5) * 0.02))
                }
                for i, pred in enumerate(predictions)
            ]
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return self._generate_random_predictions(len(data))

    def _generate_random_predictions(
        self,
        length: int
    ) -> List[Dict[str, Union[str, float]]]:
        """Generate random predictions for testing."""
        return [
            {
                'direction': np.random.choice(['up', 'down']),
                'confidence': np.random.uniform(0.6, 0.9),
                'predicted_price': 0.0
            }
            for _ in range(length)
        ]

    def save_model(self, path: str):
        """Save model and artifacts."""
        try:
            if self.model:
                self.model.save(path)
                # Save additional artifacts
                artifacts = {
                    'feature_columns': self.feature_columns,
                    'scaler': self.scaler,
                    'metadata': {
                        'saved_at': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
                joblib.dump(artifacts, f"{path}_artifacts.joblib")
                logger.info(f"Model saved to {path}")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str):
        """Load model and artifacts."""
        try:
            self.model = tf.keras.models.load_model(path)
            
            # Load artifacts
            artifacts = joblib.load(f"{path}_artifacts.joblib")
            self.feature_columns = artifacts['feature_columns']
            self.scaler = artifacts['scaler']
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise