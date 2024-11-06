# models/training.py  

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error
)
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, 
        ModelCheckpoint, 
        ReduceLROnPlateau
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class TrainingResult:
    """Data class to store training results."""
    model: Optional[object]
    history: Dict
    validation_metrics: Dict
    feature_importance: Dict[str, float]
    training_time: float
    best_epoch: int

class ModelTrainer:
    """Class for training deep learning models for time series prediction."""
    
    def __init__(
        self,
        model_config: Dict,
        training_config: Dict,
        feature_columns: List[str]
    ):
        """Initialize the model trainer."""
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required. Install with: pip install tensorflow>=2.8.0"
            )
            
        self.model_config = model_config
        self.training_config = training_config
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.model = None
        
        try:
            self.model = self._build_model()
            logger.info("Model built successfully")
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
        
    def _build_model(self) -> tf.keras.Model:
        """Build and compile the LSTM model architecture."""
        try:
            input_shape = (
                self.model_config.get('sequence_length', 60),
                len(self.feature_columns)
            )
            
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(
                    units=self.model_config.get('lstm_units', 128),
                    return_sequences=True,
                    input_shape=input_shape
                ),
                tf.keras.layers.Dropout(
                    self.model_config.get('dropout_rate', 0.2)
                ),
                tf.keras.layers.LSTM(
                    units=self.model_config.get('lstm_units_2', 64),
                    return_sequences=True
                ),
                tf.keras.layers.Dropout(
                    self.model_config.get('dropout_rate', 0.2)
                ),
                tf.keras.layers.LSTM(
                    units=self.model_config.get('lstm_units_3', 32)
                ),
                tf.keras.layers.Dropout(
                    self.model_config.get('dropout_rate', 0.2)
                ),
                tf.keras.layers.Dense(
                    units=self.model_config.get('dense_units', 16),
                    activation='relu'
                ),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.training_config.get('learning_rate', 0.001)
                ),
                loss=self.training_config.get('loss_function', 'binary_crossentropy'),
                metrics=['accuracy', 'mae']
            )
            
            model.summary()
            return model
            
        except Exception as e:
            logger.error(f"Error in model building: {str(e)}")
            raise
        
    def prepare_data(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and sequence data for training."""
        try:
            # Validate input data
            if data.empty or target.empty:
                raise ValueError("Empty input data")
            
            if len(data) != len(target):
                raise ValueError("Features and target must have same length")
            
            # Scale features
            if fit_scaler:
                X = self.scaler.fit_transform(data[self.feature_columns])
            else:
                X = self.scaler.transform(data[self.feature_columns])
            
            sequence_length = self.model_config.get('sequence_length', 60)
            sequences = []
            targets = []
            
            # Create sequences
            for i in range(len(X) - sequence_length):
                seq = X[i:(i + sequence_length)]
                sequences.append(seq)
                targets.append(target.iloc[i + sequence_length])
            
            if not sequences:
                raise ValueError("No sequences could be created with current parameters")
                
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
        
    def train(
        self,
        train_data: pd.DataFrame,
        train_target: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        validation_split: float = 0.2
    ) -> TrainingResult:
        """Train the model with monitoring and validation."""
        try:
            # Prepare training data
            X_train, y_train = self.prepare_data(train_data, train_target)
            
            # Prepare validation data
            if validation_data:
                X_val, y_val = self.prepare_data(
                    validation_data[0],
                    validation_data[1],
                    fit_scaler=False
                )
            else:
                # Use time series split
                split_idx = int(len(X_train) * (1 - validation_split))
                X_val = X_train[split_idx:]
                y_val = y_train[split_idx:]
                X_train = X_train[:split_idx]
                y_train = y_train[:split_idx]
            
            # Setup callbacks
            callbacks = self._setup_callbacks()
            
            # Train the model
            start_time = time.time()
            history = self.model.fit(
                X_train, y_train,
                epochs=self.training_config.get('epochs', 100),
                batch_size=self.training_config.get('batch_size', 32),
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            training_time = time.time() - start_time
            
            # Evaluate and calculate feature importance
            validation_metrics = self._evaluate_model(X_val, y_val)
            feature_importance = self._calculate_feature_importance(X_train)
            
            best_epoch = callbacks[0].stopped_epoch
            if best_epoch == 0:  # If early stopping didn't trigger
                best_epoch = len(history.history['loss'])
            
            return TrainingResult(
                model=self.model,
                history=history.history,
                validation_metrics=validation_metrics,
                feature_importance=feature_importance,
                training_time=training_time,
                best_epoch=best_epoch
            )
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
def _setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Setup training callbacks for monitoring and optimization."""
        try:
            model_path = self.training_config.get('model_checkpoint_path', 'models/checkpoints/best_model.h5')
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.training_config.get('early_stopping_patience', 10),
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath=model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.training_config.get('lr_reduction_factor', 0.5),
                    patience=self.training_config.get('lr_patience', 5),
                    min_lr=self.training_config.get('min_lr', 1e-6),
                    verbose=1
                )
            ]
            
            # Add optional tensorboard callback if configured
            if self.training_config.get('use_tensorboard', False):
                tensorboard_dir = self.training_config.get('tensorboard_dir', 'logs/tensorboard')
                callbacks.append(
                    tf.keras.callbacks.TensorBoard(
                        log_dir=tensorboard_dir,
                        histogram_freq=1
                    )
                )
            
            return callbacks
            
        except Exception as e:
            logger.error(f"Error setting up callbacks: {str(e)}")
            raise
        
    def _evaluate_model(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        try:
            # Make predictions
            predictions = self.model.predict(X_val, batch_size=self.training_config.get('batch_size', 32))
            predictions_binary = (predictions > 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_val, predictions_binary)),
                'precision': float(precision_score(y_val, predictions_binary)),
                'recall': float(recall_score(y_val, predictions_binary)),
                'f1': float(f1_score(y_val, predictions_binary)),
                'mae': float(mean_absolute_error(y_val, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(y_val, predictions)))
            }
            
            # Add custom metrics if configured
            if self.training_config.get('calculate_custom_metrics', False):
                custom_metrics = self._calculate_custom_metrics(y_val, predictions)
                metrics.update(custom_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
        
    def _calculate_feature_importance(
        self,
        X_train: np.ndarray,
        sample_size: int = 100
    ) -> Dict[str, float]:
        """Calculate feature importance using integrated gradients."""
        try:
            importance_scores = {}
            baseline = np.zeros_like(X_train[0])
            
            # Use a subset of data for efficiency
            if len(X_train) > sample_size:
                indices = np.random.choice(len(X_train), sample_size, replace=False)
                samples = X_train[indices]
            else:
                samples = X_train
            
            # Calculate importance for each feature
            for i, feature in enumerate(self.feature_columns):
                gradients = []
                
                for sample in samples:
                    with tf.GradientTape() as tape:
                        inputs = tf.convert_to_tensor([sample], dtype=tf.float32)
                        tape.watch(inputs)
                        prediction = self.model(inputs)
                        grad = tape.gradient(prediction, inputs)
                        gradients.append(grad)
                
                # Calculate mean absolute gradient for the feature
                feature_importance = np.mean([
                    abs(g[0, :, i].numpy()).mean() 
                    for g in gradients
                ])
                importance_scores[feature] = feature_importance
            
            # Normalize scores
            total = sum(importance_scores.values())
            normalized_scores = {
                k: float(v/total) 
                for k, v in importance_scores.items()
            }
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise
            
    def _calculate_custom_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate additional custom metrics."""
        try:
            custom_metrics = {}
            
            # Trading-specific metrics
            if self.training_config.get('calculate_trading_metrics', False):
                # Direction accuracy
                direction_correct = np.sum(np.sign(y_pred[1:] - y_pred[:-1]) == 
                                        np.sign(y_true[1:] - y_true[:-1]))
                direction_accuracy = direction_correct / (len(y_true) - 1)
                custom_metrics['direction_accuracy'] = float(direction_accuracy)
                
                # Maximum drawdown
                cumulative_returns = np.cumprod(1 + y_pred)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                custom_metrics['max_drawdown'] = float(drawdowns.min())
            
            return custom_metrics
            
        except Exception as e:
            logger.error(f"Error calculating custom metrics: {str(e)}")
            return {}
    
    def save_model(self, path: str):
        """Save the trained model."""
        try:
            if self.model is None:
                raise ValueError("No model to save")
                
            self.model.save(path)
            # Save scaler and configs
            model_artifacts = {
                'feature_columns': self.feature_columns,
                'model_config': self.model_config,
                'training_config': self.training_config,
                'scaler': self.scaler
            }
            np.save(f"{path}_artifacts.npy", model_artifacts)
            logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str):
        """Load a trained model."""
        try:
            # Load model
            self.model = tf.keras.models.load_model(path)
            
            # Load artifacts
            artifacts = np.load(f"{path}_artifacts.npy", allow_pickle=True).item()
            self.feature_columns = artifacts['feature_columns']
            self.model_config = artifacts['model_config']
            self.training_config = artifacts['training_config']
            self.scaler = artifacts['scaler']
            
            logger.info(f"Model loaded successfully from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise