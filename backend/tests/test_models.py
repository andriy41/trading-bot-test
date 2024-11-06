# tests/test_models.py 
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from models.backtesting import Backtester
from models.ensemble import TradingEnsemble
from models.training import ModelTrainer
from models.prediction import PricePredictor

class TestModels(unittest.TestCase):
    def setUp(self):
        # Generate sample data
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
        self.sample_data = pd.DataFrame({
            'open': np.random.random(1000) * 100,
            'high': np.random.random(1000) * 110,
            'low': np.random.random(1000) * 90,
            'close': np.random.random(1000) * 100,
            'volume': np.random.random(1000) * 1000000
        }, index=dates)
        
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume']
        
    def test_backtester(self):
        backtester = Backtester(initial_capital=100000)
        signals = pd.DataFrame({
            'action': ['buy', 'sell'] * 50,
            'confidence': np.random.random(100)
        }, index=self.sample_data.index[:100])
        
        result = backtester.run_backtest(self.sample_data, signals)
        
        self.assertIsNotNone(result.total_return)
        self.assertIsNotNone(result.sharpe_ratio)
        self.assertIsNotNone(result.max_drawdown)
        self.assertGreater(len(result.trades), 0)
        self.assertEqual(len(result.equity_curve), len(self.sample_data))
        
    def test_ensemble_model(self):
        ensemble = TradingEnsemble(
            feature_columns=self.feature_columns,
            target_column='close'
        )
        
        X = self.sample_data[self.feature_columns]
        y = (self.sample_data['close'].shift(-1) > self.sample_data['close']).astype(int)
        y = y[:-1]  # Remove last NaN
        
        ensemble.train(X[:-1], y)
        result = ensemble.predict(X[-10:])
        
        self.assertEqual(len(result.predictions), 10)
        self.assertGreater(len(result.model_weights), 0)
        self.assertTrue(all(0 <= score <= 1 for score in result.confidence_scores))
        
    def test_model_trainer(self):
        model_config = {
            'sequence_length': 10,
            'lstm_units': 64
        }
        
        training_config = {
            'epochs': 2,
            'batch_size': 32,
            'learning_rate': 0.001,
            'loss_function': 'binary_crossentropy'
        }
        
        trainer = ModelTrainer(
            model_config=model_config,
            training_config=training_config,
            feature_columns=self.feature_columns
        )
        
        X = self.sample_data[self.feature_columns]
        y = (self.sample_data['close'].shift(-1) > self.sample_data['close']).astype(int)
        y = y[:-1]
        
        result = trainer.train(X[:-1], y)
        
        self.assertIsNotNone(result.model)
        self.assertIn('loss', result.history)
        self.assertGreater(len(result.validation_metrics), 0)
        self.assertGreater(result.training_time, 0)
        
    def test_price_predictor(self):
        predictor = PricePredictor(
            sequence_length=10,
            n_features=len(self.feature_columns)
        )
        
        X = self.sample_data[self.feature_columns]
        result = predictor.predict(X)
        
        self.assertEqual(len(result.predictions), len(X) - predictor.sequence_length)
        self.assertTrue(all(0 <= conf <= 1 for conf in result.confidence))
        self.assertGreater(len(result.support_levels), 0)
        self.assertGreater(len(result.resistance_levels), 0)
        
    def test_model_integration(self):
        # Test integration between different model components
        ensemble = TradingEnsemble(
            feature_columns=self.feature_columns,
            target_column='close'
        )
        
        backtester = Backtester(initial_capital=100000)
        
        # Train ensemble and generate predictions
        X = self.sample_data[self.feature_columns]
        y = (self.sample_data['close'].shift(-1) > self.sample_data['close']).astype(int)
        y = y[:-1]
        
        ensemble.train(X[:-1], y)
        predictions = ensemble.predict(X)
        
        # Convert predictions to signals
        signals = pd.DataFrame({
            'action': ['buy' if p > 0.5 else 'sell' for p in predictions.predictions],
            'confidence': predictions.confidence_scores
        }, index=self.sample_data.index[:len(predictions.predictions)])
        
        # Run backtest
        result = backtester.run_backtest(self.sample_data, signals)
        
        self.assertIsNotNone(result.total_return)
        self.assertGreater(len(result.trades), 0)
        
    def test_model_persistence(self):
        # Test model saving and loading
        ensemble = TradingEnsemble(
            feature_columns=self.feature_columns,
            target_column='close'
        )
        
        X = self.sample_data[self.feature_columns]
        y = (self.sample_data['close'].shift(-1) > self.sample_data['close']).astype(int)
        y = y[:-1]
        
        ensemble.train(X[:-1], y)
        
        # Save and load model weights
        ensemble.save_weights('test_model_weights.h5')
        ensemble.load_weights('test_model_weights.h5')
        
        # Verify predictions still work
        result = ensemble.predict(X[-10:])
        self.assertEqual(len(result.predictions), 10)

if __name__ == '__main__':
    unittest.main()
