# tests/test_indicators.py 
import unittest
import numpy as np
import pandas as pd
from indicators.ema import ExponentialMovingAverage, calculate_ema
from indicators.macd import MACD, calculate_macd
from indicators.rsi import RSI, calculate_rsi
from indicators.sma import SMA, calculate_sma
from indicators.others import TechnicalIndicators

class TestIndicators(unittest.TestCase):
    def setUp(self):
        # Create sample price data
        self.prices = pd.Series(np.random.random(100) * 100)
        self.high = pd.Series(np.random.random(100) * 110)
        self.low = pd.Series(np.random.random(100) * 90)
        self.volume = pd.Series(np.random.random(100) * 1000000)
        
    def test_ema_calculation(self):
        ema = ExponentialMovingAverage(periods=[9, 21])
        result = ema.calculate(self.prices)
        
        self.assertIsInstance(result.values, dict)
        self.assertEqual(len(result.values), 2)
        self.assertTrue(all(isinstance(v, np.ndarray) for v in result.values.values()))
        
    def test_macd_calculation(self):
        macd = MACD()
        result = macd.calculate(self.prices)
        
        self.assertIsInstance(result.macd_line, np.ndarray)
        self.assertIsInstance(result.signal_line, np.ndarray)
        self.assertIsInstance(result.histogram, np.ndarray)
        self.assertEqual(len(result.macd_line), len(result.signal_line))
        
    def test_rsi_calculation(self):
        rsi = RSI(period=14)
        result = rsi.calculate(self.prices)
        
        self.assertIsInstance(result.values, np.ndarray)
        self.assertTrue(all(0 <= x <= 100 for x in result.values))
        self.assertEqual(len(result.values), len(self.prices))
        
    def test_sma_calculation(self):
        sma = SMA(periods=[10, 20, 50])
        result = sma.calculate(self.prices)
        
        self.assertIsInstance(result.values, dict)
        self.assertEqual(len(result.values), 3)
        self.assertTrue(all(len(v) > 0 for v in result.values.values()))
        
    def test_bollinger_bands(self):
        ti = TechnicalIndicators()
        result = ti.calculate_bollinger_bands(self.prices)
        
        self.assertIn('upper', result)
        self.assertIn('middle', result)
        self.assertIn('lower', result)
        self.assertTrue(all(result['upper'] >= result['middle']))
        self.assertTrue(all(result['middle'] >= result['lower']))
        
    def test_adx_calculation(self):
        ti = TechnicalIndicators()
        result = ti.calculate_adx(self.high, self.low, self.prices)
        
        self.assertIn('adx', result)
        self.assertIn('plus_di', result)
        self.assertIn('minus_di', result)
        self.assertTrue(all(0 <= x <= 100 for x in result['adx']))
        
    def test_stochastic_oscillator(self):
        ti = TechnicalIndicators()
        result = ti.calculate_stochastic(self.high, self.low, self.prices)
        
        self.assertIn('k_fast', result)
        self.assertIn('k_slow', result)
        self.assertIn('d_slow', result)
        self.assertTrue(all(0 <= x <= 100 for x in result['k_fast']))
        
    def test_indicator_crossovers(self):
        ema = ExponentialMovingAverage(periods=[9, 21])
        result = ema.calculate(self.prices)
        
        self.assertIn('9_21', result.crossovers)
        self.assertIsInstance(result.crossovers['9_21'], dict)
        self.assertIn('bullish', result.crossovers['9_21'])
        self.assertIn('bearish', result.crossovers['9_21'])
        
    def test_trend_strength(self):
        ema = ExponentialMovingAverage(periods=[9, 21])
        result = ema.calculate(self.prices)
        
        self.assertIsInstance(result.trend_strength, np.ndarray)
        self.assertEqual(len(result.trend_strength), len(self.prices))
        
    def test_signal_generation(self):
        macd = MACD()
        result = macd.calculate(self.prices)
        
        self.assertIn('buy', result.signals)
        self.assertIn('sell', result.signals)
        self.assertIsInstance(result.signals['buy'], list)
        self.assertIsInstance(result.signals['sell'], list)

if __name__ == '__main__':
    unittest.main()
