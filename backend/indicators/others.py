#others.py
# indicators/others.py

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple
from dataclasses import dataclass

@dataclass
class TechnicalIndicators:
    """Technical indicators calculation class."""
    
    def __init__(self):
        """Initialize TechnicalIndicators class."""
        self.indicators = {}

    def calculate_bollinger_bands(
        self,
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands with dynamic bandwidth."""
        if data.empty:
            raise ValueError("Input data cannot be empty")

        middle_band = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std()

        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        bandwidth = (upper_band - lower_band) / middle_band

        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'bandwidth': bandwidth
        }

    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index with trend strength."""
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("Input series must be of equal length")

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        tr_smoothed = self._smoothed_average(tr, period)
        plus_di = 100 * self._smoothed_average(pd.Series(plus_dm), period) / tr_smoothed
        minus_di = 100 * self._smoothed_average(pd.Series(minus_dm), period) / tr_smoothed

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = self._smoothed_average(dx, period)

        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }

    def calculate_ichimoku(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components."""
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("Input series must be of equal length")

        tenkan_period = 9
        kijun_period = 26
        senkou_span_b_period = 52

        tenkan_sen = self._midpoint(high, low, tenkan_period)
        kijun_sen = self._midpoint(high, low, kijun_period)

        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        senkou_span_b = self._midpoint(high, low, senkou_span_b_period).shift(kijun_period)

        chikou_span = close.shift(-kijun_period)

        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }

    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator with smoothing."""
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError("Input series must be of equal length")

        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_fast = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k_slow = k_fast.rolling(window=d_period).mean()
        d_slow = k_slow.rolling(window=d_period).mean()

        return {
            'k_fast': k_fast,
            'k_slow': k_slow,
            'd_slow': d_slow
        }

    def calculate_fibonacci_levels(
        self,
        high: float,
        low: float
    ) -> Dict[str, float]:
        """Calculate Fibonacci retracement and extension levels."""
        if high <= low:
            raise ValueError("High price must be greater than low price")

        diff = high - low

        return {
            'extension_1.618': high + (diff * 0.618),
            'extension_1.272': high + (diff * 0.272),
            'level_1.000': high,
            'level_0.786': high - (diff * 0.786),
            'level_0.618': high - (diff * 0.618),
            'level_0.500': high - (diff * 0.500),
            'level_0.382': high - (diff * 0.382),
            'level_0.236': high - (diff * 0.236),
            'level_0.000': low
        }

    @staticmethod
    def _smoothed_average(data: pd.Series, period: int) -> pd.Series:
        """Calculate smoothed moving average."""
        if period <= 0:
            raise ValueError("Period must be positive")
        alpha = 1.0 / period
        return data.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def _midpoint(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        """Calculate midpoint price over period."""
        if period <= 0:
            raise ValueError("Period must be positive")
        return (
            high.rolling(window=period).max() +
            low.rolling(window=period).min()
        ) / 2

# Initialize technical indicators
technical_indicators = TechnicalIndicators()