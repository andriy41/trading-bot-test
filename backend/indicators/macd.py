# indicators/macd.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, NamedTuple
from dataclasses import dataclass
from .ema import calculate_ema

@dataclass
class MACDResult:
    macd_line: np.ndarray
    signal_line: np.ndarray
    histogram: np.ndarray
    signals: Dict[str, list]
    divergences: Dict[str, list]
    strength: np.ndarray

class MACD:
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def calculate(self, data: pd.Series) -> MACDResult:
        """
        Calculate MACD with advanced signal generation and pattern recognition.
        """
        # Calculate MACD components using vectorized operations
        fast_ema = calculate_ema(data, self.fast_period).values[self.fast_period]
        slow_ema = calculate_ema(data, self.slow_period).values[self.slow_period]
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = calculate_ema(macd_line, self.signal_period).values[self.signal_period]
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Generate signals and analyze patterns
        signals = self._generate_signals(macd_line, signal_line, histogram, data)
        divergences = self._detect_divergences(macd_line, data)
        strength = self._calculate_strength(macd_line, histogram)
        
        return MACDResult(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
            signals=signals,
            divergences=divergences,
            strength=strength
        )
        
    def _generate_signals(
        self,
        macd_line: np.ndarray,
        signal_line: np.ndarray,
        histogram: np.ndarray,
        price_data: np.ndarray
    ) -> Dict[str, list]:
        """
        Generate trading signals with multiple confirmation factors.
        """
        signals = {
            'buy': [],
            'sell': [],
            'strength': []
        }
        
        # Vectorized signal detection
        crossover_buy = (macd_line[:-1] <= signal_line[:-1]) & (macd_line[1:] > signal_line[1:])
        crossover_sell = (macd_line[:-1] >= signal_line[:-1]) & (macd_line[1:] < signal_line[1:])
        
        # Zero-line crossovers
        zero_cross_up = (macd_line[:-1] <= 0) & (macd_line[1:] > 0)
        zero_cross_down = (macd_line[:-1] >= 0) & (macd_line[1:] < 0)
        
        # Combine signals with confirmation
        for i in range(1, len(macd_line)):
            if crossover_buy[i-1]:
                if histogram[i] > 0 and macd_line[i] > macd_line[i-1]:
                    strength = self._calculate_signal_strength(
                        macd_line[i],
                        histogram[i],
                        price_data[i]
                    )
                    if strength > 0.7:  # Strong signal threshold
                        signals['buy'].append(i)
                        signals['strength'].append(strength)
                        
            if crossover_sell[i-1]:
                if histogram[i] < 0 and macd_line[i] < macd_line[i-1]:
                    strength = self._calculate_signal_strength(
                        macd_line[i],
                        histogram[i],
                        price_data[i]
                    )
                    if strength > 0.7:  # Strong signal threshold
                        signals['sell'].append(i)
                        signals['strength'].append(strength)
                        
        return signals
        
    def _detect_divergences(
        self,
        macd_line: np.ndarray,
        price_data: np.ndarray
    ) -> Dict[str, list]:
        """
        Detect bullish and bearish divergences.
        """
        divergences = {
            'bullish': [],
            'bearish': []
        }
        
        # Find local extrema
        macd_peaks = self._find_peaks(macd_line)
        macd_troughs = self._find_peaks(-macd_line)
        price_peaks = self._find_peaks(price_data)
        price_troughs = self._find_peaks(-price_data)
        
        # Detect bullish divergence (price lower low, MACD higher low)
        for i in range(1, len(macd_troughs)):
            if (macd_line[macd_troughs[i]] > macd_line[macd_troughs[i-1]] and
                price_data[price_troughs[i]] < price_data[price_troughs[i-1]]):
                divergences['bullish'].append(macd_troughs[i])
                
        # Detect bearish divergence (price higher high, MACD lower high)
        for i in range(1, len(macd_peaks)):
            if (macd_line[macd_peaks[i]] < macd_line[macd_peaks[i-1]] and
                price_data[price_peaks[i]] > price_data[price_peaks[i-1]]):
                divergences['bearish'].append(macd_peaks[i])
                
        return divergences
        
    def _calculate_strength(
        self,
        macd_line: np.ndarray,
        histogram: np.ndarray
    ) -> np.ndarray:
        """
        Calculate signal strength based on MACD components.
        """
        # Normalize components
        norm_macd = self._normalize(macd_line)
        norm_hist = self._normalize(histogram)
        
        # Combine factors for strength calculation
        strength = np.abs(norm_macd) * np.abs(norm_hist)
        
        # Apply momentum factor
        momentum = np.gradient(macd_line)
        norm_momentum = self._normalize(momentum)
        
        return strength * (1 + norm_momentum)
        
    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [-1, 1] range.
        """
        return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
        
    @staticmethod
    def _find_peaks(data: np.ndarray) -> np.ndarray:
        """
        Find local maxima in data series.
        """
        return np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1

def calculate_macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> MACDResult:
    """
    Convenience function for MACD calculation.
    """
    calculator = MACD(fast_period, slow_period, signal_period)
    return calculator.calculate(data)
