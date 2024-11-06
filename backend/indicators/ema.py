# indicators/ema.py
# backend/indicators/ema.py

import numpy as np
import pandas as pd
from typing import Union, List, Dict
from dataclasses import dataclass

@dataclass
class EMAResult:
    values: np.ndarray
    crossovers: Dict[str, List[int]]
    trend_strength: np.ndarray
    signals: Dict[str, List[int]]

class ExponentialMovingAverage:
    def __init__(self, periods: Union[List[int], int] = [9, 21, 55]):
        self.periods = [periods] if isinstance(periods, int) else periods
        self.values = {}
        
    def calculate(self, data: Union[pd.Series, np.ndarray], alpha: float = None) -> EMAResult:
        """
        Calculate EMA with advanced features and optimizations.
        
        Args:
            data: Price data
            alpha: Optional smoothing factor override
        """
        if isinstance(data, pd.Series):
            data = data.values
            
        results = {}
        crossovers = {}
        
        # Vectorized EMA calculation for all periods
        for period in self.periods:
            alpha = alpha or 2.0 / (period + 1)
            weights = (1 - alpha) ** np.arange(len(data))
            weights /= weights.sum()
            
            # Optimize calculation using NumPy's convolve
            ema = np.convolve(data, weights[::-1])[:len(data)]
            results[period] = ema
            
        # Calculate crossovers and signals
        if len(self.periods) >= 2:
            crossovers = self._calculate_crossovers(results)
            
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(data, results)
        
        # Generate trading signals
        signals = self._generate_signals(results, crossovers, trend_strength)
        
        return EMAResult(
            values=results,
            crossovers=crossovers,
            trend_strength=trend_strength,
            signals=signals
        )
    
    def _calculate_crossovers(self, ema_values: Dict[int, np.ndarray]) -> Dict[str, List[int]]:
        """
        Detect EMA crossovers for trading signals.
        """
        crossovers = {}
        periods = sorted(self.periods)
        
        for i in range(len(periods)-1):
            for j in range(i+1, len(periods)):
                short_period = periods[i]
                long_period = periods[j]
                key = f"{short_period}_{long_period}"
                
                # Vectorized crossover detection
                short_ema = ema_values[short_period]
                long_ema = ema_values[long_period]
                
                # Bullish crossover
                bullish = (short_ema[:-1] <= long_ema[:-1]) & (short_ema[1:] > long_ema[1:])
                # Bearish crossover
                bearish = (short_ema[:-1] >= long_ema[:-1]) & (short_ema[1:] < long_ema[1:])
                
                crossovers[key] = {
                    'bullish': np.where(bullish)[0] + 1,
                    'bearish': np.where(bearish)[0] + 1
                }
                
        return crossovers
    
    def _calculate_trend_strength(self, data: np.ndarray, ema_values: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Calculate trend strength using multiple EMAs.
        """
        trend_strength = np.zeros(len(data))
        
        for period, ema in ema_values.items():
            # Calculate distance from price to EMA
            distance = data - ema
            # Normalize distance
            normalized_distance = distance / (data * 0.01)  # Convert to percentage
            trend_strength += np.sign(distance) * np.abs(normalized_distance)
            
        return trend_strength / len(ema_values)
    
    def _generate_signals(
        self,
        ema_values: Dict[int, np.ndarray],
        crossovers: Dict[str, List[int]],
        trend_strength: np.ndarray
    ) -> Dict[str, List[int]]:
        """
        Generate trading signals based on EMA analysis.
        """
        signals = {
            'buy': [],
            'sell': [],
            'strength': []
        }
        
        # Combine crossover signals with trend strength
        for key, crosses in crossovers.items():
            for bull_idx in crosses['bullish']:
                if trend_strength[bull_idx] > 0.5:  # Strong uptrend confirmation
                    signals['buy'].append(bull_idx)
                    signals['strength'].append(trend_strength[bull_idx])
                    
            for bear_idx in crosses['bearish']:
                if trend_strength[bear_idx] < -0.5:  # Strong downtrend confirmation
                    signals['sell'].append(bear_idx)
                    signals['strength'].append(abs(trend_strength[bear_idx]))
                    
        return signals

def calculate_ema(
    data: Union[pd.Series, np.ndarray],
    periods: Union[List[int], int] = [9, 21, 55],
    alpha: float = None
) -> EMAResult:
    """
    Convenience function for EMA calculation.
    """
    calculator = ExponentialMovingAverage(periods)
    return calculator.calculate(data, alpha)
