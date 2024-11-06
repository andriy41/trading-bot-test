# indicators/sma.py
# backend/indicators/sma.py

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple
from dataclasses import dataclass

@dataclass
class SMAResult:
    values: Dict[int, np.ndarray]
    crossovers: Dict[str, List[int]]
    support_resistance: Dict[str, List[float]]
    signals: Dict[str, List[Dict]]
    trend_strength: np.ndarray

class SMA:
    def __init__(self, periods: Union[List[int], int] = [20, 50, 200]):
        self.periods = [periods] if isinstance(periods, int) else periods
        
    def calculate(self, data: pd.Series) -> SMAResult:
        """
        Enhanced SMA calculation with advanced analysis features.
        """
        # Calculate SMAs for all periods using vectorized operations
        sma_values = {}
        for period in self.periods:
            sma_values[period] = self._calculate_single_sma(data, period)
            
        # Generate advanced analysis
        crossovers = self._detect_crossovers(sma_values)
        support_resistance = self._identify_support_resistance(data, sma_values)
        signals = self._generate_signals(data, sma_values, crossovers)
        trend_strength = self._calculate_trend_strength(data, sma_values)
        
        return SMAResult(
            values=sma_values,
            crossovers=crossovers,
            support_resistance=support_resistance,
            signals=signals,
            trend_strength=trend_strength
        )
    
    def _calculate_single_sma(self, data: pd.Series, period: int) -> np.ndarray:
        """
        Optimized SMA calculation using convolution.
        """
        weights = np.ones(period) / period
        return np.convolve(data, weights, mode='valid')
    
    def _detect_crossovers(self, sma_values: Dict[int, np.ndarray]) -> Dict[str, List[int]]:
        """
        Detect SMA crossovers including golden/death crosses.
        """
        crossovers = {}
        periods = sorted(self.periods)
        
        for i in range(len(periods)-1):
            for j in range(i+1, len(periods)):
                short_period = periods[i]
                long_period = periods[j]
                key = f"{short_period}_{long_period}"
                
                # Align arrays for comparison
                min_len = min(len(sma_values[short_period]), len(sma_values[long_period]))
                short_sma = sma_values[short_period][-min_len:]
                long_sma = sma_values[long_period][-min_len:]
                
                # Detect crossovers using vectorized operations
                bullish = (short_sma[:-1] <= long_sma[:-1]) & (short_sma[1:] > long_sma[1:])
                bearish = (short_sma[:-1] >= long_sma[:-1]) & (short_sma[1:] < long_sma[1:])
                
                crossovers[key] = {
                    'bullish': np.where(bullish)[0] + 1,
                    'bearish': np.where(bearish)[0] + 1
                }
                
        return crossovers
    
    def _identify_support_resistance(
        self,
        data: pd.Series,
        sma_values: Dict[int, np.ndarray]
    ) -> Dict[str, List[float]]:
        """
        Identify potential support and resistance levels using SMAs.
        """
        levels = {
            'support': [],
            'resistance': []
        }
        
        for period, sma in sma_values.items():
            # Calculate price-SMA interaction points
            price_above = data > sma
            crossings = np.diff(price_above.astype(int))
            
            support_points = data[np.where(crossings == -1)[0]]
            resistance_points = data[np.where(crossings == 1)[0]]
            
            levels['support'].extend(support_points)
            levels['resistance'].extend(resistance_points)
            
        # Cluster nearby levels
        levels['support'] = self._cluster_levels(levels['support'])
        levels['resistance'] = self._cluster_levels(levels['resistance'])
        
        return levels
    
    def _generate_signals(
        self,
        data: pd.Series,
        sma_values: Dict[int, np.ndarray],
        crossovers: Dict[str, List[int]]
    ) -> Dict[str, List[Dict]]:
        """
        Generate trading signals with confidence levels.
        """
        signals = {
            'buy': [],
            'sell': []
        }
        
        # Process crossover signals
        for key, crosses in crossovers.items():
            short_period, long_period = map(int, key.split('_'))
            
            for idx in crosses['bullish']:
                if self._validate_signal(data, sma_values, idx, 'buy'):
                    confidence = self._calculate_signal_confidence(
                        data, sma_values, idx, 'buy'
                    )
                    signals['buy'].append({
                        'index': idx,
                        'type': 'golden_cross' if long_period == 200 else 'bullish_cross',
                        'confidence': confidence
                    })
                    
            for idx in crosses['bearish']:
                if self._validate_signal(data, sma_values, idx, 'sell'):
                    confidence = self._calculate_signal_confidence(
                        data, sma_values, idx, 'sell'
                    )
                    signals['sell'].append({
                        'index': idx,
                        'type': 'death_cross' if long_period == 200 else 'bearish_cross',
                        'confidence': confidence
                    })
                    
        return signals
    
    def _calculate_trend_strength(
        self,
        data: pd.Series,
        sma_values: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Calculate trend strength using multiple SMAs.
        """
        strength = np.zeros(len(data))
        
        for period, sma in sma_values.items():
            # Calculate normalized distance from price to SMA
            diff = (data[-len(sma):] - sma) / data[-len(sma):]
            strength[-len(sma):] += np.sign(diff) * np.abs(diff)
            
        return strength / len(sma_values)
    
    @staticmethod
    def _cluster_levels(levels: List[float], tolerance: float = 0.001) -> List[float]:
        """
        Cluster nearby price levels.
        """
        if not levels:
            return []
            
        levels = sorted(levels)
        clusters = [[levels[0]]]
        
        for level in levels[1:]:
            if abs(level - clusters[-1][-1]) <= tolerance * level:
                clusters[-1].append(level)
            else:
                clusters.append([level])
                
        return [sum(cluster) / len(cluster) for cluster in clusters]

def calculate_sma(data: pd.Series, periods: Union[List[int], int] = [20, 50, 200]) -> SMAResult:
    """
    Convenience function for SMA calculation.
    """
    calculator = SMA(periods)
    return calculator.calculate(data)
