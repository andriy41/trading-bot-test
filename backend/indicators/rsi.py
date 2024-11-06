# indicators/rsi.py
# backend/indicators/rsi.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RSIResult:
    values: np.ndarray
    overbought_regions: List[Tuple[int, int]]
    oversold_regions: List[Tuple[int, int]]
    divergences: Dict[str, List[int]]
    signals: Dict[str, List[Dict]]
    strength: np.ndarray

class RSI:
    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
    def calculate(self, data: pd.Series) -> RSIResult:
        """
        Enhanced RSI calculation with advanced signal generation.
        """
        # Calculate price changes
        delta = data.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = self._wilders_smoothing(gains)
        avg_losses = self._wilders_smoothing(losses)
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Generate advanced analysis
        overbought_regions = self._find_regions(rsi, self.overbought, '>')
        oversold_regions = self._find_regions(rsi, self.oversold, '<')
        divergences = self._detect_divergences(rsi, data)
        signals = self._generate_signals(rsi, data, divergences)
        strength = self._calculate_strength(rsi, data)
        
        return RSIResult(
            values=rsi.values,
            overbought_regions=overbought_regions,
            oversold_regions=oversold_regions,
            divergences=divergences,
            signals=signals,
            strength=strength
        )
    
    def _wilders_smoothing(self, data: pd.Series) -> pd.Series:
        """
        Implementation of Wilder's Smoothing Method.
        """
        alpha = 1/self.period
        return data.ewm(alpha=alpha, adjust=False).mean()
    
    def _find_regions(
        self,
        rsi: pd.Series,
        threshold: float,
        comparison: str
    ) -> List[Tuple[int, int]]:
        """
        Find continuous regions where RSI is above/below threshold.
        """
        regions = []
        start_idx = None
        
        for i in range(len(rsi)):
            condition = rsi[i] > threshold if comparison == '>' else rsi[i] < threshold
            
            if condition and start_idx is None:
                start_idx = i
            elif not condition and start_idx is not None:
                regions.append((start_idx, i))
                start_idx = None
                
        if start_idx is not None:
            regions.append((start_idx, len(rsi)-1))
            
        return regions
    
    def _detect_divergences(
        self,
        rsi: pd.Series,
        price: pd.Series
    ) -> Dict[str, List[int]]:
        """
        Detect bullish and bearish divergences.
        """
        divergences = {
            'bullish': [],
            'bearish': []
        }
        
        # Find local extrema
        rsi_peaks = self._find_peaks(rsi.values)
        rsi_troughs = self._find_peaks(-rsi.values)
        price_peaks = self._find_peaks(price.values)
        price_troughs = self._find_peaks(-price.values)
        
        # Detect bullish divergences
        for i in range(1, len(rsi_troughs)):
            if (rsi.iloc[rsi_troughs[i]] > rsi.iloc[rsi_troughs[i-1]] and
                price.iloc[price_troughs[i]] < price.iloc[price_troughs[i-1]]):
                divergences['bullish'].append(rsi_troughs[i])
        
        # Detect bearish divergences
        for i in range(1, len(rsi_peaks)):
            if (rsi.iloc[rsi_peaks[i]] < rsi.iloc[rsi_peaks[i-1]] and
                price.iloc[price_peaks[i]] > price.iloc[price_peaks[i-1]]):
                divergences['bearish'].append(rsi_peaks[i])
                
        return divergences
    
    def _generate_signals(
        self,
        rsi: pd.Series,
        price: pd.Series,
        divergences: Dict[str, List[int]]
    ) -> Dict[str, List[Dict]]:
        """
        Generate trading signals with confidence levels.
        """
        signals = {
            'buy': [],
            'sell': []
        }
        
        for i in range(1, len(rsi)):
            # Oversold condition with bullish divergence
            if (rsi.iloc[i-1] < self.oversold and 
                rsi.iloc[i] > self.oversold and 
                i in divergences['bullish']):
                signals['buy'].append({
                    'index': i,
                    'price': price.iloc[i],
                    'rsi': rsi.iloc[i],
                    'confidence': self._calculate_signal_confidence(rsi.iloc[i], price.iloc[i], 'buy'),
                    'type': 'divergence'
                })
            
            # Overbought condition with bearish divergence
            elif (rsi.iloc[i-1] > self.overbought and 
                  rsi.iloc[i] < self.overbought and 
                  i in divergences['bearish']):
                signals['sell'].append({
                    'index': i,
                    'price': price.iloc[i],
                    'rsi': rsi.iloc[i],
                    'confidence': self._calculate_signal_confidence(rsi.iloc[i], price.iloc[i], 'sell'),
                    'type': 'divergence'
                })
                
        return signals
    
    def _calculate_strength(self, rsi: pd.Series, price: pd.Series) -> np.ndarray:
        """
        Calculate signal strength based on RSI and price momentum.
        """
        rsi_momentum = rsi.diff()
        price_momentum = price.pct_change()
        
        # Normalize components
        norm_rsi = (rsi - 50) / 50  # Center around zero
        norm_rsi_momentum = self._normalize(rsi_momentum)
        norm_price_momentum = self._normalize(price_momentum)
        
        return (np.abs(norm_rsi) * (1 + np.abs(norm_rsi_momentum)) * 
                (1 + np.abs(norm_price_momentum)))
    
    @staticmethod
    def _normalize(data: pd.Series) -> np.ndarray:
        """
        Normalize data to [-1, 1] range.
        """
        return 2 * (data - data.min()) / (data.max() - data.min()) - 1
    
    @staticmethod
    def _find_peaks(data: np.ndarray) -> np.ndarray:
        """
        Find local maxima in data series.
        """
        return np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1

def calculate_rsi(data: pd.Series, period: int = 14) -> RSIResult:
    """
    Convenience function for RSI calculation.
    """
    calculator = RSI(period)
    return calculator.calculate(data)
