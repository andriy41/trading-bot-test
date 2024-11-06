import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from utils.logger import setup_logger

logger = setup_logger()

class SignalAnalyzer:
    def analyze_trend_strength(self, indicators: Dict, index: int) -> float:
        adx_value = indicators['adx']['adx'][index]
        adx_score = min(adx_value / 100.0, 1.0) if adx_value > 25 else 0.0
        
        ma_score = 0.0
        if self._is_uptrend(indicators, index):
            ma_score = 1.0
            if self._is_accelerating_trend(indicators, index, 'up'):
                ma_score *= 1.2
        elif self._is_downtrend(indicators, index):
            ma_score = 1.0
            if self._is_accelerating_trend(indicators, index, 'down'):
                ma_score *= 1.2
        
        consistency_score = self._check_trend_consistency(indicators, index)
        return (adx_score + ma_score + consistency_score) / 3

    def _is_uptrend(self, indicators: Dict, index: int) -> bool:
        ma20 = indicators['ma20'][index]
        ma50 = indicators['ma50'][index]
        price = indicators['close'][index]
        return price > ma20 > ma50

    def _is_downtrend(self, indicators: Dict, index: int) -> bool:
        ma20 = indicators['ma20'][index]
        ma50 = indicators['ma50'][index]
        price = indicators['close'][index]
        return price < ma20 < ma50

    def _is_accelerating_trend(self, indicators: Dict, index: int, direction: str) -> bool:
        if index < 5:
            return False
        
        ma20_slope = indicators['ma20'][index] - indicators['ma20'][index-5]
        ma50_slope = indicators['ma50'][index] - indicators['ma50'][index-5]
        
        if direction == 'up':
            return ma20_slope > 0 and ma20_slope > ma50_slope
        return ma20_slope < 0 and ma20_slope < ma50_slope

    def _check_trend_consistency(self, indicators: Dict, index: int) -> float:
        if index < 10:
            return 0.0
            
        price_changes = np.diff(indicators['close'][index-10:index+1])
        positive_changes = np.sum(price_changes > 0)
        negative_changes = np.sum(price_changes < 0)
        
        return abs(positive_changes - negative_changes) / 10.0

    def analyze_momentum(self, indicators: Dict, index: int) -> float:
        rsi = indicators['rsi'][index]
        macd_hist = indicators['macd']['histogram'][index]
        stoch_k = indicators['stochastic']['k'][index]
        
        momentum_score = 0.0
        if 30 <= rsi <= 70:
            momentum_score += 0.3
        elif rsi < 30:
            momentum_score += 0.4
        elif rsi > 70:
            momentum_score += 0.2
        
        if macd_hist > 0:
            momentum_score += 0.3
        
        if 20 <= stoch_k <= 80:
            momentum_score += 0.4
        
        return momentum_score

    def analyze_volume(self, indicators: Dict, index: int) -> float:
        volume = indicators['volume']
        avg_volume = volume.iloc[max(0, index-20):index].mean()
        current_volume = volume.iloc[index]
        
        volume_score = 0.0
        if current_volume > avg_volume * 1.5:
            volume_score += 0.4
        elif current_volume > avg_volume:
            volume_score += 0.3
        else:
            volume_score += 0.1
            
        volume_std = volume.iloc[max(0, index-20):index].std()
        if volume_std < avg_volume * 0.5:
            volume_score += 0.2
            
        return min(volume_score, 1.0)

    def calculate_support_resistance(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
        r1 = 2 * pivot - lows[-1]
        r2 = pivot + (highs[-1] - lows[-1])
        s1 = 2 * pivot - highs[-1]
        s2 = pivot - (highs[-1] - lows[-1])
        
        window = 20
        local_highs = []
        local_lows = []
        
        for i in range(window, len(data)):
            if all(highs[i] > highs[i-window:i]):
                local_highs.append(highs[i])
            if all(lows[i] < lows[i-window:i]):
                local_lows.append(lows[i])
        
        support_levels = sorted(set([s1, s2] + local_lows[-5:]))
        resistance_levels = sorted(set([r1, r2] + local_highs[-5:]))
        
        return support_levels, resistance_levels

    def analyze_market_conditions(self, data: pd.DataFrame, indicators: Dict) -> Dict[str, float]:
        latest_idx = len(data) - 1
        return {
            'trend_strength': self.analyze_trend_strength(indicators, latest_idx),
            'momentum': self.analyze_momentum(indicators, latest_idx),
            'volume_analysis': self.analyze_volume(indicators, latest_idx)
        }