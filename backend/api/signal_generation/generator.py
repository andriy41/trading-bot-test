from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from .analyzer import SignalAnalyzer
from .risk import RiskManager
from .signal import Signal
from models.prediction import PricePredictor
from utils.logger import setup_logger

logger = setup_logger()

class SignalGenerator:
    def __init__(
        self,
        confidence_threshold: float = 0.85,
        risk_reward_min: float = 2.0,
        use_ml_predictions: bool = True,
        max_workers: int = 4
    ):
        self.confidence_threshold = confidence_threshold
        self.risk_reward_min = risk_reward_min
        self.use_ml_predictions = use_ml_predictions
        self.max_workers = max_workers
        
        self.analyzer = SignalAnalyzer()
        self.risk_manager = RiskManager()
        self.price_predictor = PricePredictor() if use_ml_predictions else None
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        return {
            'ma20': data['close'].rolling(window=20).mean(),
            'ma50': data['close'].rolling(window=50).mean(),
            'rsi': self._calculate_rsi(data['close']),
            'macd': self._calculate_macd(data['close']),
            'volume': data['volume'],
            'adx': self._calculate_adx(data),
            'stochastic': self._calculate_stochastic(data),
            'close': data['close']
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> Dict:
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return {'macd': macd, 'signal': signal, 'histogram': histogram}

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> Dict:
        high = data['high']
        low = data['low']
        close = data['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = abs(100 * (minus_dm.rolling(period).mean() / atr))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}

    def _calculate_stochastic(self, data: pd.DataFrame, period: int = 14) -> Dict:
        high_roll = data['high'].rolling(window=period).max()
        low_roll = data['low'].rolling(window=period).min()
        k = 100 * (data['close'] - low_roll) / (high_roll - low_roll)
        d = k.rolling(window=3).mean()
        return {'k': k, 'd': d}

    def _get_predictions(self, data: pd.DataFrame) -> List[Dict]:
        return self.price_predictor.predict(data) if self.price_predictor else []

    def _analyze_signal_strength(self, indicators: Dict, index: int) -> Tuple[float, str]:
        trend_strength = self.analyzer.analyze_trend_strength(indicators, index)
        momentum = self.analyzer.analyze_momentum(indicators, index)
        volume = self.analyzer.analyze_volume(indicators, index)
        
        composite_score = (trend_strength * 0.4 + momentum * 0.4 + volume * 0.2)
        
        if indicators['macd']['histogram'][index] > 0:
            action = 'buy'
        else:
            action = 'sell'
            
        return composite_score, action

    def _get_indicator_values(self, indicators: Dict, index: int) -> Dict:
        return {
            'rsi': indicators['rsi'][index],
            'macd': {
                'value': indicators['macd']['macd'][index],
                'signal': indicators['macd']['signal'][index],
                'histogram': indicators['macd']['histogram'][index]
            },
            'adx': indicators['adx']['adx'][index],
            'ma20': indicators['ma20'][index],
            'ma50': indicators['ma50'][index]
        }

    def _determine_signal_strength(self, confidence: float) -> str:
        if confidence >= 0.8:
            return 'strong'
        elif confidence >= 0.6:
            return 'medium'
        return 'weak'

    def _passes_filters(self, signal: Signal, filters: Optional[Dict]) -> bool:
        if not filters:
            return True
            
        for key, value in filters.items():
            if key == 'min_confidence' and signal.confidence < value:
                return False
            elif key == 'min_risk_reward' and signal.risk_reward_ratio < value:
                return False
            elif key == 'signal_strength' and signal.signal_strength not in value:
                return False
        return True

    def generate_signals(
        self,
        data: pd.DataFrame,
        timeframe: str,
        additional_filters: Optional[Dict] = None
    ) -> List[Signal]:
        if data.empty:
            raise ValueError("Empty dataset provided")

        required_columns = ['high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")

        indicators = self._calculate_indicators(data)
        support_levels, resistance_levels = self.analyzer.calculate_support_resistance(data)
        
        predictions = []
        if self.use_ml_predictions and self.price_predictor:
            try:
                predictions = self._get_predictions(data)
            except Exception as e:
                logger.warning(f"ML predictions failed: {str(e)}")
                predictions = [{'confidence': 0.5} for _ in range(len(data))]

        signals = []
        market_conditions = self.analyzer.analyze_market_conditions(data, indicators)

        for i in range(len(data)):
            if i < 200:  # Need enough data for indicators
                continue

            confidence, action = self._analyze_signal_strength(indicators, i)
            
            if predictions:
                ml_confidence = predictions[i]['confidence']
                confidence = 0.7 * confidence + 0.3 * ml_confidence

            if confidence >= self.confidence_threshold:
                risk_metrics = self.risk_manager.calculate_risk_metrics(
                    data, i, action, confidence
                )
                
                if risk_metrics['risk_reward_ratio'] < self.risk_reward_min:
                    continue

                signal = Signal(
                    timestamp=data.index[i],
                    symbol=str(data.get('symbol', 'Unknown')),
                    action=action,
                    confidence=confidence,
                    price=data['close'].iloc[i],
                    indicators=self._get_indicator_values(indicators, i),
                    timeframe=timeframe,
                    support_levels=support_levels,
                    resistance_levels=resistance_levels,
                    stop_loss=risk_metrics['stop_loss'],
                    take_profit=risk_metrics['take_profit'],
                    risk_reward_ratio=risk_metrics['risk_reward_ratio'],
                    expected_return=risk_metrics['expected_return'],
                    market_conditions=market_conditions,
                    signal_strength=self._determine_signal_strength(confidence)
                )
                
                if self._passes_filters(signal, additional_filters):
                    signals.append(signal)

        return signals

    def cleanup(self):
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.price_predictor:
            self.price_predictor.cleanup()
        logger.info("SignalGenerator cleanup completed")

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()