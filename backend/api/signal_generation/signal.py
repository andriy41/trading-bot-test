from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    action: str
    confidence: float
    price: float
    indicators: Dict
    timeframe: str
    support_levels: List[float]
    resistance_levels: List[float]
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float = 0.0
    expected_return: float = 0.0
    market_conditions: Dict[str, float] = field(default_factory=dict)
    signal_strength: str = 'medium'

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'action': self.action,
            'confidence': round(self.confidence, 4),
            'price': round(self.price, 4),
            'indicators': {k: round(v, 4) if isinstance(v, float) else v 
                         for k, v in self.indicators.items()},
            'timeframe': self.timeframe,
            'support_levels': [round(x, 4) for x in self.support_levels],
            'resistance_levels': [round(x, 4) for x in self.resistance_levels],
            'stop_loss': round(self.stop_loss, 4),
            'take_profit': round(self.take_profit, 4),
            'risk_reward_ratio': round(self.risk_reward_ratio, 4),
            'expected_return': round(self.expected_return, 4),
            'market_conditions': {k: round(v, 4) for k, v in self.market_conditions.items()},
            'signal_strength': self.signal_strength
        }