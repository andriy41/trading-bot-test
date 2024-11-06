from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Union
import pandas as pd

@dataclass
class MarketData:
    symbol: str
    timeframe: str
    limit: int
    data: Union[pd.DataFrame, Dict[str, Any]]

@dataclass
class TradeSignal:
    symbol: str
    action: str
    price: float
    confidence: float
    indicators: Dict[str, Any]
    quantity: float
    timestamp: datetime = None
    stop_loss: float = None
    take_profit: float = None