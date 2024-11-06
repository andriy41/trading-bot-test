import numpy as np
import pandas as pd
from typing import Dict
from utils.logger import setup_logger

logger = setup_logger()

class RiskManager:
    def calculate_risk_metrics(self, data: pd.DataFrame, index: int, action: str, confidence: float) -> Dict[str, float]:
        current_price = data['close'].iloc[index]
        volatility = data['close'].pct_change().std() * np.sqrt(252)
        
        atr = self._calculate_atr(data).iloc[index]
        stop_loss = self.calculate_stop_loss(data, index, action)
        take_profit = self.calculate_take_profit(data, index, action)
        
        risk_per_trade = 0.02
        account_size = 100000
        position_size = self.calculate_position_size(
            account_size,
            risk_per_trade,
            current_price,
            stop_loss
        )
        
        risk_reward = abs(take_profit - current_price) / abs(stop_loss - current_price)
        expected_return = confidence * risk_reward
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_reward_ratio': risk_reward,
            'expected_return': expected_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility
        }

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_stop_loss(self, data: pd.DataFrame, index: int, action: str) -> float:
        atr = self._calculate_atr(data).iloc[index]
        current_price = data['close'].iloc[index]
        
        if action == 'buy':
            return current_price - (2 * atr)
        return current_price + (2 * atr)

    def calculate_take_profit(self, data: pd.DataFrame, index: int, action: str) -> float:
        atr = self._calculate_atr(data).iloc[index]
        current_price = data['close'].iloc[index]
        
        if action == 'buy':
            return current_price + (3 * atr)
        return current_price - (3 * atr)

    def calculate_position_size(self, account_size: float, risk_percentage: float,
                              entry_price: float, stop_loss: float) -> float:
        risk_amount = account_size * risk_percentage
        price_risk = abs(entry_price - stop_loss)
        return risk_amount / price_risk if price_risk != 0 else 0.0