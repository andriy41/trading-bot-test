# models/backtesting.py
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import plotly.graph_objects as go

@dataclass
class TradeResult:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    profit_loss: float
    position_size: float
    trade_duration: pd.Timedelta
    max_drawdown: float
    risk_reward_ratio: float

@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: List[TradeResult]
    equity_curve: pd.Series
    monthly_returns: pd.Series
    performance_metrics: Dict
    trade_analytics: Dict

class Backtester:
    def __init__(
        self,
        initial_capital: float = 100000,
        position_size: float = 0.02,
        max_positions: int = 5,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame
    ) -> BacktestResult:
        """
        Execute backtest with comprehensive analysis.
        """
        equity_curve = []
        open_positions = []
        
        for timestamp, row in data.iterrows():
            # Process open positions
            self._process_open_positions(row, timestamp)
            
            # Check for new signals
            if timestamp in signals.index:
                signal = signals.loc[timestamp]
                self._process_signal(signal, row, timestamp)
                
            # Update equity curve
            current_equity = self._calculate_current_equity(row)
            equity_curve.append(current_equity)
            
        # Calculate performance metrics
        equity_series = pd.Series(equity_curve, index=data.index)
        performance = self._calculate_performance_metrics(equity_series)
        trade_analytics = self._analyze_trades()
        
        return BacktestResult(
            total_return=performance['total_return'],
            sharpe_ratio=performance['sharpe_ratio'],
            max_drawdown=performance['max_drawdown'],
            win_rate=performance['win_rate'],
            profit_factor=performance['profit_factor'],
            trades=self.trades,
            equity_curve=equity_series,
            monthly_returns=self._calculate_monthly_returns(equity_series),
            performance_metrics=performance,
            trade_analytics=trade_analytics
        )
        
    def _process_open_positions(self, current_bar: pd.Series, timestamp: datetime):
        """
        Process and update open positions.
        """
        for symbol, position in list(self.positions.items()):
            # Check stop loss
            if current_bar['low'] <= position['stop_loss']:
                self._close_position(symbol, position['stop_loss'], timestamp)
                continue
                
            # Check take profit
            if current_bar['high'] >= position['take_profit']:
                self._close_position(symbol, position['take_profit'], timestamp)
                continue
                
            # Update position metrics
            position['current_price'] = current_bar['close']
            position['unrealized_pnl'] = (
                (position['current_price'] - position['entry_price']) *
                position['quantity']
            )
            
    def _process_signal(
        self,
        signal: pd.Series,
        current_bar: pd.Series,
        timestamp: datetime
    ):
        """
        Process trading signals and execute trades.
        """
        if len(self.positions) >= self.max_positions:
            return
            
        if signal['action'] == 'buy' and signal['confidence'] > 0.7:
            position_size = self.current_capital * self.position_size
            quantity = position_size / current_bar['close']
            
            self.positions[signal['symbol']] = {
                'entry_price': current_bar['close'],
                'quantity': quantity,
                'stop_loss': current_bar['close'] * (1 - self.stop_loss_pct),
                'take_profit': current_bar['close'] * (1 + self.take_profit_pct),
                'entry_time': timestamp,
                'current_price': current_bar['close'],
                'unrealized_pnl': 0
            }
            
    def _close_position(self, symbol: str, exit_price: float, timestamp: datetime):
        """
        Close position and record trade results.
        """
        position = self.positions[symbol]
        profit_loss = (exit_price - position['entry_price']) * position['quantity']
        
        trade = TradeResult(
            entry_time=position['entry_time'],
            exit_time=timestamp,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            profit_loss=profit_loss,
            position_size=position['quantity'] * position['entry_price'],
            trade_duration=timestamp - position['entry_time'],
            max_drawdown=self._calculate_trade_drawdown(position),
            risk_reward_ratio=self._calculate_risk_reward(position, exit_price)
        )
        
        self.trades.append(trade)
        self.current_capital += profit_loss
        del self.positions[symbol]
        
    def _calculate_performance_metrics(self, equity_curve: pd.Series) -> Dict:
        """
        Calculate comprehensive performance metrics.
        """
        returns = equity_curve.pct_change().dropna()
        
        return {
            'total_return': (equity_curve[-1] - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'win_rate': len([t for t in self.trades if t.profit_loss > 0]) / len(self.trades),
            'profit_factor': self._calculate_profit_factor(),
            'volatility': returns.std() * np.sqrt(252),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(equity_curve, returns)
        }
