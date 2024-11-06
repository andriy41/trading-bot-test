# database/models.py
# backend/database/models.py

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, 
    Enum, JSON, Date, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()

class TradeStatus(enum.Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    action = Column(String(10), nullable=False)  # buy or sell
    status = Column(Enum(TradeStatus), default=TradeStatus.PENDING)
    confidence_score = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    execution_time = Column(Float)  # in milliseconds
    slippage = Column(Float)
    commission = Column(Float)
    strategy_name = Column(String(50))
    indicators_snapshot = Column(JSON)
    metadata = Column(JSON)
    
    # Relationships
    signals = relationship("Signal", back_populates="trade")
    performance = relationship("Performance", back_populates="trades")
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )

class Signal(Base):
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, ForeignKey('trades.id'))
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    signal_type = Column(String(20), nullable=False)  # buy, sell, neutral
    timeframe = Column(String(10), nullable=False)
    confidence_score = Column(Float, nullable=False)
    price_target = Column(Float)
    stop_loss = Column(Float)
    indicators = Column(JSON)
    ml_predictions = Column(JSON)
    market_context = Column(JSON)
    
    # Relationships
    trade = relationship("Trade", back_populates="signals")
    
    __table_args__ = (
        Index('idx_signal_symbol_timestamp', 'symbol', 'timestamp'),
    )

class Position(Base):
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, unique=True)
    quantity = Column(Float, default=0)
    average_price = Column(Float)
    unrealized_pnl = Column(Float, default=0)
    realized_pnl = Column(Float, default=0)
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_position_symbol', 'symbol'),
    )

class Performance(Base):
    __tablename__ = 'performance'
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    symbol = Column(String(20), nullable=False)
    realized_pnl = Column(Float, default=0)
    unrealized_pnl = Column(Float, default=0)
    trade_count = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    largest_win = Column(Float)
    largest_loss = Column(Float)
    average_win = Column(Float)
    average_loss = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Relationships
    trades = relationship("Trade", back_populates="performance")
    
    __table_args__ = (
        UniqueConstraint('date', 'symbol', name='uq_performance_date_symbol'),
        Index('idx_performance_date_symbol', 'date', 'symbol'),
    )

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    timeframe = Column(String(10), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    indicators = Column(JSON)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', 'timeframe', 
                        name='uq_market_data_symbol_timestamp_timeframe'),
        Index('idx_market_data_symbol_timestamp', 'symbol', 'timestamp'),
    )

class RiskMetrics(Base):
    __tablename__ = 'risk_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    portfolio_value = Column(Float, nullable=False)
    margin_used = Column(Float)
    margin_available = Column(Float)
    risk_exposure = Column(Float)
    var_95 = Column(Float)  # Value at Risk
    expected_shortfall = Column(Float)
    beta = Column(Float)
    correlation_matrix = Column(JSON)
    
    __table_args__ = (
        Index('idx_risk_metrics_timestamp', 'timestamp'),
    )
