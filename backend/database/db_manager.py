# database/db_manager.py
# backend/database/db_manager.py

from typing import List, Dict, Optional
from datetime import datetime
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, and_, or_
from contextlib import asynccontextmanager

from .models import Trade, Signal, Position, Performance
from backend.appfiles.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class DatabaseManager:
    def __init__(self):
        self.engine = create_async_engine(
            Config.ASYNC_DATABASE_URI,
            pool_size=20,
            max_overflow=30,
            pool_timeout=30,
            pool_recycle=1800
        )
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    @asynccontextmanager
    async def session(self):
        """Provide a transactional scope around a series of operations."""
        session = self.async_session()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database transaction error: {str(e)}")
            raise
        finally:
            await session.close()

    async def add_trade(self, trade_data: Dict) -> Trade:
        """Add a new trade with validation and error handling."""
        async with self.session() as session:
            try:
                trade = Trade(**trade_data)
                session.add(trade)
                await session.flush()
                await self._update_position(session, trade)
                await self._calculate_performance(session, trade)
                return trade
            except Exception as e:
                logger.error(f"Error adding trade: {str(e)}")
                raise

    async def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Trade]:
        """Retrieve trades with filtering options."""
        async with self.session() as session:
            query = select(Trade)
            
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            if start_date:
                query = query.filter(Trade.timestamp >= start_date)
            if end_date:
                query = query.filter(Trade.timestamp <= end_date)
                
            query = query.order_by(Trade.timestamp.desc()).limit(limit)
            result = await session.execute(query)
            return result.scalars().all()

    async def _update_position(self, session: AsyncSession, trade: Trade):
        """Update position tracking after trade execution."""
        query = select(Position).filter(Position.symbol == trade.symbol)
        result = await session.execute(query)
        position = result.scalar_one_or_none()

        if position is None:
            position = Position(symbol=trade.symbol)
            session.add(position)

        if trade.action == "buy":
            position.quantity += trade.quantity
            position.average_price = (
                (position.average_price * (position.quantity - trade.quantity) +
                 trade.price * trade.quantity) / position.quantity
            )
        else:
            position.quantity -= trade.quantity

        position.last_updated = datetime.utcnow()
        await session.flush()

    async def _calculate_performance(self, session: AsyncSession, trade: Trade):
        """Calculate and store performance metrics."""
        query = select(Performance).filter(
            and_(
                Performance.symbol == trade.symbol,
                Performance.date == trade.timestamp.date()
            )
        )
        result = await session.execute(query)
        performance = result.scalar_one_or_none()

        if performance is None:
            performance = Performance(
                symbol=trade.symbol,
                date=trade.timestamp.date()
            )
            session.add(performance)

        if trade.action == "sell":
            profit = (trade.price - trade.entry_price) * trade.quantity
            performance.realized_pnl += profit
            performance.trade_count += 1
            performance.win_count += 1 if profit > 0 else 0

        await session.flush()

    async def bulk_insert_trades(self, trades: List[Dict]):
        """Efficiently insert multiple trades."""
        async with self.session() as session:
            session.add_all([Trade(**trade) for trade in trades])

    async def get_performance_metrics(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """Calculate comprehensive performance metrics."""
        async with self.session() as session:
            query = select(Performance)
            
            if symbol:
                query = query.filter(Performance.symbol == symbol)
            if start_date:
                query = query.filter(Performance.date >= start_date)
            if end_date:
                query = query.filter(Performance.date <= end_date)

            result = await session.execute(query)
            performances = result.scalars().all()

            return {
                'total_pnl': sum(p.realized_pnl for p in performances),
                'win_rate': sum(p.win_count for p in performances) / 
                           sum(p.trade_count for p in performances) if performances else 0,
                'trade_count': sum(p.trade_count for p in performances)
            }

    async def cleanup_old_data(self, days: int = 90):
        """Clean up old data while maintaining important records."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        async with self.session() as session:
            await session.execute(
                Trade.__table__.delete().where(Trade.timestamp < cutoff_date)
            )
