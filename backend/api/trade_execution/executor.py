# executor.py

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from decimal import Decimal

from .orders import OrderDetails
from .position import PositionManager
from database.db_manager import add_trade
from utils.logger import setup_logger
from backend.appfiles.config import Config
from telegram.notifications import send_notification

logger = setup_logger(__name__)

class TradeExecutor:
    def __init__(self):
        self.order_queue: asyncio.Queue = asyncio.Queue()
        self.position_manager = PositionManager()
        self._in_flight_orders: Dict[str, OrderDetails] = {}
        self._last_execution_time: Dict[str, datetime] = {}

    async def execute_trade(self, signal: Dict[str, Any], risk_params: Dict[str, Any]) -> bool:
        """Execute trade with order management and risk checks."""
        try:
            # Rate limiting check
            symbol = signal['symbol']
            if symbol in self._last_execution_time:
                time_since_last = datetime.now() - self._last_execution_time[symbol]
                if time_since_last.total_seconds() < Config.MIN_ORDER_INTERVAL:
                    logger.warning(f"Rate limit hit for {symbol}")
                    return False

            if not self._validate_signal(signal):
                logger.warning(f"Invalid signal for {symbol}")
                return False

            quantity = self.position_manager.calculate_position_size(
                signal['price'],
                risk_params['risk_per_trade'],
                signal['stop_loss']
            )

            order = OrderDetails(
                symbol=symbol,
                quantity=quantity,
                price=Decimal(str(signal['price'])),
                order_type='limit',
                time_in_force='gtc',
                stop_loss=Decimal(str(signal['stop_loss'])) if signal.get('stop_loss') else None,
                take_profit=Decimal(str(signal['take_profit'])) if signal.get('take_profit') else None
            )

            # Execute main order
            success = await self._execute_order(order)
            if success:
                # Place protective orders
                await self._place_protective_orders(order)
                
                # Record trade and send notification
                await self._record_trade(order, signal)
                await self._send_notification(order, signal)
                
                # Update execution time
                self._last_execution_time[symbol] = datetime.now()
                
                return True

        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            await self._handle_execution_error(signal, e)

        return False

    async def _execute_order(self, order: OrderDetails) -> bool:
        """Execute the main order."""
        try:
            # Add order to tracking
            self._in_flight_orders[order.symbol] = order
            
            # Submit to order queue
            await self.order_queue.put(order)
            
            logger.info(f"Order queued for execution: {order.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Order execution failed: {str(e)}")
            return False
        finally:
            # Clean up tracking
            self._in_flight_orders.pop(order.symbol, None)

    async def _place_protective_orders(self, order: OrderDetails) -> None:
        """Place stop-loss and take-profit orders."""
        try:
            if order.stop_loss:
                stop_order = OrderDetails(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=order.stop_loss,
                    order_type='stop',
                    time_in_force='gtc'
                )
                await self.order_queue.put(stop_order)
                logger.info(f"Stop-loss order placed for {order.symbol}")

            if order.take_profit:
                tp_order = OrderDetails(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=order.take_profit,
                    order_type='limit',
                    time_in_force='gtc'
                )
                await self.order_queue.put(tp_order)
                logger.info(f"Take-profit order placed for {order.symbol}")
                
        except Exception as e:
            logger.error(f"Error placing protective orders: {str(e)}")
            raise

    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate trading signal parameters."""
        try:
            # Check required fields
            required_fields = ['symbol', 'price', 'action', 'confidence']
            if not all(field in signal for field in required_fields):
                logger.warning(f"Missing required fields in signal: {signal}")
                return False

            # Validate confidence threshold
            if signal['confidence'] < Config.MIN_TRADE_CONFIDENCE:
                logger.warning(f"Signal confidence below threshold: {signal['confidence']}")
                return False

            # Check risk limits
            if not self.position_manager.check_risk_limits(signal['symbol']):
                logger.warning(f"Risk limits exceeded for {signal['symbol']}")
                return False

            return True
            
        except Exception as e:
            logger.error(f"Signal validation error: {str(e)}")
            return False

    async def _record_trade(self, order: OrderDetails, signal: Dict[str, Any]) -> None:
        """Record trade details in database."""
        try:
            trade_data = {
                'symbol': order.symbol,
                'quantity': float(order.quantity),
                'price': float(order.price),
                'action': signal['action'],
                'timestamp': datetime.now(),
                'confidence': signal['confidence']
            }
            await add_trade(**trade_data)
            logger.info(f"Trade recorded: {trade_data}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
            raise

    async def _send_notification(self, order: OrderDetails, signal: Dict[str, Any]) -> None:
        """Send detailed trade notification."""
        try:
            message = (
                f"Trade executed:\n"
                f"Symbol: {order.symbol}\n"
                f"Action: {signal['action']}\n"
                f"Quantity: {order.quantity}\n"
                f"Price: {order.price}\n"
                f"Confidence: {signal['confidence']}\n"
                f"Stop Loss: {order.stop_loss}\n"
                f"Take Profit: {order.take_profit}"
            )
            await send_notification(message)
            
        except Exception as e:
            logger.error(f"Notification error: {str(e)}")
            raise

    async def _handle_execution_error(self, signal: Dict[str, Any], error: Exception) -> None:
        """Handle trade execution errors."""
        try:
            error_message = f"Execution error for {signal['symbol']}: {str(error)}"
            logger.error(error_message)
            await send_notification(f"Trade execution failed for {signal['symbol']}")
            
            # Clean up any partial executions or pending orders
            symbol = signal['symbol']
            if symbol in self._in_flight_orders:
                del self._in_flight_orders[symbol]
                
        except Exception as e:
            logger.error(f"Error handling execution error: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Cancel any pending orders
            for order in self._in_flight_orders.values():
                await self.order_queue.put(None)  # Signal to cancel
                
            self._in_flight_orders.clear()
            self._last_execution_time.clear()
            
            logger.info("Trade executor cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")