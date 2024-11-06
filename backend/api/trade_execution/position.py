#position.py
from decimal import Decimal
from typing import Dict, Optional
from datetime import datetime
from backend.appfiles.config import Config
from .orders import OrderDetails
from utils.logger import setup_logger

logger = setup_logger(__name__)

class PositionManager:
    def __init__(self):
        self.positions: Dict[str, Dict] = {}
        self.risk_limits = Config.RISK_LIMITS
        self.sector_cache: Dict[str, str] = {}
        self.sector_exposures: Dict[str, Decimal] = {}
        self._last_update = datetime.now()

    def calculate_position_size(
        self,
        price: float,
        risk_amount: float,
        stop_loss: float,
        volatility: Optional[float] = None
    ) -> Decimal:
        """Calculate position size with volatility adjustment."""
        try:
            risk_per_share = abs(price - stop_loss)
            
            # Adjust risk based on volatility if provided
            if volatility:
                risk_adjustment = 1.0 - min(volatility, 0.5)  # Cap volatility impact
                risk_amount *= risk_adjustment
                
            quantity = Decimal(risk_amount / risk_per_share)
            return self._adjust_for_limits(quantity)
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return Decimal('0')

    def update_position(self, symbol: str, order: OrderDetails) -> None:
        """Update position after order execution."""
        try:
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': Decimal('0'),
                    'avg_price': Decimal('0'),
                    'value': Decimal('0'),
                    'unrealized_pnl': Decimal('0'),
                    'sector': self.get_sector(symbol)
                }
                
            position = self.positions[symbol]
            old_value = position['quantity'] * position['avg_price']
            new_value = order.quantity * order.price
            
            position['quantity'] += order.quantity
            
            if position['quantity'] != Decimal('0'):
                position['avg_price'] = (old_value + new_value) / position['quantity']
            position['value'] = position['quantity'] * position['avg_price']
            
            # Update sector exposure
            self._update_sector_exposure(symbol, position['value'])
            self._last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")
            raise

    def check_risk_limits(self, symbol: str, order_value: Optional[Decimal] = None) -> bool:
        """Check if order complies with risk limits."""
        try:
            # Current portfolio exposure
            current_exposure = sum(pos['value'] for pos in self.positions.values())
            
            # Add potential new exposure
            if order_value:
                total_exposure = current_exposure + order_value
            else:
                total_exposure = current_exposure
                
            # Check multiple risk limits
            if total_exposure > self.risk_limits['max_portfolio_risk']:
                logger.warning(f"Portfolio risk limit exceeded: {total_exposure}")
                return False
                
            # Check concentration limits
            if symbol in self.positions:
                position_exposure = self.positions[symbol]['value']
                if order_value:
                    position_exposure += order_value
                if position_exposure > self.risk_limits['max_position_size']:
                    logger.warning(f"Position size limit exceeded for {symbol}")
                    return False
                    
            # Check sector limits
            sector = self.get_sector(symbol)
            if sector and 'max_sector_exposure' in self.risk_limits:
                sector_exposure = self.get_sector_exposure(sector)
                if sector_exposure > self.risk_limits['max_sector_exposure']:
                    logger.warning(f"Sector exposure limit exceeded for {sector}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return False

    def _adjust_for_limits(self, quantity: Decimal) -> Decimal:
        """Adjust position size for risk limits and lot sizes."""
        try:
            max_position = Decimal(str(self.risk_limits['max_position_size']))
            min_position = Decimal(str(self.risk_limits.get('min_position_size', '0')))
            
            # Round to nearest lot size
            lot_size = Decimal(str(self.risk_limits.get('lot_size', '0.01')))
            quantity = round(quantity / lot_size) * lot_size
            
            # Apply position limits
            quantity = max(min_position, min(quantity, max_position))
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error adjusting position size: {str(e)}")
            return Decimal('0')

    def get_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a symbol."""
        try:
            if symbol in self.sector_cache:
                return self.sector_cache[symbol]
                
            # Here you would typically implement sector lookup logic
            # For example, querying an external service or database
            # For now, we'll return None
            return None
            
        except Exception as e:
            logger.error(f"Error getting sector for {symbol}: {str(e)}")
            return None

    def get_sector_exposure(self, sector: str) -> Decimal:
        """Get total exposure for a sector."""
        try:
            return self.sector_exposures.get(sector, Decimal('0'))
        except Exception as e:
            logger.error(f"Error getting sector exposure: {str(e)}")
            return Decimal('0')

    def _update_sector_exposure(self, symbol: str, value: Decimal) -> None:
        """Update sector exposure tracking."""
        try:
            sector = self.get_sector(symbol)
            if sector:
                old_value = self.sector_exposures.get(sector, Decimal('0'))
                if symbol in self.positions:
                    old_value -= self.positions[symbol]['value']
                self.sector_exposures[sector] = old_value + value
                
        except Exception as e:
            logger.error(f"Error updating sector exposure: {str(e)}")

    def get_portfolio_stats(self) -> Dict:
        """Get current portfolio statistics."""
        try:
            return {
                'total_value': sum(pos['value'] for pos in self.positions.values()),
                'position_count': len(self.positions),
                'top_positions': sorted(
                    self.positions.items(),
                    key=lambda x: x[1]['value'],
                    reverse=True
                )[:5],
                'sector_exposures': dict(self.sector_exposures),
                'last_update': self._last_update
            }
        except Exception as e:
            logger.error(f"Error getting portfolio stats: {str(e)}")
            return {}

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.positions.clear()
            self.sector_cache.clear()
            self.sector_exposures.clear()
            logger.info("Position manager cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")