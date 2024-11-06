# orders.py - Enhanced order validation and utilities
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Dict
from datetime import datetime

@dataclass
class OrderDetails:
    symbol: str
    quantity: Decimal
    price: Decimal
    order_type: str
    time_in_force: str
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate order parameters."""
        try:
            if self.quantity <= 0 or self.price <= 0:
                return False
            if self.order_type not in {'market', 'limit', 'stop', 'stop_limit'}:
                return False
            if self.time_in_force not in {'day', 'gtc', 'ioc', 'fok'}:
                return False
            if self.stop_loss and self.stop_loss >= self.price:
                return False
            if self.take_profit and self.take_profit <= self.price:
                return False
            return True
        except Exception:
            return False

    def to_dict(self) -> Dict:
        """Convert order to dictionary format."""
        return {
            'symbol': self.symbol,
            'quantity': float(self.quantity),
            'price': float(self.price),
            'type': self.order_type,
            'time_in_force': self.time_in_force,
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': float(self.take_profit) if self.take_profit else None,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }