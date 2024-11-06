# telegram/notifications.py
# backend/telegram/notifications.py

from telegram import Bot, ParseMode
from telegram.error import TelegramError
from typing import Dict, Optional, List
import logging
from datetime import datetime
from backend.appfiles.config import Config

class TradingNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        self.logger = logging.getLogger(__name__)
        
    def send_trade_alert(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        confidence: float,
        indicators: Dict
    ):
        """Send detailed trade execution alert."""
        message = (
            f"🚨 *TRADE ALERT*\n"
            f"{'🟢 BUY' if action.lower() == 'buy' else '🔴 SELL'} {symbol}\n"
            f"Price: ${price:.2f}\n"
            f"Quantity: {quantity:.4f}\n"
            f"Confidence: {confidence:.2%}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"*Key Indicators:*\n"
            f"RSI: {indicators.get('rsi', 'N/A')}\n"
            f"MACD: {indicators.get('macd', 'N/A')}\n"
            f"Signal: {indicators.get('signal', 'N/A')}"
        )
        
        self._send_message(message)

    def send_performance_update(
        self,
        portfolio_value: float,
        daily_pnl: float,
        top_performers: List[Dict],
        risk_metrics: Dict
    ):
        """Send daily performance summary."""
        performance_emoji = "📈" if daily_pnl >= 0 else "📉"
        
        message = (
            f"📊 *DAILY PERFORMANCE UPDATE* {performance_emoji}\n"
            f"Portfolio Value: ${portfolio_value:,.2f}\n"
            f"Daily P&L: {daily_pnl:+,.2f} ({(daily_pnl/portfolio_value):.2%})\n\n"
            f"*Top Performers:*\n"
        )
        
        for performer in top_performers[:3]:
            message += (
                f"• {performer['symbol']}: {performer['return']:.2%}\n"
            )
            
        message += (
            f"\n*Risk Metrics:*\n"
            f"Sharpe Ratio: {risk_metrics.get('sharpe', 'N/A'):.2f}\n"
            f"Max Drawdown: {risk_metrics.get('drawdown', 'N/A'):.2%}\n"
            f"Beta: {risk_metrics.get('beta', 'N/A'):.2f}"
        )
        
        self._send_message(message)

    def send_error_alert(
        self,
        error_type: str,
        details: str,
        severity: str = "HIGH"
    ):
        """Send system error notifications."""
        message = (
            f"⚠️ *SYSTEM ALERT*\n"
            f"Severity: {severity}\n"
            f"Type: {error_type}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Details:\n{details}"
        )
        
        self._send_message(message, priority=True)

    def send_market_analysis(
        self,
        market_conditions: Dict,
        opportunities: List[Dict],
        risks: List[Dict]
    ):
        """Send market analysis and opportunities."""
        message = (
            f"🔍 *MARKET ANALYSIS*\n"
            f"Market Condition: {market_conditions['status']}\n"
            f"Volatility: {market_conditions['volatility']:.2%}\n\n"
            f"*Trading Opportunities:*\n"
        )
        
        for opp in opportunities[:3]:
            message += (
                f"• {opp['symbol']}: {opp['strategy']} "
                f"(Confidence: {opp['confidence']:.2%})\n"
            )
            
        message += "\n*Risk Alerts:*\n"
        for risk in risks:
            message += f"• {risk['description']}\n"
            
        self._send_message(message)

    def _send_message(
        self,
        message: str,
        priority: bool = False,
        retry_count: int = 3
    ):
        """Send message with retry logic and error handling."""
        for attempt in range(retry_count):
            try:
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=True
                )
                break
            except TelegramError as e:
                self.logger.error(f"Telegram error: {str(e)}")
                if attempt == retry_count - 1:
                    self.logger.critical(
                        f"Failed to send message after {retry_count} attempts"
                    )
                    raise

# Initialize the notifier
notifier = TradingNotifier(
    bot_token=Config.TELEGRAM_BOT_TOKEN,
    chat_id=Config.TELEGRAM_CHAT_ID
)
