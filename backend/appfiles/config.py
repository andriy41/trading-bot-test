# backend/config.py

import os
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class APIConfig:
    ALPHA_VANTAGE_API_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    FINNHUB_API_KEY: str = os.getenv('FINNHUB_API_KEY', '')
    POLYGON_API_KEY: str = os.getenv('POLYGON_API_KEY', '')
    TELEGRAM_BOT_TOKEN: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.getenv('TELEGRAM_CHAT_ID', '')

@dataclass
class DatabaseConfig:
    HOST: str = os.getenv('DB_HOST', 'localhost')
    PORT: int = int(os.getenv('DB_PORT', 5432))
    NAME: str = os.getenv('DB_NAME', 'trading_bot')
    USER: str = os.getenv('DB_USER', 'username')
    PASSWORD: str = os.getenv('DB_PASSWORD', 'password')
    
    @property
    def URI(self) -> str:
        return f'postgresql://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.NAME}'

@dataclass
class RedisConfig:
    HOST: str = os.getenv('REDIS_HOST', 'localhost')
    PORT: int = int(os.getenv('REDIS_PORT', 6379))
    DB: int = int(os.getenv('REDIS_DB', 0))
    
    @property
    def URL(self) -> str:
        return f"redis://{self.HOST}:{self.PORT}/{self.DB}"

class Config:
    # API Configuration
    API = APIConfig()
    
    # Database Configuration
    DB = DatabaseConfig()
    SQLALCHEMY_DATABASE_URI = DB.URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Redis Configuration
    REDIS = RedisConfig()
    REDIS_URL = REDIS.URL
    
    # Rate Limiting Configuration
    RATE_LIMIT: Dict[str, Dict[str, int]] = {
        'alpha_vantage': {
            'requests_per_minute': 5,
            'daily_limit': 500,
            'burst_limit': 10
        },
        'finnhub': {
            'requests_per_minute': 60,
            'burst_limit': 30
        },
        'polygon': {
            'requests_per_minute': 100,
            'daily_limit': 1000
        }
    }
    
    # Security Configuration
    SECRET_KEY: str = os.getenv('SECRET_KEY', os.urandom(24).hex())
    JWT_SECRET_KEY: str = os.getenv('JWT_SECRET_KEY', os.urandom(24).hex())
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    
    # Trading Configuration
    TRADING_PAIRS: List[str] = ['BTC/USD', 'ETH/USD', 'XRP/USD']
    MAX_POSITIONS: int = 5
    POSITION_SIZE_LIMIT: float = 0.1  # 10% of portfolio
    STOP_LOSS_THRESHOLD: float = 0.02  # 2%
    TAKE_PROFIT_THRESHOLD: float = 0.05  # 5%
    
    # Monitoring Configuration
    ENABLE_TELEGRAM_NOTIFICATIONS: bool = True
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    MONITORING_INTERVAL: int = 60  # seconds
    
    # Model Configuration
    MODEL_CHECKPOINT_DIR: str = 'models/checkpoints'
    FEATURE_COLUMNS: List[str] = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'ema_9', 'ema_21', 'bb_upper',
        'bb_lower', 'atr'
    ]
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate critical configuration settings."""
        required_settings = [
            cls.API.ALPHA_VANTAGE_API_KEY,
            cls.API.FINNHUB_API_KEY,
            cls.DB.URI,
            cls.SECRET_KEY
        ]
        
        return all(required_settings)

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    LOG_LEVEL = 'WARNING'
    
    # Enhanced security settings for production
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    PERMANENT_SESSION_LIFETIME = 3600

class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    
    # Test-specific settings
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

# Set the active configuration
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

active_config = config[os.getenv('FLASK_ENV', 'default')]
