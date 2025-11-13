import os
from datetime import timedelta

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = True
    
    # CORS settings
    CORS_HEADERS = 'Content-Type'
    
    # Model paths
    MODEL_DIR = 'models/'
    DATA_DIR = 'data/'
    
    # Logging
    LOG_DIR = 'logs/'
    LOG_FILE = 'app.log'
    LOG_LEVEL = 'INFO'
    
    # API settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size
    JSON_SORT_KEYS = False
    
    # Rate limiting (optional)
    RATELIMIT_ENABLED = False
    RATELIMIT_DEFAULT = "100 per hour"