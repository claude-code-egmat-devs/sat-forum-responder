"""
Configuration settings for SAT Forum Responder
"""

import os
import json
import secrets
from pathlib import Path

# Base directories
SRC_DIR = Path(__file__).parent.parent  # src/
PROJECT_DIR = SRC_DIR.parent  # sat-forum-responder/


class Config:
    """Application configuration"""

    # ==========================================================================
    # SERVER SETTINGS
    # ==========================================================================
    HOST = '0.0.0.0'
    PORT = 5004  # Different port from GMAT (5003)
    DEBUG = False

    # ==========================================================================
    # WEBHOOK SETTINGS
    # ==========================================================================
    WEBHOOK_ENDPOINT = '/webhook'
    HEALTH_ENDPOINT = '/health'
    STATS_ENDPOINT = '/stats'
    LOCATION_PATH = 'sat-forum-webhook'

    # ==========================================================================
    # SECURITY SETTINGS
    # ==========================================================================
    API_KEY_HEADER = 'X-Webhook-API-Key'
    API_KEY_HEADERS = ['X-Webhook-API-Key', 'X-API-Key', 'Authorization']

    @staticmethod
    def get_webhook_api_key():
        key_file = PROJECT_DIR / 'keys' / 'webhook_api_key.txt'
        if key_file.exists():
            return key_file.read_text().strip()
        else:
            new_key = secrets.token_urlsafe(32)
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.write_text(new_key)
            return new_key

    # ==========================================================================
    # PROCESSING SETTINGS
    # ==========================================================================
    MAX_WORKERS = 4
    REQUEST_TIMEOUT = 30
    PROCESSING_TIMEOUT = 300

    # ==========================================================================
    # DATABASE SETTINGS
    # ==========================================================================
    WEBHOOK_DB_PATH = PROJECT_DIR / 'db' / 'webhooks.db'

    # ==========================================================================
    # LOGGING SETTINGS
    # ==========================================================================
    LOG_DIR = PROJECT_DIR / 'logs'
    LOG_FILE = LOG_DIR / 'sat_forum.log'
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # ==========================================================================
    # PROMPTS DIRECTORY
    # ==========================================================================
    PROMPTS_DIR = PROJECT_DIR / 'prompts'

    # ==========================================================================
    # RATE LIMITING
    # ==========================================================================
    RATE_LIMIT_PER_MINUTE = 60
    MAX_QUEUE_SIZE = 100

    # ==========================================================================
    # API CONFIGURATIONS
    # ==========================================================================
    @staticmethod
    def _load_api_keys():
        """Load API keys from shared keys file"""
        keys_file = PROJECT_DIR / 'keys' / 'api_keys.json'
        if keys_file.exists():
            with open(keys_file, 'r') as f:
                return json.load(f)
        return {}

    @staticmethod
    def get_anthropic_config():
        return Config._load_api_keys().get('anthropic', {})

    @staticmethod
    def get_airtable_config():
        return Config._load_api_keys().get('airtable', {})

    @staticmethod
    def get_forum_post_api_config():
        return Config._load_api_keys().get('forum_post_api', {})

    @staticmethod
    def get_neuron_get_api_config():
        return Config._load_api_keys().get('neuron_get_api', {})

    @staticmethod
    def get_forum_api_config():
        return Config._load_api_keys().get('forum_api', {})

    @staticmethod
    def get_teams_notification_config():
        return Config._load_api_keys().get('teams_notification', {})


# Create singleton instance
config = Config()

# Initialize API key on import
WEBHOOK_API_KEY = Config.get_webhook_api_key()


if __name__ == '__main__':
    print("=" * 60)
    print("SAT Forum Responder - Configuration")
    print("=" * 60)
    print(f"Source Directory: {SRC_DIR}")
    print(f"Project Directory: {PROJECT_DIR}")
    print(f"Server: {config.HOST}:{config.PORT}")
    print(f"Database: {config.WEBHOOK_DB_PATH}")
    print(f"Prompts Directory: {config.PROMPTS_DIR}")
    print(f"Log File: {config.LOG_FILE}")
    print("=" * 60)
