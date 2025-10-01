#!/usr/bin/env python3
"""
🔒 SECURE FIREBOLT CONFIGURATION
=================================
Environment variable-based configuration for Firebolt Cloud connection.
No hardcoded credentials - all sensitive data from environment variables.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class FireboltConfig:
    """Secure configuration management for Firebolt connections"""
    
    def __init__(self):
        """Initialize configuration from environment variables"""
        
        # 🌐 Firebolt Cloud Configuration
        self.CLOUD_CLIENT_ID = self._get_env_var('FIREBOLT_CLIENT_ID')
        self.CLOUD_CLIENT_SECRET = self._get_env_var('FIREBOLT_CLIENT_SECRET')
        self.CLOUD_ACCOUNT = self._get_env_var('FIREBOLT_ACCOUNT', default='se-demo-account')
        self.CLOUD_DATABASE = self._get_env_var('FIREBOLT_DATABASE', default='experimental_john')
        self.CLOUD_ENGINE = self._get_env_var('FIREBOLT_ENGINE', default='ecommerceengine')
        self.CLOUD_TABLE = self._get_env_var('FIREBOLT_TABLE', default='home_depot_semantic_search')
        
        # 🤖 AI/Embedding Configuration  
        self.OLLAMA_URL = self._get_env_var('OLLAMA_URL', default='http://localhost:11434')
        self.EMBEDDING_MODEL = self._get_env_var('EMBEDDING_MODEL', default='nomic-embed-text')
        self.EMBEDDING_DIMENSIONS = int(self._get_env_var('EMBEDDING_DIMENSIONS', default='768'))
        
        # 🚀 Application Configuration
        self.APP_PORT = int(self._get_env_var('STREAMLIT_PORT', default='8501'))
        self.DEBUG_MODE = self._get_env_var('DEBUG_MODE', default='false').lower() == 'true'
        
        # Validate required credentials are present
        self._validate_config()
    
    def _get_env_var(self, key: str, default: Optional[str] = None) -> str:
        """Get environment variable with optional default"""
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def _validate_config(self):
        """Validate that all required configuration is present"""
        required_vars = [
            ('FIREBOLT_CLIENT_ID', self.CLOUD_CLIENT_ID),
            ('FIREBOLT_CLIENT_SECRET', self.CLOUD_CLIENT_SECRET),
        ]
        
        missing_vars = []
        for var_name, value in required_vars:
            if not value or value == 'NOT_SET':
                missing_vars.append(var_name)
        
        if missing_vars:
            error_msg = f"""
🔒 MISSING REQUIRED ENVIRONMENT VARIABLES:
{chr(10).join(f'  - {var}' for var in missing_vars)}

Please set these in your .env file or environment:
export FIREBOLT_CLIENT_ID="your_client_id"
export FIREBOLT_CLIENT_SECRET="your_client_secret"

See .env.example for a complete template.
"""
            raise ValueError(error_msg)
    
    def get_firebolt_connection_params(self) -> dict:
        """Get connection parameters for Firebolt SDK"""
        return {
            'client_id': self.CLOUD_CLIENT_ID,
            'client_secret': self.CLOUD_CLIENT_SECRET,
            'account_name': self.CLOUD_ACCOUNT,
            'database': self.CLOUD_DATABASE,
            'engine_name': self.CLOUD_ENGINE
        }
    
    def print_config_summary(self):
        """Print configuration summary (without sensitive data)"""
        print("🔒 FIREBOLT CONFIGURATION SUMMARY")
        print("=" * 40)
        print(f"Account: {self.CLOUD_ACCOUNT}")
        print(f"Database: {self.CLOUD_DATABASE}")  
        print(f"Engine: {self.CLOUD_ENGINE}")
        print(f"Table: {self.CLOUD_TABLE}")
        print(f"Ollama: {self.OLLAMA_URL}")
        print(f"Embedding Model: {self.EMBEDDING_MODEL}")
        print(f"Credentials: {'✅ Set' if self.CLOUD_CLIENT_ID else '❌ Missing'}")

# Global configuration instance
config = FireboltConfig()

if __name__ == "__main__":
    config.print_config_summary()
