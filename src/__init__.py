"""
Dark Web Criminal Network Intelligence Platform
A comprehensive data science platform for analyzing criminal networks,
cryptocurrency transactions, and threat actor behaviors.
"""

__version__ = "0.1.0"
__author__ = "GoatVenom"

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"Initializing Criminal Intelligence Platform v{__version__}")
