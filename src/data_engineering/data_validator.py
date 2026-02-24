"""
Data Validator Module
For validating data quality and integrity
"""

import logging
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data quality and integrity
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_completeness(self, data: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, Dict]:
        """
        Check if required columns exist and have acceptable null rates
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, results_dict)
        """
        logger.info("Validating data completeness...")
        
        results = {
            'missing_columns': [],
            'null_percentages': {},
            'is_valid': True
        }
        
        # Check for required columns
        for col in required_columns:
            if col not in data.columns:
                results['missing_columns'].append(col)
                results['is_valid'] = False
        
        # Check null rates
        null_rates = (data.isnull().sum() / len(data) * 100).to_dict()
        results['null_percentages'] = null_rates
        
        return results['is_valid'], results
    
    def validate_consistency(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Check for data consistency issues
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, results_dict)
        """
        logger.info("Validating data consistency...")
        
        results = {
            'duplicate_rows': len(data) - len(data.drop_duplicates()),
            'is_valid': True
        }
        
        if results['duplicate_rows'] > 0:
            logger.warning(f"Found {results['duplicate_rows']} duplicate rows")
        
        return results['is_valid'], results
