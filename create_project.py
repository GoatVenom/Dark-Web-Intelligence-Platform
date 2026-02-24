"""
Project Setup Script
Creates all necessary project files and directories
"""

import os
from pathlib import Path

# Define the project structure
PROJECT_STRUCTURE = {
    'directories': [
        'data/raw',
        'data/processed',
        'data/samples',
        'notebooks',
        'src/data_engineering',
        'src/models',
        'src/graph_analysis',
        'src/intelligence',
        'src/visualization',
        'src/utils',
        'tests',
        'config',
        'output/reports',
        'output/dashboards',
        'output/visualizations',
        'docker'
    ]
}

# File contents
FILES = {
    'requirements.txt': '''# Data Processing & Analysis
pandas==2.0.3
numpy==1.24.3
scipy==1.11.2

# Machine Learning
scikit-learn==1.3.1
xgboost==2.0.0
networkx==3.1

# Database & API
neo4j==5.13.0
sqlalchemy==2.0.21
pymongo==4.5.0
requests==2.31.0

# Visualization
matplotlib==3.8.0
plotly==5.17.0
seaborn==0.12.2

# Blockchain & Crypto Data
web3==6.10.0
ccxt==4.0.0

# Explainability & Interpretation
shap==0.42.1
lime==0.2.0

# MLOps & Utilities
python-dotenv==1.0.0
pydantic==2.4.2
pyyaml==6.0.1
logging-json==0.1.0

# Development & Testing
pytest==7.4.2
pytest-cov==4.1.0
black==23.10.0
flake8==6.1.0
''',

    '.env.example': '''# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=criminal_intelligence
DB_USER=postgres
DB_PASSWORD=your_password_here

# Neo4J Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# MongoDB Configuration
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB=intelligence
MONGO_USER=
MONGO_PASSWORD=

# API Keys (if using blockchain APIs)
ETHERSCAN_API_KEY=your_api_key_here
BLOCKCHAIN_API_KEY=your_api_key_here

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
''',

    'src/__init__.py': '''"""
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
''',

    'src/data_engineering/__init__.py': '''"""
Data Engineering Module
Handles ETL pipelines, data validation, and transformation
"""

from .etl_pipeline import ETLPipeline
from .data_validator import DataValidator

__all__ = ['ETLPipeline', 'DataValidator']
''',

    'src/data_engineering/etl_pipeline.py': '''"""
ETL Pipeline Module
Responsible for Extract, Transform, Load operations
"""

import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ETLPipeline:
    """
    Base ETL Pipeline class for data processing workflows
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize ETL Pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logger
        self.data = None
        self.transformations_applied = []
    
    def extract(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Extract data from source
        
        Args:
            source: Data source (csv, database, api, etc.)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with extracted data
        """
        self.logger.info(f"Extracting data from {source}")
        
        if source.endswith('.csv'):
            self.data = pd.read_csv(source, **kwargs)
        else:
            raise NotImplementedError(f"Source type {source} not yet implemented")
        
        self.logger.info(f"Extracted {len(self.data)} records")
        return self.data
    
    def validate(self, data: pd.DataFrame = None) -> bool:
        """
        Validate data quality
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Boolean indicating if validation passed
        """
        data = data or self.data
        self.logger.info("Validating data...")
        
        # Check for null values
        null_pct = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)
        self.logger.info(f"Null value percentages:\\n{null_pct}")
        
        return True
    
    def transform(self, data: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        Transform data
        
        Args:
            data: DataFrame to transform
            **kwargs: Transformation parameters
            
        Returns:
            Transformed DataFrame
        """
        data = data or self.data
        self.logger.info("Transforming data...")
        
        # Placeholder for actual transformations
        self.transformations_applied.append({
            'timestamp': datetime.now(),
            'operation': 'transform',
            'parameters': kwargs
        })
        
        return data
    
    def load(self, data: pd.DataFrame = None, destination: str = None) -> None:
        """
        Load data to destination
        
        Args:
            data: DataFrame to load
            destination: Destination path or database connection
        """
        data = data or self.data
        self.logger.info(f"Loading data to {destination}")
        
        if destination and destination.endswith('.csv'):
            data.to_csv(destination, index=False)
            self.logger.info(f"Data successfully loaded to {destination}")
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of current data
        
        Returns:
            Dictionary with data summary
        """
        if self.data is None:
            return {}
        
        return {
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2,  # MB
            'dtypes': self.data.dtypes.to_dict(),
            'null_values': self.data.isnull().sum().to_dict()
        }
''',

    'src/data_engineering/data_validator.py': '''"""
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
''',

    'src/models/__init__.py': '''"""
Machine Learning Models Module
Contains threat detection and classification models
"""

from .threat_classifier import ThreatClassifier
from .anomaly_detector import AnomalyDetector

__all__ = ['ThreatClassifier', 'AnomalyDetector']
''',

    'src/models/threat_classifier.py': '''"""
Threat Classifier Model
For classifying transaction and actor threat levels
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ThreatClassifier:
    """
    Machine Learning classifier for threat level prediction
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Threat Classifier
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        self.model = RandomForestClassifier(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', 15),
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for model
        
        Args:
            data: Input DataFrame
            
        Returns:
            Scaled feature array
        """
        logger.info("Preparing features for classification")
        
        # Placeholder for feature engineering
        features = data.select_dtypes(include=[np.number]).fillna(0)
        
        return self.scaler.fit_transform(features)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the classifier
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Training metrics
        """
        logger.info("Training threat classifier...")
        
        self.model.fit(X, y)
        self.is_trained = True
        self.feature_importance = self.model.feature_importances_
        
        train_score = self.model.score(X, y)
        logger.info(f"Model trained. Training accuracy: {train_score:.4f}")
        
        return {'accuracy': train_score}
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
''',

    'src/models/anomaly_detector.py': '''"""
Anomaly Detection Module
For identifying unusual transaction and behavioral patterns
"""

import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Anomaly detection for identifying suspicious patterns
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Anomaly Detector
        
        Args:
            config: Model configuration
        """
        self.config = config or {}
        self.model = IsolationForest(
            contamination=self.config.get('contamination', 0.1),
            random_state=42
        )
        self.is_trained = False
    
    def fit(self, X: np.ndarray) -> Dict:
        """
        Fit the anomaly detection model
        
        Args:
            X: Feature matrix
            
        Returns:
            Fitting metrics
        """
        logger.info("Fitting anomaly detection model...")
        
        self.model.fit(X)
        self.is_trained = True
        
        anomaly_count = sum(self.model.predict(X) == -1)
        logger.info(f"Detected {anomaly_count} anomalies in training data")
        
        return {'anomalies_detected': anomaly_count}
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (-1 for anomaly, 1 for normal) and anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        return predictions, scores
''',

    'src/utils/__init__.py': '''"""
Utilities Module
Helper functions and common utilities
"""

__all__ = []
''',
}

def create_project():
    """Create all project files and directories"""
    
    print("🚀 Creating project structure...")
    
    # Create directories
    print("\n📁 Creating directories...")
    for directory in PROJECT_STRUCTURE['directories']:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    # Create .gitkeep files
    print("\n📝 Creating .gitkeep files...")
    gitkeep_dirs = [
        'data/raw',
        'data/processed',
        'data/samples',
        'output/reports',
        'output/dashboards',
        'output/visualizations'
    ]
    for directory in gitkeep_dirs:
        gitkeep_path = Path(directory) / '.gitkeep'
        gitkeep_path.touch()
        print(f"  ✓ {gitkeep_path}")
    
    # Create files
    print("\n📄 Creating files...")
    for filepath, content in FILES.items():
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ {filepath}")
    
    print("\n✅ Project structure created successfully!")
    print("\nNext steps:")
    print("  1. Create virtual environment: python -m venv venv")
    print("  2. Activate it: source venv/bin/activate")
    print("  3. Install dependencies: pip install -r requirements.txt")

if __name__ == '__main__':
    create_project()