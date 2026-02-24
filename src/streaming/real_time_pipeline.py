"""
Real-Time Threat Detection Pipeline
Monitors live cryptocurrency transactions and marketplace activity
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle
from collections import deque
import random

class RealTimeThreatDetector:
    """
    Real-time threat detection system for dark web intelligence
    """
    
    def __init__(self, model_path='output/models/threat_classifier.pkl',
                 scaler_path='output/models/scaler.pkl'):
        # Load trained models
        print("Loading ML models...")
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Get expected features
            self.n_features = self.scaler.n_features_in_
            
            # Get feature names if available
            if hasattr(self.scaler, 'feature_names_in_'):
                self.feature_names = list(self.scaler.feature_names_in_)
                print(f"✓ Models loaded successfully")
                print(f"✓ Features: {self.feature_names}")
            else:
                self.feature_names = None
                print(f"✓ Models loaded successfully ({self.n_features} features)")
            
        except Exception as e:
            print(f"Warning: Could not load models - {e}")
            print("Running in demo mode...")
            self.model = None
            self.scaler = None
            self.n_features = 5
            self.feature_names = None
        
        # Alert queue for high-priority threats
        self.alert_queue = deque(maxlen=100)
        
        # Metrics tracking
        self.processed_count = 0
        self.threat_count = 0
        self.start_time = datetime.now()
        
        # Real-time storage
        self.recent_transactions = deque(maxlen=1000)
        
    def process_transaction(self, transaction_data):
        """
        Process a single transaction in real-time
        
        Args:
            transaction_data: dict with transaction details
            
        Returns:
            dict with threat assessment
        """
        try:
            # Simulate threat detection if models not available
            if self.model is None:
                threat_level = random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
                confidence = random.uniform(0.6, 0.99)
            else:
                # Extract features as DataFrame with proper column names
                features_df = self._extract_features(transaction_data)
                
                # Scale features
                features_scaled = self.scaler.transform(features_df)
                
                # Predict threat level
                threat_level = self.model.predict(features_scaled)[0]
                threat_proba = self.model.predict_proba(features_scaled)[0]
                confidence = max(threat_proba)
            
            # Create result
            result = {
                'timestamp': datetime.now().isoformat(),
                'transaction_id': transaction_data.get('id', 'unknown'),
                'threat_level': threat_level,
                'confidence': confidence,
                'amount': transaction_data.get('amount', 0),
                'sender': transaction_data.get('sender', 'unknown'),
                'receiver': transaction_data.get('receiver', 'unknown')
            }
            
            # Check if alert needed
            if threat_level in ['HIGH', 'CRITICAL'] and confidence > 0.8:
                self._trigger_alert(result)
            
            # Update metrics
            self.processed_count += 1
            if threat_level in ['HIGH', 'CRITICAL']:
                self.threat_count += 1
            
            # Store recent transaction
            self.recent_transactions.append(result)
            
            return result
            
        except Exception as e:
            print(f"Error processing transaction: {e}")
            return None
    
    def _extract_features(self, transaction_data):
        """Extract ML features as DataFrame with proper column names"""
        
        # Create feature dictionary
        features = {
            'amount': transaction_data.get('amount', 0),
            'risk_score': transaction_data.get('risk_score', 0),
            'sender_length': len(transaction_data.get('sender', '')),
            'transaction_type_encoded': transaction_data.get('transaction_type_encoded', 0),
            'fees': transaction_data.get('fees', 0),
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # If we know the exact feature names, reorder to match
        if self.feature_names is not None:
            # Use only the features the model expects
            df = df.reindex(columns=self.feature_names, fill_value=0)
        else:
            # Otherwise ensure we have the right number of columns
            if len(df.columns) < self.n_features:
                # Add missing columns
                for i in range(len(df.columns), self.n_features):
                    df[f'feature_{i}'] = 0
            elif len(df.columns) > self.n_features:
                # Keep only first n_features
                df = df.iloc[:, :self.n_features]
        
        return df
    
    def _trigger_alert(self, result):
        """Trigger alert for high-priority threats"""
        alert = {
            'alert_time': datetime.now().isoformat(),
            'severity': result['threat_level'],
            'confidence': result['confidence'],
            'transaction_id': result['transaction_id'],
            'details': result
        }
        
        self.alert_queue.append(alert)
        
        # Log alert
        print(f"\n🚨 ALERT: {result['threat_level']} threat detected!")
        print(f"   Transaction: {result['transaction_id']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Amount: ${result['amount']:,.2f}\n")
        
        # Save to alert file
        self._save_alert(alert)
    
    def _save_alert(self, alert):
        """Save alert to file for review"""
        alert_file = Path('output/alerts/real_time_alerts.jsonl')
        alert_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert) + '\n')
    
    def get_metrics(self):
        """Get real-time performance metrics"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        throughput = self.processed_count / elapsed if elapsed > 0 else 0
        
        return {
            'processed': self.processed_count,
            'threats_detected': self.threat_count,
            'threat_rate': self.threat_count / self.processed_count if self.processed_count > 0 else 0,
            'throughput_per_sec': throughput,
            'uptime_seconds': elapsed,
            'alerts_queued': len(self.alert_queue)
        }
    
    async def stream_monitor(self, data_source):
        """
        Monitor streaming data source
        
        Args:
            data_source: async generator yielding transactions
        """
        print("🔴 Real-time monitoring started...")
        print("="*60)
        print("Monitoring for threats... (Press Ctrl+C to stop)\n")
        
        async for transaction in data_source:
            result = self.process_transaction(transaction)
            
            # Print status every 100 transactions
            if self.processed_count % 100 == 0:
                metrics = self.get_metrics()
                print(f"✓ Processed: {metrics['processed']} | "
                      f"Threats: {metrics['threats_detected']} | "
                      f"Rate: {metrics['throughput_per_sec']:.1f}/sec")

# Simulated data stream for testing
async def simulate_transaction_stream():
    """Simulate incoming transaction data"""
    transaction_types = ['PURCHASE', 'TRANSFER', 'WITHDRAWAL', 'MARKETPLACE']
    
    while True:
        transaction = {
            'id': f'TXN-{random.randint(10000, 99999)}',
            'timestamp': datetime.now().isoformat(),
            'amount': random.uniform(100, 50000),
            'sender': f'wallet_{random.randint(1, 1000)}',
            'receiver': f'wallet_{random.randint(1, 1000)}',
            'transaction_type': random.choice(transaction_types),
            'risk_score': random.uniform(0, 1),
            'transaction_type_encoded': random.randint(0, 3),
            'fees': random.uniform(0.5, 5.0),
        }
        
        yield transaction
        await asyncio.sleep(0.1)  # 10 transactions per second

# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🕵️  REAL-TIME THREAT DETECTION SYSTEM")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = RealTimeThreatDetector()
    
    print()
    
    # Start monitoring (simulated stream)
    try:
        asyncio.run(detector.stream_monitor(simulate_transaction_stream()))
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped by user")
        
        # Print final metrics
        metrics = detector.get_metrics()
        print("\n" + "="*60)
        print("📊 FINAL METRICS")
        print("="*60)
        print(f"Total Processed:     {metrics['processed']}")
        print(f"Threats Detected:    {metrics['threats_detected']}")
        print(f"Threat Rate:         {metrics['threat_rate']:.2%}")
        print(f"Avg Throughput:      {metrics['throughput_per_sec']:.1f} transactions/sec")
        print(f"Total Uptime:        {metrics['uptime_seconds']:.1f} seconds")
        print(f"Alerts Generated:    {metrics['alerts_queued']}")
        print("="*60 + "\n")
        
        # Show where alerts are saved
        alert_file = Path('output/alerts/real_time_alerts.jsonl')
        if alert_file.exists():
            print(f"✓ Alerts saved to: {alert_file}")
            print(f"  View with: cat {alert_file}\n")
