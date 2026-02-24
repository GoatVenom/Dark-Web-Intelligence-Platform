"""
Real-Time Threat Detection Pipeline V3
Fixed to work with binary classification model (0=Safe, 1=Threat)
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle
from collections import deque, Counter
import random

class RealTimeThreatDetector:
    """
    Real-time threat detection system for cryptocurrency transactions
    """
    
    def __init__(self, 
                 model_path='output/models/threat_classifier.pkl',
                 scaler_path='output/models/scaler.pkl',
                 alert_threshold=0.3):  # Alert if threat probability > 30%
        
        print("Loading ML models...")
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.n_features = self.scaler.n_features_in_
            
            # Get feature names
            if hasattr(self.scaler, 'feature_names_in_'):
                self.feature_names = list(self.scaler.feature_names_in_)
                print(f"✓ Models loaded successfully")
                print(f"✓ Features required: {self.feature_names}")
            else:
                self.feature_names = None
                print(f"✓ Models loaded successfully ({self.n_features} features)")
            
            # Check model classes
            if hasattr(self.model, 'classes_'):
                print(f"✓ Model classes: {self.model.classes_} (0=Safe, 1=Threat)")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
        
        # Alert configuration
        self.alert_threshold = alert_threshold
        
        print(f"\n⚙️  Alert Configuration:")
        print(f"   Alert when threat probability > {alert_threshold:.0%}")
        
        # Alert queue
        self.alert_queue = deque(maxlen=100)
        
        # Metrics tracking
        self.processed_count = 0
        self.threat_count = 0
        self.alert_count = 0
        self.prediction_distribution = Counter()
        self.start_time = datetime.now()
        
        # Recent transactions
        self.recent_transactions = deque(maxlen=1000)
        
    def process_transaction(self, transaction_data):
        """Process a single transaction in real-time"""
        try:
            # Extract features as DataFrame with EXACT column names
            features_df = self._extract_features(transaction_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Predict threat (0 or 1)
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get threat probability (probability of class 1)
            threat_probability = probabilities[1] if len(probabilities) > 1 else 0
            safe_probability = probabilities[0]
            
            # Determine severity based on threat probability
            if threat_probability >= 0.7:
                severity = 'CRITICAL'
            elif threat_probability >= 0.5:
                severity = 'HIGH'
            elif threat_probability >= 0.3:
                severity = 'MEDIUM'
            else:
                severity = 'LOW'
            
            # Track prediction distribution
            self.prediction_distribution[prediction] += 1
            
            # Create result
            result = {
                'timestamp': datetime.now().isoformat(),
                'transaction_id': transaction_data.get('id', 'unknown'),
                'prediction': int(prediction),  # 0 or 1
                'is_threat': bool(prediction == 1),
                'threat_probability': float(threat_probability),
                'safe_probability': float(safe_probability),
                'severity': severity,
                'amount_usd': transaction_data.get('amount_usd', 0),
                'amount_btc': transaction_data.get('amount_btc', 0),
                'blockchain': transaction_data.get('blockchain', 'unknown'),
                'transaction_type': transaction_data.get('transaction_type', 'unknown')
            }
            
            # Check if alert needed
            should_alert = self._should_trigger_alert(result)
            
            if should_alert:
                self._trigger_alert(result)
                self.alert_count += 1
            
            # Update metrics
            self.processed_count += 1
            if prediction == 1:
                self.threat_count += 1
            
            # Store recent transaction
            self.recent_transactions.append(result)
            
            return result
            
        except Exception as e:
            print(f"Error processing transaction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _should_trigger_alert(self, result):
        """Determine if transaction should trigger an alert"""
        # Alert if threat probability exceeds threshold
        if result['threat_probability'] >= self.alert_threshold:
            return True
        
        # Additional criteria: high-value transaction with moderate threat
        if (result['amount_usd'] > 50000 and 
            result['threat_probability'] >= 0.2):
            return True
        
        return False
    
    def _extract_features(self, transaction_data):
        """Extract ML features matching training data EXACTLY"""
        
        # Create features matching EXACT training columns
        features = {
            'amount_btc': transaction_data.get('amount_btc', 0),
            'amount_usd': transaction_data.get('amount_usd', 0),
            'transaction_type': transaction_data.get('transaction_type_encoded', 0),
            'blockchain': transaction_data.get('blockchain_encoded', 0),
            'confidence_score': transaction_data.get('confidence_score', 0.5),
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Reorder to match training feature names
        if self.feature_names is not None:
            df = df.reindex(columns=self.feature_names, fill_value=0)
        
        return df
    
    def _trigger_alert(self, result):
        """Trigger alert for suspicious transactions"""
        alert = {
            'alert_time': datetime.now().isoformat(),
            'severity': result['severity'],
            'transaction_id': result['transaction_id'],
            'threat_probability': result['threat_probability'],
            'amount_usd': result['amount_usd'],
            'amount_btc': result['amount_btc'],
            'blockchain': result['blockchain'],
            'transaction_type': result['transaction_type'],
            'details': result
        }
        
        self.alert_queue.append(alert)
        
        # Determine alert emoji based on severity
        emoji_map = {
            'CRITICAL': '🔴',
            'HIGH': '🟠',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }
        emoji = emoji_map.get(result['severity'], '⚠️')
        
        # Log alert
        print(f"\n{emoji} ALERT: {result['severity']} threat detected!")
        print(f"   Transaction: {result['transaction_id']}")
        print(f"   Threat Probability: {result['threat_probability']:.1%}")
        print(f"   Amount: ${result['amount_usd']:,.2f} ({result['amount_btc']:.4f} BTC)")
        print(f"   Type: {result['transaction_type']}")
        print(f"   Blockchain: {result['blockchain']}\n")
        
        # Save to alert file
        self._save_alert(alert)
    
    def _save_alert(self, alert):
        """Save alert to file"""
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
            'alerts_generated': self.alert_count,
            'threat_rate': self.threat_count / self.processed_count if self.processed_count > 0 else 0,
            'alert_rate': self.alert_count / self.processed_count if self.processed_count > 0 else 0,
            'throughput_per_sec': throughput,
            'uptime_seconds': elapsed,
            'alerts_queued': len(self.alert_queue),
            'prediction_distribution': dict(self.prediction_distribution)
        }
    
    async def stream_monitor(self, data_source):
        """Monitor streaming data source"""
        print("🔴 Real-time monitoring started...")
        print("="*60)
        print("Monitoring for threats... (Press Ctrl+C to stop)\n")
        
        async for transaction in data_source:
            result = self.process_transaction(transaction)
            
            # Print status every 50 transactions
            if self.processed_count % 50 == 0:
                metrics = self.get_metrics()
                print(f"✓ Processed: {metrics['processed']} | "
                      f"Threats: {metrics['threats_detected']} | "
                      f"Alerts: {metrics['alerts_generated']} | "
                      f"Rate: {metrics['throughput_per_sec']:.1f}/sec")

# Simulated data stream with realistic crypto transactions
async def simulate_transaction_stream():
    """Simulate incoming cryptocurrency transaction data"""
    
    transaction_types = ['transfer', 'purchase', 'exchange', 'withdrawal']
    blockchains = ['bitcoin', 'ethereum', 'litecoin', 'monero']
    
    while True:
        # Generate transactions with varying risk profiles
        risk_roll = random.random()
        
        # BTC price ~ $60,000
        btc_price = random.uniform(58000, 62000)
        
        if risk_roll < 0.6:  # 60% normal transactions
            amount_btc = random.uniform(0.001, 0.1)
            confidence = random.uniform(0.7, 0.95)
        elif risk_roll < 0.85:  # 25% suspicious transactions
            amount_btc = random.uniform(0.1, 1.0)
            confidence = random.uniform(0.4, 0.7)
        else:  # 15% high-risk transactions
            amount_btc = random.uniform(1.0, 10.0)
            confidence = random.uniform(0.1, 0.5)
        
        amount_usd = amount_btc * btc_price
        
        transaction = {
            'id': f'TXN-{random.randint(100000, 999999)}',
            'timestamp': datetime.now().isoformat(),
            'amount_btc': amount_btc,
            'amount_usd': amount_usd,
            'transaction_type': random.choice(transaction_types),
            'transaction_type_encoded': random.randint(0, 3),
            'blockchain': random.choice(blockchains),
            'blockchain_encoded': random.randint(0, 3),
            'confidence_score': confidence,
        }
        
        yield transaction
        await asyncio.sleep(0.05)  # 20 transactions per second

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🕵️  REAL-TIME CRYPTOCURRENCY THREAT DETECTION")
    print("="*60 + "\n")
    
    # Initialize detector
    # Lowering threshold to 0.2 means we alert on 20%+ threat probability
    detector = RealTimeThreatDetector(alert_threshold=0.2)
    
    print()
    
    # Start monitoring
    try:
        asyncio.run(detector.stream_monitor(simulate_transaction_stream()))
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped by user")
        
        # Print detailed final metrics
        metrics = detector.get_metrics()
        print("\n" + "="*60)
        print("📊 FINAL METRICS")
        print("="*60)
        print(f"Total Processed:     {metrics['processed']}")
        print(f"Threats Detected:    {metrics['threats_detected']}")
        print(f"Alerts Generated:    {metrics['alerts_generated']}")
        print(f"Threat Rate:         {metrics['threat_rate']:.2%}")
        print(f"Alert Rate:          {metrics['alert_rate']:.2%}")
        print(f"Avg Throughput:      {metrics['throughput_per_sec']:.1f} transactions/sec")
        print(f"Total Uptime:        {metrics['uptime_seconds']:.1f} seconds")
        
        print(f"\nPrediction Distribution:")
        total = sum(metrics['prediction_distribution'].values())
        for pred, count in sorted(metrics['prediction_distribution'].items()):
            pct = count / total * 100 if total > 0 else 0
            label = "SAFE" if pred == 0 else "THREAT"
            print(f"  {label:10s} (class {pred}): {count:4d} ({pct:5.1f}%)")
        
        print("="*60 + "\n")
        
        # Show alert summary
        alert_file = Path('output/alerts/real_time_alerts.jsonl')
        if alert_file.exists():
            print(f"✓ Alerts saved to: {alert_file}")
            
            # Count alerts by severity
            severity_counts = Counter()
            with open(alert_file, 'r') as f:
                for line in f:
                    try:
                        alert = json.loads(line)
                        severity_counts[alert['severity']] += 1
                    except:
                        pass
            
            if severity_counts:
                print(f"\n  Alert Breakdown:")
                for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                    if severity in severity_counts:
                        print(f"    {severity}: {severity_counts[severity]}")
            
            print(f"\n  View alerts: cat {alert_file}")
            print()
