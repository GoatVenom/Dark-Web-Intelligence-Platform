"""
Real-Time Dark Web Intelligence Dashboard
Live monitoring with WebSocket updates
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import Counter, deque
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'darkweb-intel-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global state
class DashboardState:
    def __init__(self):
        self.alerts = deque(maxlen=100)
        self.metrics = {
            'total_processed': 0,
            'total_threats': 0,
            'total_alerts': 0,
            'threat_rate': 0,
            'alert_rate': 0,
            'uptime': 0
        }
        self.alert_timeline = []
        self.threat_distribution = Counter()
        self.blockchain_stats = Counter()
        self.last_update = datetime.now()

state = DashboardState()

def load_alerts():
    """Load recent alerts from file"""
    alert_file = Path('output/alerts/real_time_alerts.jsonl')
    if not alert_file.exists():
        return []
    
    alerts = []
    cutoff_time = datetime.now() - timedelta(hours=1)
    
    with open(alert_file, 'r') as f:
        for line in f:
            try:
                alert = json.loads(line)
                alert_time = datetime.fromisoformat(alert['alert_time'])
                if alert_time > cutoff_time:
                    alerts.append(alert)
            except:
                continue
    
    return alerts[-100:]  # Last 100 alerts

def monitor_alerts():
    """Background thread to monitor for new alerts"""
    last_size = 0
    alert_file = Path('output/alerts/real_time_alerts.jsonl')
    
    while True:
        try:
            if alert_file.exists():
                current_size = alert_file.stat().st_size
                
                if current_size > last_size:
                    # New data available
                    alerts = load_alerts()
                    
                    if alerts:
                        latest_alert = alerts[-1]
                        
                        # Update state
                        state.alerts.append(latest_alert)
                        state.metrics['total_alerts'] = len(alerts)
                        state.threat_distribution[latest_alert['severity']] += 1
                        state.blockchain_stats[latest_alert.get('blockchain', 'unknown')] += 1
                        
                        # Emit to all connected clients
                        socketio.emit('new_alert', latest_alert)
                        socketio.emit('metrics_update', state.metrics)
                    
                    last_size = current_size
            
            time.sleep(1)  # Check every second
            
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(5)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/alerts')
def get_alerts():
    """API endpoint for recent alerts"""
    alerts = load_alerts()
    return jsonify(alerts)

@app.route('/api/metrics')
def get_metrics():
    """API endpoint for metrics"""
    alerts = load_alerts()
    
    severity_counts = Counter()
    blockchain_counts = Counter()
    
    for alert in alerts:
        severity_counts[alert['severity']] += 1
        blockchain_counts[alert.get('blockchain', 'unknown')] += 1
    
    metrics = {
        'total_alerts': len(alerts),
        'critical_count': severity_counts['CRITICAL'],
        'high_count': severity_counts['HIGH'],
        'medium_count': severity_counts['MEDIUM'],
        'severity_distribution': dict(severity_counts),
        'blockchain_distribution': dict(blockchain_counts),
        'last_update': datetime.now().isoformat()
    }
    
    return jsonify(metrics)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    # Send current state to new client
    emit('initial_data', {
        'alerts': list(state.alerts),
        'metrics': state.metrics
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # Create templates directory
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Start background monitoring thread
    monitor_thread = threading.Thread(target=monitor_alerts, daemon=True)
    monitor_thread.start()
    
    print("\n" + "="*60)
    print("🚀 REAL-TIME INTELLIGENCE DASHBOARD STARTING...")
    print("="*60)
    print("\n📊 Dashboard URL: http://localhost:5000")
    print("🔴 Live monitoring active")
    print("⚡ WebSocket enabled for real-time updates")
    print("\nPress Ctrl+C to stop\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
