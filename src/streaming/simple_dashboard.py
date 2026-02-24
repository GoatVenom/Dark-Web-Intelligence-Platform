"""
Simplified Real-Time Dashboard
"""

from flask import Flask, render_template_string, jsonify
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import Counter

app = Flask(__name__)

def load_alerts():
    """Load recent alerts from file"""
    alert_file = Path('output/alerts/real_time_alerts.jsonl')
    if not alert_file.exists():
        return []
    
    alerts = []
    cutoff_time = datetime.now() - timedelta(hours=1)
    
    try:
        with open(alert_file, 'r') as f:
            for line in f:
                try:
                    alert = json.loads(line)
                    alert_time = datetime.fromisoformat(alert['alert_time'])
                    if alert_time > cutoff_time:
                        alerts.append(alert)
                except Exception as e:
                    continue
    except FileNotFoundError:
        pass
    
    return alerts[-100:]

@app.route('/')
def dashboard():
    """Main dashboard"""
    html = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="5">
    <title>Dark Web Intelligence Platform</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
            color: #fff;
            margin: 0;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .live-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #00ff88;
            border-radius: 50%;
            animation: pulse 2s infinite;
            margin-right: 8px;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: rgba(26, 31, 58, 0.9);
            padding: 25px;
            border-radius: 15px;
            border-left: 5px solid #00ff88;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .metric-card.critical {
            border-left-color: #ff0055;
        }
        .metric-card.high {
            border-left-color: #ff8800;
        }
        .metric-card.medium {
            border-left-color: #ffdd00;
        }
        .metric-value {
            font-size: 3em;
            color: #00ff88;
            font-weight: bold;
        }
        .metric-card.critical .metric-value {
            color: #ff0055;
        }
        .metric-card.high .metric-value {
            color: #ff8800;
        }
        .metric-card.medium .metric-value {
            color: #ffdd00;
        }
        .metric-label {
            color: #aaa;
            margin-top: 10px;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 1px;
        }
        .alert-section {
            background: rgba(26, 31, 58, 0.9);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .alert-section h2 {
            color: #00d4ff;
            margin-bottom: 20px;
        }
        .alert-item {
            background: rgba(15, 20, 25, 0.8);
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #ff0055;
        }
        .critical { 
            border-left-color: #ff0055; 
            background: rgba(255, 0, 85, 0.1);
        }
        .high { 
            border-left-color: #ff8800;
            background: rgba(255, 136, 0, 0.1);
        }
        .medium { 
            border-left-color: #ffdd00;
            background: rgba(255, 221, 0, 0.1);
        }
        .low { 
            border-left-color: #00ff88;
            background: rgba(0, 255, 136, 0.1);
        }
        .alert-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .alert-severity {
            font-weight: bold;
            font-size: 1.1em;
        }
        .alert-time {
            color: #888;
            font-size: 0.9em;
        }
        .alert-details {
            color: #ccc;
            line-height: 1.5;
        }
        .status-bar {
            background: rgba(15, 20, 25, 0.8);
            padding: 10px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .empty-state {
            text-align: center;
            color: #888;
            padding: 60px 20px;
            background: rgba(15, 20, 25, 0.5);
            border-radius: 10px;
            margin: 20px 0;
        }
        .empty-state code {
            background: rgba(0, 255, 136, 0.1);
            padding: 5px 10px;
            border-radius: 5px;
            color: #00ff88;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><span class="live-indicator"></span>Dark Web Intelligence Platform</h1>
        <p>Real-Time Threat Detection & Monitoring</p>
    </div>
    
    <div class="status-bar">
        <span class="live-indicator"></span>
        <strong>LIVE MONITORING</strong> - Auto-refresh every 5 seconds - 
        Last update: {{ current_time }}
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{{ total_alerts }}</div>
            <div class="metric-label">Total Alerts (1h)</div>
        </div>
        <div class="metric-card critical">
            <div class="metric-value">{{ critical }}</div>
            <div class="metric-label">Critical Threats</div>
        </div>
        <div class="metric-card high">
            <div class="metric-value">{{ high }}</div>
            <div class="metric-label">High Priority</div>
        </div>
        <div class="metric-card medium">
            <div class="metric-value">{{ medium }}</div>
            <div class="metric-label">Medium Priority</div>
        </div>
    </div>
    
    <div class="alert-section">
        <h2>🚨 Recent Alerts</h2>
        
        {% if alerts|length > 0 %}
            {% for alert in alerts[:20] %}
            <div class="alert-item {{ alert.severity|lower }}">
                <div class="alert-header">
                    <span class="alert-severity">
                        {% if alert.severity == 'CRITICAL' %}🔴{% endif %}
                        {% if alert.severity == 'HIGH' %}🟠{% endif %}
                        {% if alert.severity == 'MEDIUM' %}🟡{% endif %}
                        {% if alert.severity == 'LOW' %}🟢{% endif %}
                        {{ alert.severity }} THREAT
                    </span>
                    <span class="alert-time">{{ alert.time_display }}</span>
                </div>
                <div class="alert-details">
                    <strong>Transaction:</strong> {{ alert.transaction_id }}<br>
                    <strong>Amount:</strong> ${{ alert.amount_usd_display }} ({{ alert.amount_btc_display }} BTC)<br>
                    <strong>Threat Probability:</strong> {{ alert.threat_pct }}%<br>
                    <strong>Blockchain:</strong> {{ alert.blockchain }} | 
                    <strong>Type:</strong> {{ alert.transaction_type }}
                </div>
            </div>
            {% endfor %}
        {% else %}
            <div class="empty-state">
                <h3>No Alerts Yet</h3>
                <p>The system is ready and monitoring for threats.</p>
                <p>Start the detector in a separate terminal:</p>
                <code>python src/streaming/real_time_pipeline_v3.py</code>
            </div>
        {% endif %}
    </div>
    
</body>
</html>
    '''
    
    alerts = load_alerts()
    
    # Count by severity - safely handle missing keys
    severity_counts = {
        'CRITICAL': 0,
        'HIGH': 0,
        'MEDIUM': 0,
        'LOW': 0
    }
    
    # Process alerts for display
    processed_alerts = []
    for alert in alerts:
        severity = alert.get('severity', 'UNKNOWN')
        if severity in severity_counts:
            severity_counts[severity] += 1
        
        # Format alert for display
        try:
            alert_time = datetime.fromisoformat(alert['alert_time'])
            time_display = alert_time.strftime('%H:%M:%S')
        except:
            time_display = 'N/A'
        
        processed_alert = {
            'severity': severity,
            'transaction_id': alert.get('transaction_id', 'N/A'),
            'amount_usd': alert.get('amount_usd', 0),
            'amount_usd_display': f"{alert.get('amount_usd', 0):,.2f}",
            'amount_btc': alert.get('amount_btc', 0),
            'amount_btc_display': f"{alert.get('amount_btc', 0):.4f}",
            'threat_probability': alert.get('threat_probability', 0),
            'threat_pct': f"{alert.get('threat_probability', 0) * 100:.1f}",
            'blockchain': alert.get('blockchain', 'unknown'),
            'transaction_type': alert.get('transaction_type', 'unknown'),
            'time_display': time_display
        }
        processed_alerts.append(processed_alert)
    
    return render_template_string(
        html,
        alerts=processed_alerts,
        total_alerts=len(alerts),
        critical=severity_counts['CRITICAL'],
        high=severity_counts['HIGH'],
        medium=severity_counts['MEDIUM'],
        current_time=datetime.now().strftime('%H:%M:%S')
    )

@app.route('/api/alerts')
def get_alerts_api():
    """API endpoint"""
    return jsonify(load_alerts())

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING INTELLIGENCE DASHBOARD")
    print("="*60)
    print("\n📊 Dashboard URL: http://localhost:5000")
    print("🔄 Auto-refresh: Every 5 seconds")
    print("\n✅ Dashboard ready!")
    print("\nIn another terminal, run the detector:")
    print("   python src/streaming/real_time_pipeline_v3.py")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print("\n⚠️  Port 5000 is already in use!")
            print("Trying port 8080 instead...\n")
            app.run(host='0.0.0.0', port=8080, debug=False)
        else:
            raise
