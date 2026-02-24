"""
Alert Viewer - Formatted display of recent alerts
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter

def view_alerts(num_alerts=20):
    alert_file = Path('output/alerts/real_time_alerts.jsonl')
    
    if not alert_file.exists():
        print("\n⚠️  No alerts file found!")
        print("\nMake sure the detector is running:")
        print("  python src/streaming/real_time_pipeline_v3.py\n")
        return
    
    # Load all alerts
    alerts = []
    with open(alert_file, 'r') as f:
        for line in f:
            try:
                alerts.append(json.loads(line))
            except:
                pass
    
    if not alerts:
        print("\n⚠️  Alert file is empty!")
        return
    
    # Statistics
    severity_counts = Counter(a['severity'] for a in alerts)
    
    print("\n" + "="*70)
    print(f"ALERT DASHBOARD - {len(alerts)} Total Alerts")
    print("="*70)
    
    # Summary
    print("\n📊 SUMMARY:")
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = severity_counts.get(severity, 0)
        pct = (count / len(alerts) * 100) if alerts else 0
        emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}[severity]
        bar = '█' * int(pct / 2)
        print(f"  {emoji} {severity:10s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Recent alerts
    print(f"\n🚨 RECENT ALERTS (Last {num_alerts}):")
    print("="*70)
    
    for i, alert in enumerate(alerts[-num_alerts:], 1):
        emoji = {
            'CRITICAL': '🔴',
            'HIGH': '🟠',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }.get(alert['severity'], '⚠️')
        
        # Parse time
        try:
            alert_time = datetime.fromisoformat(alert['alert_time'])
            time_str = alert_time.strftime('%Y-%m-%d %H:%M:%S')
        except:
            time_str = alert['alert_time']
        
        print(f"\n{emoji} #{len(alerts) - num_alerts + i} - {alert['severity']} THREAT")
        print(f"   Time: {time_str}")
        print(f"   Transaction: {alert['transaction_id']}")
        print(f"   Amount: ${alert['amount_usd']:,.2f} ({alert['amount_btc']:.4f} BTC)")
        print(f"   Threat Probability: {alert['threat_probability']:.1%}")
        print(f"   Blockchain: {alert['blockchain']} | Type: {alert.get('transaction_type', 'N/A')}")
    
    print("\n" + "="*70)
    print(f"\n📁 Alert file: {alert_file}")
    print(f"📊 Dashboard: http://localhost:5000")
    print()

if __name__ == '__main__':
    import sys
    
    num = 20
    if len(sys.argv) > 1:
        try:
            num = int(sys.argv[1])
        except:
            pass
    
    view_alerts(num)
