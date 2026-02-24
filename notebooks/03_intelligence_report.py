"""
Intelligence Report Generator
Creates comprehensive threat intelligence reports
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load all datasets
logger.info("Loading all datasets...")
transactions_df = pd.read_csv('data/processed/transactions.csv')
threat_actors_df = pd.read_csv('data/processed/threat_actors.csv')
marketplace_df = pd.read_csv('data/processed/marketplace_listings.csv')
network_nodes_df = pd.read_csv('data/processed/network_nodes.csv')
network_edges_df = pd.read_csv('data/processed/network_edges.csv')

# Create output directory
output_dir = Path('output/reports')
output_dir.mkdir(parents=True, exist_ok=True)

# Generate Report
report_path = output_dir / 'intelligence_report.txt'

with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("DARK WEB CRIMINAL NETWORK INTELLIGENCE PLATFORM\n")
    f.write("COMPREHENSIVE THREAT ASSESSMENT REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Report Period: 2024-01-01 to 2024-12-31\n\n")
    
    # EXECUTIVE SUMMARY
    f.write("="*80 + "\n")
    f.write("EXECUTIVE SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write("This report provides a comprehensive analysis of dark web activities.\n\n")
    
    total_volume_btc = transactions_df['amount_btc'].sum()
    total_volume_usd = transactions_df['amount_usd'].sum()
    flagged_pct = transactions_df['is_flagged'].mean() * 100
    
    f.write("KEY FINDINGS:\n")
    f.write(f"  Total Cryptocurrency Transactions: {len(transactions_df):,}\n")
    f.write(f"  Total Transaction Volume: {total_volume_btc:.2f} BTC\n")
    f.write(f"  Total USD Value: ${total_volume_usd:,.2f}\n")
    f.write(f"  Flagged High-Risk Transactions: {transactions_df['is_flagged'].sum()}\n")
    f.write(f"  Risk Percentage: {flagged_pct:.1f}%\n")
    f.write(f"  Identified Threat Actors: {len(threat_actors_df)}\n")
    f.write(f"  Critical Threat Actors: {len(threat_actors_df[threat_actors_df['threat_level'] == 'critical'])}\n")
    f.write(f"  Marketplaces Monitored: {marketplace_df['marketplace'].nunique()}\n")
    f.write(f"  Active Listings: {marketplace_df['is_active'].sum()}\n")
    f.write(f"  Network Nodes: {len(network_nodes_df)}\n")
    f.write(f"  Network Connections: {len(network_edges_df)}\n\n")
    
    # THREAT LANDSCAPE
    f.write("="*80 + "\n")
    f.write("1. THREAT LANDSCAPE\n")
    f.write("="*80 + "\n\n")
    
    f.write("1.1 THREAT ACTOR OVERVIEW\n")
    f.write("-" * 80 + "\n\n")
    
    threat_summary = threat_actors_df['threat_level'].value_counts()
    f.write("Threat Level Distribution:\n")
    for level, count in threat_summary.items():
        pct = count / len(threat_actors_df) * 100
        f.write(f"  {level.upper()}: {count} actors ({pct:.1f}%)\n")
    
    f.write("\nThreat Actor Types:\n")
    actor_types = threat_actors_df['actor_type'].value_counts()
    for atype, count in actor_types.items():
        f.write(f"  {atype}: {count} actors\n")
    
    f.write("\nGeographic Distribution:\n")
    countries = threat_actors_df['associated_country'].value_counts().head(10)
    for country, count in countries.items():
        f.write(f"  {country}: {count} actors\n")
    
    f.write("\n1.2 CRITICAL THREAT ACTORS\n")
    f.write("-" * 80 + "\n\n")
    
    critical_actors = threat_actors_df[threat_actors_df['threat_level'] == 'critical'].sort_values(
        'num_known_transactions', ascending=False
    )
    
    for idx, actor in critical_actors.head(5).iterrows():
        f.write(f"Actor ID: {actor['actor_id']}\n")
        f.write(f"  Name: {actor['actor_name']}\n")
        f.write(f"  Type: {actor['actor_type']}\n")
        f.write(f"  Country: {actor['associated_country']}\n")
        f.write(f"  Transactions: {actor['num_known_transactions']}\n")
        f.write(f"  Confidence: {actor['confidence_score']:.2%}\n\n")
    
    # CRYPTOCURRENCY ANALYSIS
    f.write("="*80 + "\n")
    f.write("2. CRYPTOCURRENCY TRANSACTION ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write("2.1 TRANSACTION OVERVIEW\n")
    f.write("-" * 80 + "\n\n")
    
    f.write("Transaction Types:\n")
    tx_types = transactions_df['transaction_type'].value_counts()
    for ttype, count in tx_types.items():
        pct = count / len(transactions_df) * 100
        f.write(f"  {ttype}: {count} ({pct:.1f}%)\n")
    
    f.write("\nBlockchain Distribution:\n")
    blockchains = transactions_df['blockchain'].value_counts()
    for bc, count in blockchains.items():
        pct = count / len(transactions_df) * 100
        f.write(f"  {bc}: {count} ({pct:.1f}%)\n")
    
    f.write("\n2.2 TRANSACTION VOLUME ANALYSIS\n")
    f.write("-" * 80 + "\n\n")
    
    f.write(f"Total Volume (BTC): {total_volume_btc:.2f}\n")
    f.write(f"Total Volume (USD): ${total_volume_usd:,.2f}\n")
    f.write(f"Average Transaction: {transactions_df['amount_btc'].mean():.4f} BTC\n")
    f.write(f"Median Transaction: {transactions_df['amount_btc'].median():.4f} BTC\n")
    f.write(f"Max Transaction: {transactions_df['amount_btc'].max():.4f} BTC\n\n")
    
    f.write("2.3 HIGH-RISK TRANSACTIONS\n")
    f.write("-" * 80 + "\n\n")
    
    high_risk_tx = transactions_df[transactions_df['is_flagged'] == 1]
    f.write(f"Total High-Risk Transactions: {len(high_risk_tx)}\n")
    f.write(f"High-Risk Volume (BTC): {high_risk_tx['amount_btc'].sum():.2f}\n")
    f.write(f"Percentage of Total: {(high_risk_tx['amount_btc'].sum() / total_volume_btc * 100):.1f}%\n\n")
    
    # MARKETPLACE ANALYSIS
    f.write("="*80 + "\n")
    f.write("3. DARK WEB MARKETPLACE ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write("3.1 MARKETPLACE OVERVIEW\n")
    f.write("-" * 80 + "\n\n")
    
    f.write("Marketplace Distribution:\n")
    markets = marketplace_df['marketplace'].value_counts()
    for market, count in markets.items():
        f.write(f"  {market}: {count} listings\n")
    
    f.write("\nListing Categories:\n")
    categories = marketplace_df['category'].value_counts()
    for cat, count in categories.items():
        f.write(f"  {cat}: {count} listings\n")
    
    f.write("\n3.2 MARKETPLACE METRICS\n")
    f.write("-" * 80 + "\n\n")
    
    f.write(f"Total Listings: {len(marketplace_df)}\n")
    f.write(f"Active Listings: {marketplace_df['is_active'].sum()}\n")
    f.write(f"Total Market Value: ${marketplace_df['price_usd'].sum():,.2f}\n")
    f.write(f"Average Price: ${marketplace_df['price_usd'].mean():,.2f}\n")
    f.write(f"Average Rating: {marketplace_df['rating'].mean():.2f}/5.0\n\n")
    
    # NETWORK ANALYSIS
    f.write("="*80 + "\n")
    f.write("4. CRIMINAL NETWORK ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write("4.1 NETWORK TOPOLOGY\n")
    f.write("-" * 80 + "\n\n")
    
    f.write(f"Total*
