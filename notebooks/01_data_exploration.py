"""
Data Exploration Notebook
Analyzes the generated dark web intelligence dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Load data
data_dir = 'data/processed'

logger.info("Loading datasets...")
transactions_df = pd.read_csv(f'{data_dir}/transactions.csv')
threat_actors_df = pd.read_csv(f'{data_dir}/threat_actors.csv')
marketplace_df = pd.read_csv(f'{data_dir}/marketplace_listings.csv')
network_nodes_df = pd.read_csv(f'{data_dir}/network_nodes.csv')
network_edges_df = pd.read_csv(f'{data_dir}/network_edges.csv')

print("="*70)
print("DARK WEB INTELLIGENCE PLATFORM - DATA EXPLORATION REPORT")
print("="*70)

# 1. TRANSACTION ANALYSIS
print("\n" + "="*70)
print("1. CRYPTOCURRENCY TRANSACTION ANALYSIS")
print("="*70)

print(f"\nTotal Transactions: {len(transactions_df)}")
print(f"Date Range: {transactions_df['timestamp'].min()} to {transactions_df['timestamp'].max()}")
print(f"Total Volume (BTC): {transactions_df['amount_btc'].sum():.2f}")
print(f"Total Volume (USD): ${transactions_df['amount_usd'].sum():,.2f}")
print(f"Flagged Transactions: {transactions_df['is_flagged'].sum()} ({transactions_df['is_flagged'].mean()*100:.1f}%)")

print("\nTransaction Types Distribution:")
print(transactions_df['transaction_type'].value_counts())

print("\nBlockchain Distribution:")
print(transactions_df['blockchain'].value_counts())

print("\nTransaction Amount Statistics (BTC):")
print(transactions_df['amount_btc'].describe())

# 2. THREAT ACTOR ANALYSIS
print("\n" + "="*70)
print("2. THREAT ACTOR ANALYSIS")
print("="*70)

print(f"\nTotal Threat Actors: {len(threat_actors_df)}")
print(f"Average Threat Level: High/Critical: {(threat_actors_df['threat_level'].isin(['high', 'critical']).sum() / len(threat_actors_df) * 100):.1f}%")

print("\nThreat Actor Types:")
print(threat_actors_df['actor_type'].value_counts())

print("\nThreat Level Distribution:")
print(threat_actors_df['threat_level'].value_counts())

print("\nTop Associated Countries:")
print(threat_actors_df['associated_country'].value_counts().head(10))

print("\nThreat Actor with Most Transactions:")
top_actor = threat_actors_df.loc[threat_actors_df['num_known_transactions'].idxmax()]
print(f"  {top_actor['actor_name']}: {top_actor['num_known_transactions']} transactions")

# 3. MARKETPLACE ANALYSIS
print("\n" + "="*70)
print("3. DARK WEB MARKETPLACE ANALYSIS")
print("="*70)

print(f"\nTotal Listings: {len(marketplace_df)}")
print(f"Active Listings: {marketplace_df['is_active'].sum()} ({marketplace_df['is_active'].mean()*100:.1f}%)")
print(f"Total Market Value (USD): ${marketplace_df['price_usd'].sum():,.2f}")

print("\nListing Categories:")
print(marketplace_df['category'].value_counts())

print("\nMarketplace Distribution:")
print(marketplace_df['marketplace'].value_counts())

print("\nAverage Rating by Category:")
print(marketplace_df.groupby('category')['rating'].mean().sort_values(ascending=False))

print("\nAverage Price by Category (USD):")
print(marketplace_df.groupby('category')['price_usd'].mean().sort_values(ascending=False))

# 4. NETWORK ANALYSIS
print("\n" + "="*70)
print("4. NETWORK ANALYSIS")
print("="*70)

print(f"\nTotal Nodes: {len(network_nodes_df)}")
print(f"Total Edges: {len(network_edges_df)}")
print(f"Average Degree: {len(network_edges_df) * 2 / len(network_nodes_df):.2f}")

print("\nNode Types Distribution:")
print(network_nodes_df['node_type'].value_counts())

print("\nNetwork Statistics:")
print(f"  Total BTC in Network: {network_nodes_df['balance_btc'].sum():.2f}")
print(f"  Average BTC per Node: {network_nodes_df['balance_btc'].mean():.2f}")
print(f"  Max BTC in Single Node: {network_nodes_df['balance_btc'].max():.2f}")
print(f"  Average Transactions per Node: {network_nodes_df['transaction_count'].mean():.2f}")

print("\nEdge Statistics:")
print(f"  Total Transaction Volume (BTC): {network_edges_df['transaction_amount_btc'].sum():.2f}")
print(f"  Average Transaction per Edge: {network_edges_df['transaction_amount_btc'].mean():.2f}")
print(f"  Total Connection Events: {network_edges_df['transaction_count'].sum()}")

# 5. RISK ASSESSMENT
print("\n" + "="*70)
print("5. RISK ASSESSMENT")
print("="*70)

high_risk_tx = transactions_df[transactions_df['is_flagged'] == 1]
print(f"\nHigh-Risk Transactions: {len(high_risk_tx)}")
print(f"High-Risk Transaction Volume (BTC): {high_risk_tx['amount_btc'].sum():.2f}")

critical_actors = threat_actors_df[threat_actors_df['threat_level'] == 'critical']
print(f"\nCritical Threat Actors: {len(critical_actors)}")
print(f"Critical Actor Transactions: {critical_actors['num_known_transactions'].sum()}")

high_risk_listings = marketplace_df[marketplace_df['risk_score'] > 0.7]
print(f"\nHigh-Risk Marketplace Listings: {len(high_risk_listings)}")
print(f"High-Risk Market Value: ${high_risk_listings['price_usd'].sum():,.2f}")

# 6. VISUALIZATION - SAVE PLOTS
print("\n" + "="*70)
print("6. GENERATING VISUALIZATIONS")
print("="*70)

output_dir = Path('output/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Transaction Type Distribution
fig, ax = plt.subplots(figsize=(10, 6))
transactions_df['transaction_type'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Distribution of Transaction Types', fontsize=14, fontweight='bold')
ax.set_xlabel('Transaction Type')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_transaction_types.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: 01_transaction_types.png")
plt.close()

# Plot 2: Threat Level Distribution
fig, ax = plt.subplots(figsize=(10, 6))
threat_actors_df['threat_level'].value_counts().plot(kind='bar', ax=ax, color=['red', 'orange', 'yellow', 'green'])
ax.set_title('Threat Actor Threat Levels', fontsize=14, fontweight='bold')
ax.set_xlabel('Threat Level')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_threat_levels.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: 02_threat_levels.png")
plt.close()

# Plot 3: Transaction Amount Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(transactions_df['amount_btc'], bins=50, color='teal', edgecolor='black', alpha=0.7)
ax.set_title('Distribution of Transaction Amounts (BTC)', fontsize=14, fontweight='bold')
ax.set_xlabel('Amount (BTC)')
ax.set_ylabel('Frequency')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'{output_dir}/03_transaction_amounts.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: 03_transaction_amounts.png")
plt.close()

# Plot 4: Marketplace Categories
fig, ax = plt.subplots(figsize=(10, 6))
marketplace_df['category'].value_counts().plot(kind='barh', ax=ax, color='coral')
ax.set_title('Marketplace Listings by Category', fontsize=14, fontweight='bold')
ax.set_xlabel('Count')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_marketplace_categories.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: 04_marketplace_categories.png")
plt.close()

# Plot 5: Risk Score Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(marketplace_df['risk_score'], bins=30, color='crimson', edgecolor='black', alpha=0.7)
ax.set_title('Distribution of Marketplace Risk Scores', fontsize=14, fontweight='bold')
ax.set_xlabel('Risk Score')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(f'{output_dir}/05_risk_scores.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: 05_risk_scores.png")
plt.close()

# Plot 6: Node Type Distribution
fig, ax = plt.subplots(figsize=(10, 6))
network_nodes_df['node_type'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc'])
ax.set_title('Network Node Types', fontsize=14, fontweight='bold')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(f'{output_dir}/06_node_types.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: 06_node_types.png")
plt.close()

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)
print(f"\nAll visualizations saved to: {output_dir}")
print("\nNext Steps:")
print("  1. Review the generated visualizations")
print("  2. Build ML models for threat classification")
print("  3. Create network analysis reports")
print("  4. Set up automated monitoring")