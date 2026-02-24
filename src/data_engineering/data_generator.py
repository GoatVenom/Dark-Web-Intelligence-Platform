"""
Data Generator Module
Generates synthetic dark web intelligence datasets for testing and development
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class DarkWebDataGenerator:
    """
    Generates realistic synthetic dark web intelligence data
    including cryptocurrency transactions, threat actors, and network analysis
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize data generator
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
        self.logger = logger
    
    def generate_transactions(self, n_records: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic cryptocurrency transaction data
        
        Args:
            n_records: Number of transactions to generate
            
        Returns:
            DataFrame with transaction data
        """
        self.logger.info(f"Generating {n_records} cryptocurrency transactions...")
        
        # Generate transaction data
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=int(x)) for x in np.random.rand(n_records) * 365]
        
        data = {
            'transaction_id': [f'TX_{i:06d}' for i in range(n_records)],
            'timestamp': dates,
            'from_address': [f'0x{"".join(np.random.choice(list("0123456789abcdef"), 40))}' for _ in range(n_records)],
	'to_address': [f'0x{"".join(np.random.choice(list("0123456789abcdef"), 40))}' for _ in range(n_records)],
            'amount_btc': np.random.exponential(scale=0.5, size=n_records),
            'amount_usd': np.random.exponential(scale=5000, size=n_records),
            'transaction_type': np.random.choice(['transfer', 'mixing', 'exchange', 'payment'], n_records),
            'blockchain': np.random.choice(['bitcoin', 'ethereum', 'monero'], n_records),
            'is_flagged': np.random.choice([0, 1], n_records, p=[0.85, 0.15]),
            'confidence_score': np.random.uniform(0, 1, n_records)
        }
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated {len(df)} transactions")
        
        return df
    
    def generate_threat_actors(self, n_actors: int = 100) -> pd.DataFrame:
        """
        Generate synthetic threat actor profiles
        
        Args:
            n_actors: Number of threat actors to generate
            
        Returns:
            DataFrame with threat actor data
        """
        self.logger.info(f"Generating {n_actors} threat actor profiles...")
        
        actor_types = ['cybercriminal', 'ransomware_group', 'money_launderer', 'drug_trafficker', 'hacker_collective']
        countries = ['Russia', 'China', 'North Korea', 'Iran', 'Unknown', 'Brazil', 'Romania', 'Ukraine']
        
        data = {
            'actor_id': [f'ACTOR_{i:05d}' for i in range(n_actors)],
            'actor_name': [f'Actor_{np.random.randint(1000, 9999)}' for _ in range(n_actors)],
            'actor_type': np.random.choice(actor_types, n_actors),
            'associated_country': np.random.choice(countries, n_actors),
            'threat_level': np.random.choice(['low', 'medium', 'high', 'critical'], n_actors, p=[0.3, 0.3, 0.25, 0.15]),
            'first_seen': [datetime(2023, 1, 1) + timedelta(days=int(x)) for x in np.random.rand(n_actors) * 365],
            'last_seen': [datetime(2024, 1, 1) + timedelta(days=int(x)) for x in np.random.rand(n_actors) * 365],
            'num_known_transactions': np.random.randint(1, 500, n_actors),
            'num_known_aliases': np.random.randint(0, 10, n_actors),
            'confidence_score': np.random.uniform(0.3, 1, n_actors)
        }
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated {len(df)} threat actor profiles")
        
        return df
    
    def generate_marketplace_listings(self, n_listings: int = 500) -> pd.DataFrame:
        """
        Generate synthetic dark web marketplace listings
        
        Args:
            n_listings: Number of listings to generate
            
        Returns:
            DataFrame with marketplace data
        """
        self.logger.info(f"Generating {n_listings} marketplace listings...")
        
        categories = ['drugs', 'weapons', 'stolen_data', 'hacking_tools', 'forged_documents', 'other']
        marketplaces = ['Market_A', 'Market_B', 'Market_C', 'Market_D', 'Market_E']
        
        data = {
            'listing_id': [f'LIST_{i:06d}' for i in range(n_listings)],
            'marketplace': np.random.choice(marketplaces, n_listings),
            'seller_id': [f'SELLER_{np.random.randint(1000, 9999)}' for _ in range(n_listings)],
            'category': np.random.choice(categories, n_listings),
            'price_usd': np.random.exponential(scale=500, size=n_listings),
            'quantity': np.random.randint(1, 100, n_listings),
            'reviews_count': np.random.randint(0, 500, n_listings),
            'rating': np.random.uniform(0, 5, n_listings),
            'posted_date': [datetime(2024, 1, 1) + timedelta(days=int(x)) for x in np.random.rand(n_listings) * 365],
            'is_active': np.random.choice([0, 1], n_listings, p=[0.3, 0.7]),
            'risk_score': np.random.uniform(0, 1, n_listings)
        }
        
        df = pd.DataFrame(data)
        self.logger.info(f"Generated {len(df)} marketplace listings")
        
        return df
    
    def generate_network_graph_data(self, n_nodes: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic network graph data (nodes and edges)
        
        Args:
            n_nodes: Number of network nodes
            
        Returns:
            Tuple of (nodes_df, edges_df)
        """
        self.logger.info(f"Generating network graph with {n_nodes} nodes...")
        
        # Generate nodes
        node_types = ['wallet', 'exchange', 'mixing_service', 'marketplace', 'unknown']
        nodes_data = {
            'node_id': [f'NODE_{i:05d}' for i in range(n_nodes)],
            'node_type': np.random.choice(node_types, n_nodes),
            'balance_btc': np.random.exponential(scale=1, size=n_nodes),
            'transaction_count': np.random.randint(0, 1000, n_nodes),
            'first_transaction': [datetime(2023, 1, 1) + timedelta(days=int(x)) for x in np.random.rand(n_nodes) * 365],
            'risk_score': np.random.uniform(0, 1, n_nodes)
        }
        nodes_df = pd.DataFrame(nodes_data)
        
        # Generate edges (connections between nodes)
        n_edges = int(n_nodes * 1.5)  # 1.5 edges per node on average
        edges_data = {
            'source_node': np.random.choice(nodes_df['node_id'], n_edges),
            'target_node': np.random.choice(nodes_df['node_id'], n_edges),
            'transaction_amount_btc': np.random.exponential(scale=0.5, size=n_edges),
            'transaction_count': np.random.randint(1, 50, n_edges),
            'first_connection': [datetime(2023, 1, 1) + timedelta(days=int(x)) for x in np.random.rand(n_edges) * 365],
            'last_connection': [datetime(2024, 1, 1) + timedelta(days=int(x)) for x in np.random.rand(n_edges) * 365],
        }
        edges_df = pd.DataFrame(edges_data)
        
        self.logger.info(f"Generated {len(nodes_df)} nodes and {len(edges_df)} edges")
        
        return nodes_df, edges_df
    
    def generate_full_dataset(self) -> dict:
        """
        Generate complete dataset with all components
        
        Returns:
            Dictionary containing all generated dataframes
        """
        self.logger.info("Generating complete dark web intelligence dataset...")
        
        dataset = {
            'transactions': self.generate_transactions(n_records=1000),
            'threat_actors': self.generate_threat_actors(n_actors=100),
            'marketplace_listings': self.generate_marketplace_listings(n_listings=500),
        }
        
        nodes, edges = self.generate_network_graph_data(n_nodes=200)
        dataset['network_nodes'] = nodes
        dataset['network_edges'] = edges
        
        self.logger.info("Dataset generation complete!")
        
        return dataset


def main():
    """Main function for testing data generation"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate data
    generator = DarkWebDataGenerator()
    dataset = generator.generate_full_dataset()
    
    # Save to CSV files
    output_dir = 'data/processed'
    
    for name, df in dataset.items():
        filepath = f'{output_dir}/{name}.csv'
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {name} to {filepath}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    
    for name, df in dataset.items():
        print(f"\n{name.upper()}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")


if __name__ == '__main__':
    main()