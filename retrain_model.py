"""
Retrain the threat classification model with proper encoding
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

print("\n" + "="*60)
print("RETRAINING THREAT CLASSIFICATION MODEL")
print("="*60 + "\n")

# Load the training data
print("Loading training data...")
df = pd.read_csv('data/processed/transactions.csv')
print(f"✓ Loaded {len(df)} transactions")

# Display columns and data types
print(f"\nAvailable columns:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print(f"\nFirst few rows:")
print(df.head())

# Prepare features
feature_columns = ['amount_btc', 'amount_usd', 'transaction_type', 'blockchain', 'confidence_score']

# Check what columns exist
existing_cols = df.columns.tolist()

# Create/encode required features
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# 1. amount_btc and amount_usd
if 'amount_btc' not in df.columns:
    if 'amount' in df.columns:
        print("\n✓ Creating amount_btc and amount_usd from 'amount'...")
        # Assume amount is in USD
        btc_price = 60000  # Approximate BTC price
        df['amount_usd'] = df['amount']
        df['amount_btc'] = df['amount'] / btc_price
    else:
        print("\n⚠️  No amount column found, using default values")
        df['amount_usd'] = 1000
        df['amount_btc'] = 1000 / 60000

# 2. transaction_type (encode if it's text)
if 'transaction_type' not in df.columns:
    if 'type' in df.columns:
        print(f"\n✓ Encoding 'type' column as transaction_type...")
        print(f"  Unique values: {df['type'].unique()}")
        le_type = LabelEncoder()
        df['transaction_type'] = le_type.fit_transform(df['type'].astype(str))
        print(f"  Encoded as: {dict(enumerate(le_type.classes_))}")
    else:
        print("\n⚠️  No type column, using default")
        df['transaction_type'] = 0
elif df['transaction_type'].dtype == 'object':
    # transaction_type exists but is text, encode it
    print(f"\n✓ Encoding transaction_type (currently text)...")
    print(f"  Unique values: {df['transaction_type'].unique()}")
    le_type = LabelEncoder()
    df['transaction_type'] = le_type.fit_transform(df['transaction_type'].astype(str))
    print(f"  Encoded as: {dict(enumerate(le_type.classes_))}")

# 3. blockchain (encode if it's text)
if 'blockchain' not in df.columns:
    print("\n✓ Creating blockchain column...")
    df['blockchain'] = 0  # Default to 0
elif df['blockchain'].dtype == 'object':
    print(f"\n✓ Encoding blockchain (currently text)...")
    print(f"  Unique values: {df['blockchain'].unique()}")
    le_blockchain = LabelEncoder()
    df['blockchain'] = le_blockchain.fit_transform(df['blockchain'].astype(str))
    print(f"  Encoded as: {dict(enumerate(le_blockchain.classes_))}")

# 4. confidence_score
if 'confidence_score' not in df.columns:
    if 'risk_score' in df.columns:
        print("\n✓ Creating confidence_score from risk_score...")
        df['confidence_score'] = 1 - df['risk_score']
    elif 'confidence' in df.columns:
        df['confidence_score'] = df['confidence']
    else:
        print("\n✓ Creating default confidence_score...")
        df['confidence_score'] = np.random.uniform(0.5, 0.95, len(df))

# Verify all features are numeric
print("\n" + "="*60)
print("FEATURE VERIFICATION")
print("="*60)
for col in feature_columns:
    if col in df.columns:
        print(f"✓ {col}: {df[col].dtype} - Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
    else:
        print(f"✗ {col}: MISSING")

# Extract features
X = df[feature_columns].copy()

# Ensure all features are numeric
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"\n⚠️  {col} is still text, encoding...")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

print(f"\n✓ Features prepared: {feature_columns}")
print(f"  Shape: {X.shape}")
print(f"\nFeature statistics:")
print(X.describe())

# Prepare target variable
print("\n" + "="*60)
print("TARGET VARIABLE")
print("="*60)

if 'threat_level' in df.columns:
    print("✓ Using 'threat_level' column")
    if df['threat_level'].dtype == 'object':
        # threat_level is categorical (HIGH, LOW, etc.)
        y = df['threat_level'].isin(['HIGH', 'CRITICAL', 'MEDIUM']).astype(int)
    else:
        # threat_level is numeric
        y = (df['threat_level'] > df['threat_level'].median()).astype(int)
elif 'risk_score' in df.columns:
    print("✓ Using 'risk_score' column")
    y = (df['risk_score'] > 0.5).astype(int)
elif 'label' in df.columns:
    print("✓ Using 'label' column")
    if df['label'].dtype == 'object':
        le_label = LabelEncoder()
        y = le_label.fit_transform(df['label'])
    else:
        y = df['label']
else:
    print("✓ Creating synthetic labels based on amount (high amount = threat)")
    y = (df['amount_usd'] > df['amount_usd'].quantile(0.7)).astype(int)

print(f"\nTarget distribution:")
print(f"  Safe (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"  Threat (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Data split:")
print(f"  Training set: {len(X_train)} samples")
print(f"  Test set: {len(X_test)} samples")

# Scale features
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled")

# Train model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("✓ Model trained")

# Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print("MODEL PERFORMANCE")
print("="*60)
print(f"\nAccuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Threat']))

# Feature importance
print("\nFeature Importance:")
importances = sorted(zip(feature_columns, model.feature_importances_), 
                     key=lambda x: x[1], reverse=True)
for feat, imp in importances:
    print(f"  {feat:20s}: {imp:.3f} {'█' * int(imp * 50)}")

# Save models
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

model_dir = Path('output/models')
model_dir.mkdir(parents=True, exist_ok=True)

# Backup old models
old_model = model_dir / 'threat_classifier.pkl'
old_scaler = model_dir / 'scaler.pkl'

if old_model.exists():
    old_model.rename(model_dir / 'threat_classifier.pkl.backup')
    print("✓ Backed up old model")

if old_scaler.exists():
    old_scaler.rename(model_dir / 'scaler.pkl.backup')
    print("✓ Backed up old scaler")

# Save new models
with open(model_dir / 'threat_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f"✓ Model saved to: {model_dir / 'threat_classifier.pkl'}")

with open(model_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved to: {model_dir / 'scaler.pkl'}")

# Test the saved models
print("\nTesting saved models...")
with open(model_dir / 'threat_classifier.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
with open(model_dir / 'scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

test_pred = loaded_model.predict(loaded_scaler.transform(X_test))
test_accuracy = accuracy_score(y_test, test_pred)
print(f"✓ Loaded model accuracy: {test_accuracy:.2%}")

# Show some example predictions
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

sample_size = 5
sample_X = X_test.iloc[:sample_size]
sample_scaled = loaded_scaler.transform(sample_X)
sample_pred = loaded_model.predict(sample_scaled)
sample_proba = loaded_model.predict_proba(sample_scaled)

for i in range(sample_size):
    print(f"\nSample {i+1}:")
    print(f"  Features: {sample_X.iloc[i].to_dict()}")
    print(f"  Prediction: {'THREAT' if sample_pred[i] == 1 else 'SAFE'}")
    print(f"  Probability: {sample_proba[i][1]:.1%} threat")

print("\n" + "="*60)
print("✅ MODEL RETRAINING COMPLETE")
print("="*60)
print("\nYou can now restart the detector:")
print("  python src/streaming/real_time_pipeline_v3.py")
print("\nThe dashboard should continue running without errors!")
print()

