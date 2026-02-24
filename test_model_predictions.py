"""
Test what the model is actually predicting
"""
import pickle
import pandas as pd
import random

# Load models
with open('output/models/threat_classifier.pkl', 'rb') as f:
    model = pickle.load(f)
with open('output/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("="*60)
print("MODEL ANALYSIS")
print("="*60)

# Check what classes the model predicts
print(f"\nModel type: {type(model).__name__}")
if hasattr(model, 'classes_'):
    print(f"Threat levels the model can predict: {model.classes_}")

# Check feature names
if hasattr(scaler, 'feature_names_in_'):
    print(f"\nFeatures expected:")
    for i, name in enumerate(scaler.feature_names_in_):
        print(f"  {i+1}. {name}")
else:
    print(f"\nNumber of features: {scaler.n_features_in_}")

# Generate some test transactions and see what it predicts
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

transaction_types = ['PURCHASE', 'TRANSFER', 'WITHDRAWAL', 'MARKETPLACE']

for i in range(10):
    # Create test transaction
    features = {
        'amount': random.uniform(100, 50000),
        'risk_score': random.uniform(0, 1),
        'sender_length': random.randint(5, 20),
        'transaction_type_encoded': random.randint(0, 3),
        'fees': random.uniform(0.5, 5.0),
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # If we know feature names, reorder
    if hasattr(scaler, 'feature_names_in_'):
        df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    
    # Scale and predict
    features_scaled = scaler.transform(df)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = max(probabilities)
    
    print(f"\nTransaction {i+1}:")
    print(f"  Amount: ${features['amount']:,.2f}")
    print(f"  Risk Score: {features['risk_score']:.3f}")
    print(f"  Prediction: {prediction}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  All probabilities: {dict(zip(model.classes_, probabilities))}")

# Check alert threshold
print("\n" + "="*60)
print("ALERT CRITERIA ANALYSIS")
print("="*60)
print("\nCurrent alert rules:")
print("  - Threat level must be: HIGH or CRITICAL")
print("  - Confidence must be: > 80%")

# Count how many would trigger alerts
alert_count = 0
test_samples = 100

for i in range(test_samples):
    features = {
        'amount': random.uniform(100, 50000),
        'risk_score': random.uniform(0, 1),
        'sender_length': random.randint(5, 20),
        'transaction_type_encoded': random.randint(0, 3),
        'fees': random.uniform(0.5, 5.0),
    }
    
    df = pd.DataFrame([features])
    if hasattr(scaler, 'feature_names_in_'):
        df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    
    features_scaled = scaler.transform(df)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = max(probabilities)
    
    if prediction in ['HIGH', 'CRITICAL'] and confidence > 0.8:
        alert_count += 1

print(f"\nOut of {test_samples} random transactions:")
print(f"  {alert_count} would trigger alerts ({alert_count/test_samples*100:.1f}%)")

if alert_count == 0:
    print("\n⚠️  NO ALERTS WOULD BE TRIGGERED!")
    print("\nRecommendations:")
    print("  1. Lower confidence threshold (try 0.5 instead of 0.8)")
    print("  2. Add more threat levels to alert on")
    print("  3. Add additional alert conditions")

print("\n" + "="*60)
