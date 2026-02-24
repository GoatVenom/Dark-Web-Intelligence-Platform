"""
Threat Classification Model
Builds and trains ML models to classify cryptocurrency transactions as high/low risk
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")

# Load data
logger.info("Loading transaction data...")
transactions_df = pd.read_csv('data/processed/transactions.csv')
threat_actors_df = pd.read_csv('data/processed/threat_actors.csv')

print("="*70)
print("THREAT CLASSIFICATION MODEL - TRAINING & EVALUATION")
print("="*70)

# Data Preparation
logger.info("Preparing features...")

# Create feature matrix
X = transactions_df.copy()

# Encode categorical variables
le_type = LabelEncoder()
le_blockchain = LabelEncoder()

X['transaction_type'] = le_type.fit_transform(X['transaction_type'])
X['blockchain'] = le_blockchain.fit_transform(X['blockchain'])

# Drop non-numeric and irrelevant columns
X = X.drop(['transaction_id', 'timestamp', 'from_address', 'to_address'], axis=1)

# Target variable
y = X['is_flagged']
X = X.drop('is_flagged', axis=1)

print(f"\nFeature Matrix Shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"\nTarget Distribution:")
print(f"  Normal (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"  Flagged (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

# Train-Test Split
logger.info("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain Set: {X_train.shape}")
print(f"Test Set: {X_test.shape}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Multiple Models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\n" + "="*70)
print("MODEL TRAINING & EVALUATION")
print("="*70)

for model_name, model in models.items():
    logger.info(f"Training {model_name}...")
    
    # Train on scaled data
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[model_name] = {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cm': confusion_matrix(y_test, y_pred)
    }
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

# Feature Importance (Random Forest)
logger.info("Extracting feature importance...")
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*70)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*70)
print(feature_importance.to_string(index=False))

# Visualizations
output_dir = Path('output/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Model Comparison
fig, ax = plt.subplots(figsize=(12, 6))
metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
})

metrics_df.set_index('Model').plot(kind='bar', ax=ax)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_xlabel('')
ax.legend(loc='lower right')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{output_dir}/07_model_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: 07_model_comparison.png")
plt.close()

# Plot 2: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (model_name, result) in enumerate(results.items()):
    cm = result['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{model_name}\nConfusion Matrix')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(f'{output_dir}/08_confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: 08_confusion_matrices.png")
plt.close()

# Plot 3: ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
for model_name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
    ax.plot(fpr, tpr, label=f"{model_name} (AUC: {result['roc_auc']:.3f})")

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/09_roc_curves.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: 09_roc_curves.png")
plt.close()

# Plot 4: Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
top_features = feature_importance.head(10)
ax.barh(top_features['feature'], top_features['importance'], color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Top 10 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/10_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: 10_feature_importance.png")
plt.close()

# Save Best Model
logger.info("Saving best model...")
best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

model_path = Path('output/models')
model_path.mkdir(parents=True, exist_ok=True)

with open(f'{model_path}/threat_classifier.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open(f'{model_path}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n✅ Saved best model: {best_model_name}")

print("\n" + "="*70)
print("✅ MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\nBest Model: {best_model_name} (F1-Score: {results[best_model_name]['f1']:.4f})")
print(f"Models saved to: output/models/")
print(f"Visualizations saved to: output/visualizations/")