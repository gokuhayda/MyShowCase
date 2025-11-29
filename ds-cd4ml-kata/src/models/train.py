"""
Model Training Pipeline with MLflow Tracking
"""
import pandas as pd
import numpy as np
import pickle
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import mlflow
import mlflow.sklearn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_params(path: str = "params.yaml") -> dict:
    """Load parameters from YAML."""
    logger.info(f"📖 Loading parameters from {path}")
    with open(path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_data(path: str) -> pd.DataFrame:
    """Load processed data."""
    logger.info(f"📂 Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"   Shape: {df.shape}")
    return df

def prepare_data(df: pd.DataFrame, params: dict):
    """Split features and target."""
    logger.info("🔧 Preparing train/test split...")
    
    # Separate features and target
    X = df.drop('quality_binary', axis=1)
    y = df['quality_binary']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state'],
        stratify=y if params['data']['stratify'] else None
    )
    
    logger.info(f"   Train: {X_train.shape}")
    logger.info(f"   Test:  {X_test.shape}")
    logger.info(f"   Train target dist: {y_train.value_counts(normalize=True).to_dict()}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, params: dict):
    """Train Random Forest model."""
    logger.info("🎯 Training model...")
    
    model_params = params['model'].copy()
    algorithm = model_params.pop('algorithm')
    
    logger.info(f"   Algorithm: {algorithm}")
    logger.info(f"   Params: {model_params}")
    
    # Initialize model
    model = RandomForestClassifier(**model_params)
    
    # Train
    model.fit(X_train, y_train)
    
    logger.info("✅ Model trained successfully")
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test, params: dict) -> dict:
    """Evaluate model on train and test sets."""
    logger.info("📊 Evaluating model...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities for AUC
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        # Train metrics
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        
        # Test metrics
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
        
        # Overfitting check
        'accuracy_gap': accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred),
    }
    
    # Cross-validation
    logger.info("🔄 Running cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=params['cv']['n_splits'],
        scoring='accuracy'
    )
    metrics['cv_accuracy_mean'] = cv_scores.mean()
    metrics['cv_accuracy_std'] = cv_scores.std()
    
    # Log metrics
    logger.info("\n📈 Metrics:")
    logger.info(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
    logger.info(f"   Test Accuracy:  {metrics['test_accuracy']:.4f}")
    logger.info(f"   Test Precision: {metrics['test_precision']:.4f}")
    logger.info(f"   Test Recall:    {metrics['test_recall']:.4f}")
    logger.info(f"   Test F1:        {metrics['test_f1']:.4f}")
    logger.info(f"   Test AUC:       {metrics['test_auc']:.4f}")
    logger.info(f"   CV Accuracy:    {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    logger.info(f"   Overfitting gap: {metrics['accuracy_gap']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"\n🎯 Confusion Matrix:")
    logger.info(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
    logger.info(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    return metrics

def validate_quality_gates(metrics: dict, params: dict) -> bool:
    """Check if model meets quality thresholds."""
    logger.info("\n🚦 Validating quality gates...")
    
    thresholds = params['metrics']
    passed = True
    
    checks = {
        'Accuracy': (metrics['test_accuracy'], thresholds['min_accuracy']),
        'Precision': (metrics['test_precision'], thresholds['min_precision']),
        'Recall': (metrics['test_recall'], thresholds['min_recall']),
        'F1': (metrics['test_f1'], thresholds['min_f1']),
    }
    
    for name, (value, threshold) in checks.items():
        status = "✅" if value >= threshold else "❌"
        logger.info(f"   {status} {name}: {value:.4f} (min: {threshold:.4f})")
        if value < threshold:
            passed = False
    
    # Overfitting check
    gap = metrics['accuracy_gap']
    max_gap = thresholds['max_train_test_gap']
    status = "✅" if gap <= max_gap else "❌"
    logger.info(f"   {status} Overfitting gap: {gap:.4f} (max: {max_gap:.4f})")
    if gap > max_gap:
        passed = False
    
    return passed

def save_artifacts(model, metrics: dict):
    """Save model and metrics."""
    logger.info("\n💾 Saving artifacts...")
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Save model
    model_path = "models/model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"   Model saved: {model_path}")
    
    # Save metrics
    metrics_path = "models/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"   Metrics saved: {metrics_path}")
    
    return model_path, metrics_path

def main():
    """Run training pipeline with MLflow tracking."""
    logger.info("=" * 70)
    logger.info("🚀 Starting Training Pipeline")
    logger.info("=" * 70)
    
    # Set MLflow experiment
    mlflow.set_experiment("wine-quality-classification")
    
    with mlflow.start_run():
        # Load parameters
        params = load_params()
        
        # Load data
        df = load_data("data/processed/wine_features.csv")
        
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(df, params)
        
        # Train model
        model = train_model(X_train, y_train, params)
        
        # Evaluate
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, params)
        
        # Validate quality gates
        passed = validate_quality_gates(metrics, params)
        
        # Log to MLflow
        logger.info("\n📝 Logging to MLflow...")
        mlflow.log_params(params['model'])
        mlflow.log_params({f"data_{k}": v for k, v in params['data'].items()})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        mlflow.set_tag("quality_gates_passed", passed)
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        
        # Save artifacts
        model_path, metrics_path = save_artifacts(model, metrics)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(metrics_path)
        
        # Final status
        logger.info("\n" + "=" * 70)
        if passed:
            logger.info("✅ Training completed successfully! Quality gates PASSED.")
        else:
            logger.info("⚠️  Training completed but quality gates FAILED.")
            logger.info("   Model may need tuning or more data.")
        logger.info("=" * 70)
        
        return passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)