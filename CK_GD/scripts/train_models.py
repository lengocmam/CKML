"""
Training Script for Heart Disease Stacking Ensemble
====================================================
Author: Heart Disease Prediction Team
Date: 2025

This script trains 4 base models and 1 meta-learner for heart disease prediction.

Architecture:
  INPUT (22 features) 
    ↓
  LEVEL 1: 4 Base Models (SVM, Softmax, NB, DT) → 4 Probabilities
    ↓
  LEVEL 2: Meta-Learner (Logistic Regression) → Final Decision
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# =========================
# Configuration
# =========================
DATA_PATH = "data/raw/heart_disease_data.csv"  # Đặt file CSV vào đây
OUTPUT_MODELS_DIR = "assets/models"
OUTPUT_REPORTS_DIR = "assets/reports"
TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURE_COLS = [
    "BMI", "PhysicalHealth", "MentalHealth", "SleepTime",
    "Race_American Indian/Alaskan Native", "Race_Asian", 
    "Race_Black", "Race_Hispanic", "Race_Other", "Race_White",
    "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", 
    "Sex", "AgeCategory", "Diabetic", "PhysicalActivity", 
    "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"
]

# =========================
# Helper Functions
# =========================
def print_header(text, char="=", width=60):
    """Print formatted header"""
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}")

def print_step(step, total, description):
    """Print training step"""
    print(f"\n[{step}/{total}] {description}...")

def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate all evaluation metrics"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba))
    }

def save_model(model, filename):
    """Save model to file"""
    filepath = os.path.join(OUTPUT_MODELS_DIR, filename)
    joblib.dump(model, filepath)
    print(f"  ✓ Saved: {filepath}")

# =========================
# Main Training Pipeline
# =========================
def main():
    print_header("STACKING ENSEMBLE TRAINING")
    
    # Step 1: Load Data
    print_step(1, 8, "Loading data")
    
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ ERROR: Data file not found at {DATA_PATH}")
        print(f"\nPlease place your CSV file at: {DATA_PATH}")
        print("\nExpected format:")
        print("  - Column 1: HeartDisease (0 or 1)")
        print("  - Columns 2-23: 22 features in correct order")
        return
    
    df = pd.read_csv(DATA_PATH)
    print(f"  ✓ Loaded: {len(df)} samples")
    print(f"  ✓ Features: {len(df.columns) - 1}")
    
    # Verify columns
    if 'HeartDisease' not in df.columns:
        print("  ❌ ERROR: 'HeartDisease' column not found!")
        return
    
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # Check if columns match
    if list(X.columns) != FEATURE_COLS:
        print("  ⚠️ WARNING: Column names don't match expected features!")
        print("  Expected order:", FEATURE_COLS[:5], "...")
        print("  Got:", list(X.columns)[:5], "...")
    
    print(f"  ✓ Positive cases: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"  ✓ Negative cases: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")
    
    # Step 2: Train/Test Split
    print_step(2, 8, "Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  ✓ Training set: {len(X_train)} samples")
    print(f"  ✓ Test set: {len(X_test)} samples")
    
    # Step 3: Train Base Models (LEVEL 1)
    print_step(3, 8, "Training LEVEL 1: Base Models")
    print("  → 4 models will predict independently")
    
    models_dict = {}
    
    # Model 1: SVM
    print("\n  [1/4] SVM Classifier")
    svm_model = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, 
                    C=1.0, gamma='scale', verbose=False)
    svm_model.fit(X_train, y_train)
    models_dict['svm'] = svm_model
    print("    ✓ Training complete")
    
    # Model 2: Softmax Regression
    print("\n  [2/4] Softmax Regression (Logistic)")
    softmax_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, 
                                       solver='lbfgs', verbose=0)
    softmax_model.fit(X_train, y_train)
    models_dict['softmax'] = softmax_model
    print("    ✓ Training complete")
    
    # Model 3: Naive Bayes
    print("\n  [3/4] Naive Bayes")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    models_dict['nb'] = nb_model
    print("    ✓ Training complete")
    
    # Model 4: Decision Tree
    print("\n  [4/4] Decision Tree")
    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10, 
                                      min_samples_split=10)
    dt_model.fit(X_train, y_train)
    models_dict['dt'] = dt_model
    print("    ✓ Training complete")
    
    # Step 4: Create Meta Features
    print_step(4, 8, "Creating Meta Features (Stacking)")
    print("  → Extracting probabilities from base models")
    
    meta_features_train = np.column_stack([
        svm_model.predict_proba(X_train)[:, 1],
        softmax_model.predict_proba(X_train)[:, 1],
        nb_model.predict_proba(X_train)[:, 1],
        dt_model.predict_proba(X_train)[:, 1]
    ])
    
    meta_features_test = np.column_stack([
        svm_model.predict_proba(X_test)[:, 1],
        softmax_model.predict_proba(X_test)[:, 1],
        nb_model.predict_proba(X_test)[:, 1],
        dt_model.predict_proba(X_test)[:, 1]
    ])
    
    print(f"  ✓ Meta features shape: {meta_features_train.shape}")
    print(f"  ✓ Each sample → 4 probabilities (one from each base model)")
    
    # Step 5: Train Meta-Learner (LEVEL 2)
    print_step(5, 8, "Training LEVEL 2: Meta-Learner")
    print("  → Meta-learner combines 4 base model predictions")
    
    meta_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    meta_model.fit(meta_features_train, y_train)
    models_dict['meta'] = meta_model
    
    print("  ✓ Meta-learner trained")
    print(f"  ✓ Weights: {meta_model.coef_[0]}")
    print("     (Shows trust level for each base model)")
    
    # Step 6: Evaluate Models
    print_step(6, 8, "Evaluating all models")
    
    metrics = {}
    
    # Evaluate each base model
    for name, model in [('SVM Classifier', svm_model),
                        ('SoftmaxRegression', softmax_model),
                        ('NaiveBayes', nb_model),
                        ('DecisionTreeClassifier', dt_model)]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics[name] = calculate_metrics(y_test, y_pred, y_proba)
        print(f"  ✓ {name}: Accuracy={metrics[name]['accuracy']:.4f}, AUC={metrics[name]['auc']:.4f}")
    
    # Evaluate meta-learner
    meta_pred = meta_model.predict(meta_features_test)
    meta_proba = meta_model.predict_proba(meta_features_test)[:, 1]
    metrics['Ensemble Logistic (Meta)'] = calculate_metrics(y_test, meta_pred, meta_proba)
    print(f"  ✓ Meta-Learner: Accuracy={metrics['Ensemble Logistic (Meta)']['accuracy']:.4f}, AUC={metrics['Ensemble Logistic (Meta)']['auc']:.4f}")
    
    # Step 7: Save Models
    print_step(7, 8, "Saving models")
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
    
    save_model(svm_model, 'svm_model.joblib')
    save_model(softmax_model, 'softmax_model.joblib')
    save_model(nb_model, 'nb_model.joblib')
    save_model(dt_model, 'dt_model.joblib')
    save_model(meta_model, 'meta_logistic.joblib')
    
    # Step 8: Save Metrics
    print_step(8, 8, "Saving metrics")
    os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)
    
    metrics_file = os.path.join(OUTPUT_REPORTS_DIR, 'metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Saved: {metrics_file}")
    
    # Final Summary
    print_header("✅ TRAINING COMPLETE!", "=")
    
    print("\n📊 MODEL PERFORMANCE SUMMARY:")
    print("-" * 60)
    for model_name, m in metrics.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1 Score:  {m['f1']:.4f}")
        print(f"  AUC:       {m['auc']:.4f}")
    
    print("\n" + "=" * 60)
    print("📁 OUTPUT FILES:")
    print("-" * 60)
    print(f"  ✓ {OUTPUT_MODELS_DIR}/svm_model.joblib")
    print(f"  ✓ {OUTPUT_MODELS_DIR}/softmax_model.joblib")
    print(f"  ✓ {OUTPUT_MODELS_DIR}/nb_model.joblib")
    print(f"  ✓ {OUTPUT_MODELS_DIR}/dt_model.joblib")
    print(f"  ✓ {OUTPUT_MODELS_DIR}/meta_logistic.joblib")
    print(f"  ✓ {OUTPUT_REPORTS_DIR}/metrics.json")
    print("=" * 60)
    
    print("\n🚀 NEXT STEPS:")
    print("  1. Run: streamlit run app.py")
    print("  2. Test predictions with real trained models!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
