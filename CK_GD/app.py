# app.py
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os
from pathlib import Path

# =========================
# Custom Model Classes
# =========================
class SoftmaxRegression:
    """Custom Softmax Regression implementation"""
    def __init__(self, learning_rate=0.01, epochs=1000, n_classes=2, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_classes = n_classes
        self.weights = None
        self.bias = None
        np.random.seed(random_state)

    def _initialize_parameters(self, n_features):
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot_encode(self, y):
        one_hot = np.zeros((len(y), self.n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        y_encoded = self._one_hot_encode(y)

        for i in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._softmax(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_encoded))
            db = (1 / n_samples) * np.sum(y_pred - y_encoded, axis=0, keepdims=True)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._softmax(linear_model)

    def predict(self, X):
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)

class ManualGaussianNB:
    """Custom Naive Bayes implementation"""
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]

        self.mean = np.zeros((len(self.classes), n_features))
        self.var = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i, :] = X_c.mean(axis=0)
            self.var[i, :] = X_c.var(axis=0) + self.var_smoothing
            self.priors[i] = X_c.shape[0] / float(len(X))

    def _calculate_likelihood(self, class_idx, X):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(X - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            likelihoods = np.log(self._calculate_likelihood(i, X) + 1e-15)
            posterior = prior + np.sum(likelihoods, axis=1)
            posteriors.append(posterior)
        return self.classes[np.argmax(np.array(posteriors), axis=0)]
    
    def predict_proba(self, X):
        """Add predict_proba for compatibility with stacking"""
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            likelihoods = np.log(self._calculate_likelihood(i, X) + 1e-15)
            posterior = prior + np.sum(likelihoods, axis=1)
            posteriors.append(posterior)
        
        posteriors = np.array(posteriors).T
        # Convert log posteriors to probabilities
        exp_posteriors = np.exp(posteriors - np.max(posteriors, axis=1, keepdims=True))
        proba = exp_posteriors / np.sum(exp_posteriors, axis=1, keepdims=True)
        
        # Ensure always returns 2 columns for binary classification
        if proba.shape[1] == 1:
            # If only one class, create second column with zeros
            proba = np.hstack([1 - proba, proba])
        
        return proba

class Node:
    """Node for Decision Tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier:
    """Custom Decision Tree implementation"""
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(set(y))

        if (n_labels == 1 or depth >= self.max_depth or n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(feature=None, threshold=None, left=None, right=None, value=leaf_value)

        best_feature, best_threshold = self.best_split(X, y, n_features)
        left_idx = (X[:, best_feature] <= best_threshold)
        right_idx = (X[:, best_feature] > best_threshold)

        left = self.grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self.grow_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def best_split(self, X, y, n_features):
        best_feature, best_threshold = None, None
        best_gain = -1

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thres in thresholds:
                left = X[:, feat] <= thres
                right = X[:, feat] > thres
                if (len(y[left]) == 0 or len(y[right]) == 0):
                    continue
                gain = self.gini_gain(y, y[left], y[right])
                if best_gain < gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thres
        return best_feature, best_threshold

    def gini(self, y):
        classes = np.unique(y)
        imp = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            imp -= p**2
        return imp

    def gini_gain(self, y, left, right):
        n = len(y)
        n_left = len(left)
        n_right = len(right)
        child = n_left / n * self.gini(left) + n_right / n * self.gini(right)
        return self.gini(y) - child

    def most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict_proba(self, X):
        """Add predict_proba for compatibility with stacking"""
        predictions = self.predict(X)
        n_samples = len(predictions)
        # Always return probabilities for 2 classes (binary classification)
        proba = np.zeros((n_samples, 2))
        for i, pred in enumerate(predictions):
            if int(pred) == 0:
                proba[i, 0] = 1.0
                proba[i, 1] = 0.0
            else:
                proba[i, 0] = 0.0
                proba[i, 1] = 1.0
        return proba

# =========================
# Page setup + better CSS
# =========================
st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; max-width: 1400px; }
      .stTabs [data-baseweb="tab-list"] { gap: 15px; }
      .stTabs [data-baseweb="tab"] { 
        padding: 12px 24px; 
        border-radius: 8px;
        font-weight: 600;
        border: 2px solid transparent;
        transition: all 0.3s;
      }
      .small-note { color: #666; font-size: 0.85rem; font-style: italic; margin: 0.5rem 0; }
      .section-title { 
        font-weight: 700; 
        font-size: 1.1rem; 
        margin: 1rem 0 0.6rem; 
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0066cc;
        color: #0066cc;
      }
      .card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        background: linear-gradient(135deg, #f5f7fa 0%, #fff 100%);
        margin-bottom: 1.5rem;
      }
      .input-group {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #0066cc;
      }
      .feature-label {
        font-weight: 600;
        color: #333;
        font-size: 0.95rem;
      }
      .feature-hint {
        font-size: 0.8rem;
        color: #999;
        margin-top: 0.2rem;
      }
      .pills { display: flex; gap: 8px; flex-wrap: wrap; margin: 1rem 0; }
      .pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: #e3f2fd;
        border: 1px solid #0066cc;
        font-size: 0.8rem;
        color: #0066cc;
        font-weight: 600;
      }
      .hr { border-top: 2px solid #eee; margin: 1.5rem 0; }
      .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
      }
      .stats-box h3 { margin: 0; font-size: 1.2rem; }
      .stats-box p { margin: 0.3rem 0 0; font-size: 0.85rem; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Load Models
# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    """Load 5 models cho Stacking Ensemble"""
    try:
        # Get absolute path to models directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'assets', 'models')
        
        st.info(f"🔍 Looking for models in: {models_dir}")
        
        models_dict = {}
        model_files = {
            'svm': 'svm_rbf_model.joblib',
            'softmax': 'softmax_model.joblib',
            'nb': 'naivebayes_model.joblib',
            'dt': 'decision_tree_model.joblib',
            'meta': 'stacking_model.joblib'
        }
        
        missing_models = []
        loaded_models = []
        for name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                try:
                    models_dict[name] = joblib.load(filepath)
                    loaded_models.append(f"✅ {filename}")
                except Exception as load_error:
                    missing_models.append(f"❌ {filename}: {str(load_error)}")
            else:
                missing_models.append(f"❌ {filename}: File not found")
        
        # Show loading details
        st.write("**Loading Status:**")
        for msg in loaded_models:
            st.write(msg)
        for msg in missing_models:
            st.write(msg)
        
        if missing_models:
            st.warning(f"⚠️ Failed to load {len(missing_models)}/5 models")
            return None
        
        st.success(f"✅ Successfully loaded all {len(models_dict)}/5 models!")
        return models_dict
    
    except Exception as e:
        st.error(f"❌ Fatal error loading models: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Load models
models = load_models()

# =========================
# Constants (processed schema)
# =========================
FEATURE_COLS = [
    "BMI",
    "PhysicalHealth",
    "MentalHealth",
    "SleepTime",
    "Race_American Indian/Alaskan Native",
    "Race_Asian",
    "Race_Black",
    "Race_Hispanic",
    "Race_Other",
    "Race_White",
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "Sex",
    "AgeCategory",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

RACE_OPTIONS = [
    "American Indian/Alaskan Native",
    "Asian",
    "Black",
    "Hispanic",
    "Other",
    "White",
]

MODEL_NAMES = [
    "SVM Classifier",
    "SoftmaxRegression",
    "NaiveBayes",
    "DecisionTreeClassifier",
    "Ensemble Logistic (Meta)",
]

DEFAULT_METRICS = {
    "SVM Classifier": {"accuracy": None, "precision": None, "recall": None, "f1": None, "auc": None},
    "SoftmaxRegression": {"accuracy": None, "precision": None, "recall": None, "f1": None, "auc": None},
    "NaiveBayes": {"accuracy": None, "precision": None, "recall": None, "f1": None, "auc": None},
    "DecisionTreeClassifier": {"accuracy": None, "precision": None, "recall": None, "f1": None, "auc": None},
    "Ensemble Logistic (Meta)": {"accuracy": None, "precision": None, "recall": None, "f1": None, "auc": None},
}

# 3 dòng mẫu bạn đã gửi (để demo nhanh)
SAMPLE_ROWS = {
    "Sample #1": {
        "BMI": -1.844750159,
        "PhysicalHealth": -0.046751049,
        "MentalHealth": 3.281068875,
        "SleepTime": -1.460353521,
        "Race": "White",
        "Smoking": 1,
        "AlcoholDrinking": 0,
        "Stroke": 0,
        "DiffWalking": 0,
        "Sex": 0,
        "AgeCategory": 7,
        "Diabetic": 3,
        "PhysicalActivity": 1,
        "GenHealth": 3,
        "Asthma": 1,
        "KidneyDisease": 0,
        "SkinCancer": 1,
    },
    "Sample #2": {
        "BMI": -1.256338125,
        "PhysicalHealth": -0.424069778,
        "MentalHealth": -0.490038588,
        "SleepTime": -0.067600531,
        "Race": "White",
        "Smoking": 0,
        "AlcoholDrinking": 0,
        "Stroke": 1,
        "DiffWalking": 0,
        "Sex": 0,
        "AgeCategory": 12,
        "Diabetic": 0,
        "PhysicalActivity": 1,
        "GenHealth": 3,
        "Asthma": 0,
        "KidneyDisease": 0,
        "SkinCancer": 0,
    },
    "Sample #3": {
        "BMI": -0.274602538,
        "PhysicalHealth": 2.091388419,
        "MentalHealth": 3.281068875,
        "SleepTime": 0.628775963,
        "Race": "White",
        "Smoking": 1,
        "AlcoholDrinking": 0,
        "Stroke": 0,
        "DiffWalking": 0,
        "Sex": 1,
        "AgeCategory": 9,
        "Diabetic": 3,
        "PhysicalActivity": 1,
        "GenHealth": 1,
        "Asthma": 1,
        "KidneyDisease": 0,
        "SkinCancer": 0,
    },
}

# =========================
# Utils
# =========================
def load_metrics_or_default(path="reports/metrics.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_METRICS

def build_processed_row_from_state() -> pd.DataFrame:
    row = {c: 0.0 for c in FEATURE_COLS}

    # continuous
    row["BMI"] = float(st.session_state["BMI"])
    row["PhysicalHealth"] = float(st.session_state["PhysicalHealth"])
    row["MentalHealth"] = float(st.session_state["MentalHealth"])
    row["SleepTime"] = float(st.session_state["SleepTime"])

    # race one-hot
    race_choice = st.session_state["Race"]
    for r in RACE_OPTIONS:
        row[f"Race_{r}"] = 1.0 if race_choice == r else 0.0

    # binary/ordinal
    for k in [
        "Smoking","AlcoholDrinking","Stroke","DiffWalking","Sex",
        "AgeCategory","Diabetic","PhysicalActivity","GenHealth",
        "Asthma","KidneyDisease","SkinCancer"
    ]:
        row[k] = float(st.session_state[k])

    return pd.DataFrame([row], columns=FEATURE_COLS)

def predict_with_stacking(X_df: pd.DataFrame, models_dict: dict):
    """
    Stacking Ensemble Prediction:
    Step 1: X_df (22 features) → 4 base models → 4 probabilities
    Step 2: 4 probabilities → meta_logistic → final prediction
    """
    try:
        # Convert DataFrame to numpy array for models that need it
        X_array = X_df.values
        
        # Step 1: Get probabilities from 4 base models
        base_probs = {}
        
        # SVM
        svm_proba = models_dict['svm'].predict_proba(X_array)
        base_probs['svm'] = float(svm_proba[0, 1]) if svm_proba.shape[1] > 1 else float(svm_proba[0, 0])
        
        # Softmax
        softmax_proba = models_dict['softmax'].predict_proba(X_array)
        base_probs['softmax'] = float(softmax_proba[0, 1]) if softmax_proba.shape[1] > 1 else float(softmax_proba[0, 0])
        
        # Naive Bayes
        nb_proba = models_dict['nb'].predict_proba(X_array)
        base_probs['nb'] = float(nb_proba[0, 1]) if nb_proba.shape[1] > 1 else float(nb_proba[0, 0])
        
        # Decision Tree
        dt_proba = models_dict['dt'].predict_proba(X_array)
        base_probs['dt'] = float(dt_proba[0, 1]) if dt_proba.shape[1] > 1 else float(dt_proba[0, 0])
        
        # Step 2: Create meta features (4 probabilities) - MUST MATCH TRAINING ORDER
        # Training order from CSV: svm_pred, nb_pred, dt_pred, softmax_pred
        meta_features = np.array([
            base_probs['svm'],       # Column 1: svm_pred
            base_probs['nb'],        # Column 2: nb_pred
            base_probs['dt'],        # Column 3: dt_pred
            base_probs['softmax']    # Column 4: softmax_pred
        ]).reshape(1, -1)
        
        # Step 3: Meta-learner prediction
        meta_model = models_dict['meta']
        
        if isinstance(meta_model, dict) and 'weights' in meta_model:
            # Manual logistic regression with saved weights
            weights = np.array(meta_model['weights'])
            threshold = meta_model.get('threshold', 0.5)
            
            # Check if weights include bias (5 elements: 4 features + 1 bias)
            if len(weights) == 5:
                # Last element is bias
                w = weights[:-1].reshape(-1, 1)
                bias = weights[-1]
                z = np.dot(meta_features, w)[0, 0] + bias
            else:
                # No bias term
                z = np.dot(meta_features, weights.reshape(-1, 1))[0, 0]
            
            # Sigmoid: p = 1 / (1 + exp(-z))
            final_proba = 1.0 / (1.0 + np.exp(-z))
            final_pred = 1 if final_proba >= threshold else 0
        else:
            # Standard sklearn model
            meta_proba = meta_model.predict_proba(meta_features)
            final_proba = float(meta_proba[0, 1]) if meta_proba.shape[1] > 1 else float(meta_proba[0, 0])
            final_pred = 1 if final_proba >= 0.5 else 0
        
        return final_pred, final_proba, base_probs
    
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

def metric_table(metrics_dict: dict) -> pd.DataFrame:
    rows = []
    for m, vals in metrics_dict.items():
        rows.append({
            "Model": m,
            "Accuracy": vals.get("accuracy"),
            "Precision": vals.get("precision"),
            "Recall": vals.get("recall"),
            "F1": vals.get("f1"),
            "AUC": vals.get("auc"),
        })
    return pd.DataFrame(rows)

def apply_sample(sample_name: str):
    s = SAMPLE_ROWS[sample_name]
    st.session_state["BMI"] = s["BMI"]
    st.session_state["PhysicalHealth"] = s["PhysicalHealth"]
    st.session_state["MentalHealth"] = s["MentalHealth"]
    st.session_state["SleepTime"] = s["SleepTime"]
    st.session_state["Race"] = s["Race"]
    st.session_state["Smoking"] = s["Smoking"]
    st.session_state["AlcoholDrinking"] = s["AlcoholDrinking"]
    st.session_state["Stroke"] = s["Stroke"]
    st.session_state["DiffWalking"] = s["DiffWalking"]
    st.session_state["Sex"] = s["Sex"]
    st.session_state["AgeCategory"] = s["AgeCategory"]
    st.session_state["Diabetic"] = s["Diabetic"]
    st.session_state["PhysicalActivity"] = s["PhysicalActivity"]
    st.session_state["GenHealth"] = s["GenHealth"]
    st.session_state["Asthma"] = s["Asthma"]
    st.session_state["KidneyDisease"] = s["KidneyDisease"]
    st.session_state["SkinCancer"] = s["SkinCancer"]

# init defaults if not exist
DEFAULT_INIT = SAMPLE_ROWS["Sample #3"]
for key in [
    "BMI","PhysicalHealth","MentalHealth","SleepTime","Race",
    "Smoking","AlcoholDrinking","Stroke","DiffWalking","Sex",
    "AgeCategory","Diabetic","PhysicalActivity","GenHealth",
    "Asthma","KidneyDisease","SkinCancer"
]:
    if key not in st.session_state:
        st.session_state[key] = DEFAULT_INIT.get(key, 0)

# =========================
# Header with title
# =========================
st.markdown("## ❤️ Heart Disease Prediction System", unsafe_allow_html=True)
st.markdown("**Ensemble Learning Model** | 17 Input Fields → 22 Features (with one-hot encoding)")
st.markdown("---")

metrics = load_metrics_or_default("assets/reports/metrics.json")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "⚖️ Model Comparison"])

# =========================
# TAB 1: PREDICTION
# =========================
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📝 Input Form (17 fields → 22 features)</div>', unsafe_allow_html=True)
        
        # Sample selection
        sample_col1, sample_col2, sample_col3 = st.columns([2, 1, 1])
        with sample_col1:
            sample_choice = st.selectbox("Load preset sample", list(SAMPLE_ROWS.keys()), label_visibility="collapsed")
        with sample_col2:
            if st.button("✓ Apply", use_container_width=True):
                apply_sample(sample_choice)
                st.rerun()
        with sample_col3:
            if st.button("↺ Reset", use_container_width=True):
                apply_sample("Sample #3")
                st.rerun()
        
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        
        # ===== Section A: Continuous Variables =====
        st.markdown('<div class="section-title">A) Continuous Variables (Standardized)</div>', unsafe_allow_html=True)
        
        a1, a2 = st.columns(2)
        with a1:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">BMI (Body Mass Index)</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Standardized value (use sample or leave at 0)</span>', unsafe_allow_html=True)
            st.number_input("BMI", key="BMI", step=0.1, format="%.3f", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with a2:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Physical Health (days/30)</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Standardized value</span>', unsafe_allow_html=True)
            st.number_input("PhysicalHealth", key="PhysicalHealth", step=0.1, format="%.3f", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        a3, a4 = st.columns(2)
        with a3:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Mental Health (days/30)</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Standardized value</span>', unsafe_allow_html=True)
            st.number_input("MentalHealth", key="MentalHealth", step=0.1, format="%.3f", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with a4:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Sleep Time (hours/day)</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Standardized value</span>', unsafe_allow_html=True)
            st.number_input("SleepTime", key="SleepTime", step=0.1, format="%.3f", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        
        # ===== Section B: Lifestyle (0/1) =====
        st.markdown('<div class="section-title">B) Lifestyle Factors (Yes=1, No=0)</div>', unsafe_allow_html=True)
        
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Smoking</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Do you smoke?</span>', unsafe_allow_html=True)
            st.selectbox("Smoking", [0, 1], key="Smoking", label_visibility="collapsed", format_func=lambda x: "Yes" if x==1 else "No")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with b2:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Alcohol Drinking</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Heavy drinking?</span>', unsafe_allow_html=True)
            st.selectbox("AlcoholDrinking", [0, 1], key="AlcoholDrinking", label_visibility="collapsed", format_func=lambda x: "Yes" if x==1 else "No")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with b3:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Physical Activity</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Regular exercise?</span>', unsafe_allow_html=True)
            st.selectbox("PhysicalActivity", [0, 1], key="PhysicalActivity", label_visibility="collapsed", format_func=lambda x: "Yes" if x==1 else "No")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        
        # ===== Section C: Medical Conditions (0/1) =====
        st.markdown('<div class="section-title">C) Medical Conditions (Yes=1, No=0)</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Stroke History</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Had a stroke?</span>', unsafe_allow_html=True)
            st.selectbox("Stroke", [0, 1], key="Stroke", label_visibility="collapsed", format_func=lambda x: "Yes" if x==1 else "No")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c2:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Difficulty Walking</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Difficulty walking stairs?</span>', unsafe_allow_html=True)
            st.selectbox("DiffWalking", [0, 1], key="DiffWalking", label_visibility="collapsed", format_func=lambda x: "Yes" if x==1 else "No")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c3:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Asthma</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Have asthma?</span>', unsafe_allow_html=True)
            st.selectbox("Asthma", [0, 1], key="Asthma", label_visibility="collapsed", format_func=lambda x: "Yes" if x==1 else "No")
            st.markdown('</div>', unsafe_allow_html=True)
        
        c4, c5 = st.columns(2)
        with c4:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Kidney Disease</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Have kidney disease?</span>', unsafe_allow_html=True)
            st.selectbox("KidneyDisease", [0, 1], key="KidneyDisease", label_visibility="collapsed", format_func=lambda x: "Yes" if x==1 else "No")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c5:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Skin Cancer</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Had skin cancer?</span>', unsafe_allow_html=True)
            st.selectbox("SkinCancer", [0, 1], key="SkinCancer", label_visibility="collapsed", format_func=lambda x: "Yes" if x==1 else "No")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        
        # ===== Section D: Demographics =====
        st.markdown('<div class="section-title">D) Demographics</div>', unsafe_allow_html=True)
        
        d1, d2 = st.columns(2)
        with d1:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Sex</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">Female=0, Male=1</span>', unsafe_allow_html=True)
            st.selectbox("Sex", [0, 1], key="Sex", label_visibility="collapsed", format_func=lambda x: "Male" if x==1 else "Female")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with d2:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Age Category (ordinal)</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">18-24:0, 25-29:1, ..., 80+:12</span>', unsafe_allow_html=True)
            st.number_input("AgeCategory", min_value=0, max_value=13, key="AgeCategory", step=1, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        d3, d4 = st.columns(2)
        with d3:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Race (One-hot encoded)</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">→ Creates 6 binary features</span>', unsafe_allow_html=True)
            st.selectbox("Race", RACE_OPTIONS, key="Race", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with d4:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">Diabetic Status</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">0=No, 1=Border, 2=Pregnancy, 3=Yes</span>', unsafe_allow_html=True)
            st.number_input("Diabetic", min_value=0, max_value=3, key="Diabetic", step=1, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        d5, d6 = st.columns(2)
        with d5:
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown('<span class="feature-label">General Health</span>', unsafe_allow_html=True)
            st.markdown('<span class="feature-hint">0=Poor, 1=Fair, 2=Good, 3=V.Good, 4=Excellent</span>', unsafe_allow_html=True)
            st.number_input("GenHealth", min_value=0, max_value=4, key="GenHealth", step=1, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with d6:
            st.empty()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Right column: Preview & Prediction
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin-top:0">Preview</h3>', unsafe_allow_html=True)
        
        X_df = build_processed_row_from_state()
        
        st.markdown(f'<div class="stats-box"><h3>{len(X_df.columns)}</h3><p>Features (22)</p></div>', unsafe_allow_html=True)
        
        st.dataframe(
            X_df.T.reset_index().rename(columns={"index": "Feature", 0: "Value"}),
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction section
    st.markdown('---')
    pred_col1, pred_col2, pred_col3 = st.columns([2, 1, 1])
    
    with pred_col1:
        do_predict = st.button("🔮 Predict Heart Disease", use_container_width=True, type="primary")
    with pred_col2:
        st.download_button(
            "⬇️ CSV",
            data=X_df.to_csv(index=False).encode("utf-8"),
            file_name="heart_input.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with pred_col3:
        if models:
            st.success("Models OK", icon="✅")
        else:
            st.error("Missing models", icon="⚠️")
    
    if do_predict:
        if not models:
            st.error("❌ Cannot predict: Models not loaded. Please ensure all 5 .joblib files are in assets/models/")
        else:
            pred, proba, base_probs = predict_with_stacking(X_df, models)
            
            if pred is not None:
                # Display final result with large styling
                if pred == 1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                                padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;">
                        <h1 style="color: white; margin: 0; font-size: 2.5rem;">⚠️ HIGH RISK</h1>
                        <h2 style="color: white; margin: 10px 0 0 0; font-size: 2rem;">
                            Probability: {proba:.2%}
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%); 
                                padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;">
                        <h1 style="color: white; margin: 0; font-size: 2.5rem;">✅ LOW RISK</h1>
                        <h2 style="color: white; margin: 10px 0 0 0; font-size: 2rem;">
                            Probability: {(1-proba):.2%}
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display base model probabilities with predictions
                st.markdown("---")
                st.markdown("### 🔍 Individual Model Predictions (Stacking Level 1)")
                
                base_col1, base_col2, base_col3, base_col4 = st.columns(4)
                
                # Thứ tự: SVM, Naive Bayes, Decision Tree, Softmax (theo thứ tự train)
                with base_col1:
                    svm_pred = "HIGH RISK" if base_probs['svm'] >= 0.5 else "LOW RISK"
                    svm_color = "🔴" if base_probs['svm'] >= 0.5 else "🟢"
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 10px; 
                                border: 2px solid {'#ff6b6b' if base_probs['svm'] >= 0.5 else '#51cf66'}; 
                                text-align: center;">
                        <h3 style="margin: 0; font-size: 1.1rem;">SVM</h3>
                        <h2 style="margin: 10px 0; font-size: 2rem; color: {'#ff6b6b' if base_probs['svm'] >= 0.5 else '#51cf66'};">
                            {base_probs['svm']:.1%}
                        </h2>
                        <p style="margin: 0; font-size: 1rem; font-weight: 600;">
                            {svm_color} {svm_pred}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with base_col2:
                    nb_pred = "HIGH RISK" if base_probs['nb'] >= 0.5 else "LOW RISK"
                    nb_color = "🔴" if base_probs['nb'] >= 0.5 else "🟢"
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 10px; 
                                border: 2px solid {'#ff6b6b' if base_probs['nb'] >= 0.5 else '#51cf66'}; 
                                text-align: center;">
                        <h3 style="margin: 0; font-size: 1.1rem;">Naive Bayes</h3>
                        <h2 style="margin: 10px 0; font-size: 2rem; color: {'#ff6b6b' if base_probs['nb'] >= 0.5 else '#51cf66'};">
                            {base_probs['nb']:.1%}
                        </h2>
                        <p style="margin: 0; font-size: 1rem; font-weight: 600;">
                            {nb_color} {nb_pred}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with base_col3:
                    dt_pred = "HIGH RISK" if base_probs['dt'] >= 0.5 else "LOW RISK"
                    dt_color = "🔴" if base_probs['dt'] >= 0.5 else "🟢"
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 10px; 
                                border: 2px solid {'#ff6b6b' if base_probs['dt'] >= 0.5 else '#51cf66'}; 
                                text-align: center;">
                        <h3 style="margin: 0; font-size: 1.1rem;">Decision Tree</h3>
                        <h2 style="margin: 10px 0; font-size: 2rem; color: {'#ff6b6b' if base_probs['dt'] >= 0.5 else '#51cf66'};">
                            {base_probs['dt']:.1%}
                        </h2>
                        <p style="margin: 0; font-size: 1rem; font-weight: 600;">
                            {dt_color} {dt_pred}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with base_col4:
                    softmax_pred = "HIGH RISK" if base_probs['softmax'] >= 0.5 else "LOW RISK"
                    softmax_color = "🔴" if base_probs['softmax'] >= 0.5 else "🟢"
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 10px; 
                                border: 2px solid {'#ff6b6b' if base_probs['softmax'] >= 0.5 else '#51cf66'}; 
                                text-align: center;">
                        <h3 style="margin: 0; font-size: 1.1rem;">Softmax</h3>
                        <h2 style="margin: 10px 0; font-size: 2rem; color: {'#ff6b6b' if base_probs['softmax'] >= 0.5 else '#51cf66'};">
                            {base_probs['softmax']:.1%}
                        </h2>
                        <p style="margin: 0; font-size: 1rem; font-weight: 600;">
                            {softmax_color} {softmax_pred}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.info("💡 These 4 predictions are combined by Meta-Learner (Logistic Regression) to produce the final result above.")




# =========================
# TAB 2: MODEL PERFORMANCE
# =========================
with tab2:
    st.markdown('<div class="section-title">📊 Model Performance Metrics</div>', unsafe_allow_html=True)
    
    df_metrics = metric_table(metrics)
    
    metric_col1, metric_col2 = st.columns([2, 1])
    
    with metric_col1:
        st.dataframe(
            df_metrics.style.format({
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1": "{:.3f}",
                "AUC": "{:.3f}",
            }),
            use_container_width=True,
            height=300,
            hide_index=True
        )
    
    with metric_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**Models**')
        for model in MODEL_NAMES:
            st.write(f"• {model}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('---')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('### 📈 Metric Comparison')
        metric_to_plot = st.selectbox(
            "Select metric to visualize",
            ["Accuracy", "Precision", "Recall", "F1", "AUC"],
            index=0
        )
        plot_df = df_metrics[["Model", metric_to_plot]].copy()
        plot_df[metric_to_plot] = pd.to_numeric(plot_df[metric_to_plot], errors="coerce")
        
        if plot_df[metric_to_plot].notna().any():
            st.bar_chart(plot_df.set_index("Model"), height=400)
        else:
            st.warning("No valid data to display. Update metrics.json with real values.")
    
    with col2:
        st.markdown('### 🎯 Metric Hints')
        st.markdown("""
        - **Accuracy**: Overall correctness
        - **Precision**: False positives (when model says YES)
        - **Recall**: False negatives (detecting actual cases)
        - **F1**: Balance of precision & recall
        - **AUC**: Model's discrimination ability
        """)


# =========================
# TAB 3: MODEL COMPARISON
# =========================
with tab3:
    st.markdown('<div class="section-title">⚖️ Model Comparison</div>', unsafe_allow_html=True)
    
    df_metrics = metric_table(metrics)
    df_num = df_metrics.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('### 🏆 Ranking by Metric')
        rank_metric = st.selectbox("Rank by", ["Accuracy", "F1", "AUC"], index=1)
        ranked = df_num.sort_values(rank_metric, ascending=False)
        
        ranking_df = ranked[["Model", rank_metric]].reset_index(drop=True)
        ranking_df.index = ranking_df.index + 1
        
        st.dataframe(
            ranking_df.style.format({rank_metric: "{:.3f}"}),
            use_container_width=True,
            height=300
        )
    
    with col2:
        st.markdown('### 📍 Precision vs Recall')
        scatter_df = df_num[["Model", "Precision", "Recall"]].dropna()
        
        if scatter_df.empty or scatter_df[["Precision", "Recall"]].isna().all().any():
            st.warning("⚠️ Insufficient data for scatter plot.\nUpdate assets/reports/metrics.json with real model metrics.")
        else:
            st.scatter_chart(scatter_df.set_index("Model"), height=400)
    
    st.markdown('---')
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('### 📊 AUC vs F1')
        scatter_df2 = df_num[["Model", "AUC", "F1"]].dropna()
        
        if scatter_df2.empty or scatter_df2[["AUC", "F1"]].isna().all().any():
            st.info("💡 Use this to find models that balance both metrics well")
        else:
            st.scatter_chart(scatter_df2.set_index("Model"), height=350)
    
    with col4:
        st.markdown('### 📋 Summary Stats')
        st.markdown(f"""
        **Total Models**: {len(MODEL_NAMES)}
        
        **Data Status**: 
        - ✓ 5 models configured
        - {'⚠️ Metrics needed' if df_num.isna().all().all() else '✓ Metrics loaded'}
        
        **Recommendation**:
        Update `assets/reports/metrics.json` with your trained model metrics
        """)

