# app.py
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os
from pathlib import Path

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
@st.cache_resource
def load_models():
    """Load 5 models cho Stacking Ensemble"""
    try:
        models_dict = {}
        model_files = {
            'svm': 'assets/models/svm_model.joblib',
            'softmax': 'assets/models/softmax_model.joblib',
            'nb': 'assets/models/nb_model.joblib',
            'dt': 'assets/models/dt_model.joblib',
            'meta': 'assets/models/meta_logistic.joblib'
        }
        
        missing_models = []
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                models_dict[name] = joblib.load(filepath)
            else:
                missing_models.append(filepath)
        
        if missing_models:
            st.warning(f"⚠️ Thiếu {len(missing_models)}/5 models: {missing_models}")
            return None
        
        st.success("✅ Đã load 5 models thành công!")
        return models_dict
    
    except Exception as e:
        st.error(f"❌ Lỗi khi load models: {e}")
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
        # Step 1: Get probabilities from 4 base models
        base_probs = {}
        base_probs['svm'] = models_dict['svm'].predict_proba(X_df)[0, 1]
        base_probs['softmax'] = models_dict['softmax'].predict_proba(X_df)[0, 1]
        base_probs['nb'] = models_dict['nb'].predict_proba(X_df)[0, 1]
        base_probs['dt'] = models_dict['dt'].predict_proba(X_df)[0, 1]
        
        # Step 2: Create meta features (4 probabilities)
        meta_features = pd.DataFrame([[
            base_probs['svm'],
            base_probs['softmax'],
            base_probs['nb'],
            base_probs['dt']
        ]], columns=['svm_prob', 'softmax_prob', 'nb_prob', 'dt_prob'])
        
        # Step 3: Meta-learner prediction
        final_proba = models_dict['meta'].predict_proba(meta_features)[0, 1]
        final_pred = 1 if final_proba >= 0.5 else 0
        
        return final_pred, final_proba, base_probs
    
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {e}")
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
                # Display final result
                if pred == 1:
                    st.error(f"⚠️ **FINAL RESULT: HIGH RISK** | Probability: {proba:.2%}")
                else:
                    st.success(f"✅ **FINAL RESULT: LOW RISK** | Probability: {(1-proba):.2%}")
                
                # Display base model probabilities
                st.markdown("---")
                st.markdown("### 🔍 Base Model Probabilities (Stacking Level 1)")
                
                base_col1, base_col2, base_col3, base_col4 = st.columns(4)
                
                with base_col1:
                    st.metric("SVM", f"{base_probs['svm']:.2%}")
                with base_col2:
                    st.metric("Softmax", f"{base_probs['softmax']:.2%}")
                with base_col3:
                    st.metric("Naive Bayes", f"{base_probs['nb']:.2%}")
                with base_col4:
                    st.metric("Decision Tree", f"{base_probs['dt']:.2%}")
                
                st.info("💡 These 4 probabilities are combined by Meta-Learner (Logistic Regression) to produce the final prediction.")



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

