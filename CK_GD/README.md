# ❤️ Heart Disease Prediction - Stacking Ensemble

Dự án dự đoán bệnh tim sử dụng **Stacking Ensemble Learning** với Streamlit Web App.

---

## 📁 Cấu Trúc Dự Án

```
CK_GD/
├── data/                           # 📊 DỮ LIỆU
│   ├── raw/                        # Dữ liệu gốc CSV
│   │   └── heart_disease_data.csv  # ← ĐẶT FILE CSV VÀO ĐÂY
│   └── processed/                  # Dữ liệu đã xử lý (optional)
│
├── scripts/                        # 💻 CODE TRAINING
│   ├── train_models.py            # Script train models chính
│   ├── evaluate.py                 # Đánh giá models (optional)
│   └── utils.py                    # Utility functions (optional)
│
├── notebooks/                      # 📓 JUPYTER NOTEBOOKS
│   └── exploratory_analysis.ipynb  # Phân tích dữ liệu (optional)
│
├── assets/                         # 📦 OUTPUTS
│   ├── models/                     # Models đã train
│   │   ├── svm_model.joblib        # ← AUTO GENERATED
│   │   ├── softmax_model.joblib    # ← AUTO GENERATED
│   │   ├── nb_model.joblib         # ← AUTO GENERATED
│   │   ├── dt_model.joblib         # ← AUTO GENERATED
│   │   ├── meta_logistic.joblib    # ← AUTO GENERATED
│   │   └── preprocessor.joblib     # StandardScaler (optional)
│   └── reports/                    # Metrics và reports
│       └── metrics.json            # ← AUTO GENERATED
│
├── app.py                          # 🌐 STREAMLIT WEB APP
├── HUONG_DAN_MODEL.md             # Hướng dẫn chi tiết
├── README.md                       # File này
└── requirements.txt                # Python dependencies

```

---

## 🎯 Quy Trình Sử Dụng

### **Bước 1: Chuẩn Bị Dữ Liệu**

1. Đặt file CSV vào `data/raw/heart_disease_data.csv`
2. File CSV phải có **23 cột**:
   - Cột 1: `HeartDisease` (target) - 0 hoặc 1
   - Cột 2-23: 22 features theo thứ tự:

```
HeartDisease, BMI, PhysicalHealth, MentalHealth, SleepTime,
Race_American Indian/Alaskan Native, Race_Asian, Race_Black, 
Race_Hispanic, Race_Other, Race_White,
Smoking, AlcoholDrinking, Stroke, DiffWalking, Sex, AgeCategory,
Diabetic, PhysicalActivity, GenHealth, Asthma, KidneyDisease, SkinCancer
```

**Ví dụ dòng đầu tiên:**
```csv
HeartDisease,BMI,PhysicalHealth,MentalHealth,SleepTime,Race_American Indian/Alaskan Native,Race_Asian,Race_Black,Race_Hispanic,Race_Other,Race_White,Smoking,AlcoholDrinking,Stroke,DiffWalking,Sex,AgeCategory,Diabetic,PhysicalActivity,GenHealth,Asthma,KidneyDisease,SkinCancer
0,-1.844750159,-0.046751049,3.281068875,-1.460353521,0,0,0,0,0,1,1,0,0,0,0,7,3,1,3,1,0,1
```

---

### **Bước 2: Train Models**

Chạy script training:

```bash
cd scripts
python train_models.py
```

**Output:**
```
============================================================
STACKING ENSEMBLE TRAINING
============================================================

[1/8] Loading data...
  ✓ Loaded: 100000 samples
  ✓ Features: 22
  ✓ Positive cases: 9000 (9.0%)

[2/8] Splitting data...
  ✓ Training set: 80000 samples
  ✓ Test set: 20000 samples

[3/8] Training LEVEL 1: Base Models...
  [1/4] SVM Classifier ✓
  [2/4] Softmax Regression ✓
  [3/4] Naive Bayes ✓
  [4/4] Decision Tree ✓

[4/8] Creating Meta Features (Stacking)...
  ✓ Meta features shape: (80000, 4)

[5/8] Training LEVEL 2: Meta-Learner...
  ✓ Meta-learner trained
  ✓ Weights: [0.45, 0.32, 0.18, 0.28]

[6/8] Evaluating all models...
  ✓ SVM: Accuracy=0.9120, AUC=0.9580
  ✓ Softmax: Accuracy=0.9050, AUC=0.9510
  ...

[7/8] Saving models...
  ✓ Saved: assets/models/svm_model.joblib
  ...

[8/8] Saving metrics...
  ✓ Saved: assets/reports/metrics.json

✅ TRAINING COMPLETE!
```

Script sẽ tự động:
- ✅ Train 4 base models + 1 meta-learner
- ✅ Save models vào `assets/models/`
- ✅ Save metrics vào `assets/reports/metrics.json`

---

### **Bước 3: Chạy Web App**

```bash
streamlit run app.py
```

Mở browser: **http://localhost:8501**

App có 3 tabs:
- **🔮 Prediction**: Nhập 17 fields → Dự đoán bệnh tim
- **📊 Model Performance**: Xem metrics của 5 models
- **⚖️ Model Comparison**: So sánh models

---

## 🏗️ Kiến Trúc Model (Stacking Ensemble)

```
┌─────────────────────────────────────────┐
│   INPUT: 22 Features                    │
│   (từ 17 input fields)                  │
└────────────┬────────────────────────────┘
             │
    ┌────────┴────────┐
    │   LEVEL 1       │
    │   Base Models   │
    └────────┬────────┘
             │
    ┌────────┴────────────────────────────┐
    │  4 Base Models predict độc lập:     │
    │  ├─ SVM → probability p1            │
    │  ├─ Softmax → probability p2        │
    │  ├─ Naive Bayes → probability p3    │
    │  └─ Decision Tree → probability p4  │
    └────────┬────────────────────────────┘
             │
    ┌────────┴────────┐
    │   LEVEL 2       │
    │   Meta-Learner  │
    └────────┬────────┘
             │
    ┌────────┴────────────────────────────┐
    │  Meta-Logistic nhận [p1,p2,p3,p4]  │
    │  → Quyết định cuối: 0 hoặc 1       │
    └─────────────────────────────────────┘
```

---

## 📊 Định Dạng Dữ Liệu

### Input Features (17 fields):
1. **Continuous (4)**: BMI, PhysicalHealth, MentalHealth, SleepTime
2. **Lifestyle (3)**: Smoking, AlcoholDrinking, PhysicalActivity
3. **Medical (5)**: Stroke, DiffWalking, Asthma, KidneyDisease, SkinCancer
4. **Demographics (5)**: Sex, AgeCategory, Race, Diabetic, GenHealth

### Output Features (22):
- 4 continuous
- **6 from Race** (one-hot encoded)
- 12 binary/ordinal

---

## 🔧 Requirements

```txt
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

Cài đặt:
```bash
pip install -r requirements.txt
```

---

## 📝 Cách Sử Dụng Chi Tiết

### 1. Training từ đầu

```bash
# Bước 1: Đặt CSV vào data/raw/
# Bước 2: Train
python scripts/train_models.py

# Bước 3: Chạy app
streamlit run app.py
```

### 2. Chỉ chạy app (đã có models)

```bash
streamlit run app.py
```

### 3. Re-train với data mới

```bash
# Thay file CSV trong data/raw/
# Train lại
python scripts/train_models.py
```

---

## 🎓 Hiểu Output Files

### `assets/models/*.joblib`
- **svm_model.joblib**: SVM base model
- **softmax_model.joblib**: Logistic Regression base model  
- **nb_model.joblib**: Naive Bayes base model
- **dt_model.joblib**: Decision Tree base model
- **meta_logistic.joblib**: Meta-learner (combines 4 above)

### `assets/reports/metrics.json`
```json
{
  "SVM Classifier": {
    "accuracy": 0.912,
    "precision": 0.887,
    "recall": 0.923,
    "f1": 0.905,
    "auc": 0.958
  },
  ...
}
```

---

## ⚠️ Troubleshooting

### Lỗi: "Data file not found"
```
❌ Đặt file CSV vào: data/raw/heart_disease_data.csv
```

### Lỗi: "Column names don't match"
```
❌ Kiểm tra thứ tự 22 features trong CSV
```

### Lỗi: "No module named streamlit"
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### App không load models
```bash
# Re-train models
python scripts/train_models.py
```

---

## 📚 Tài Liệu Thêm

- [HUONG_DAN_MODEL.md](HUONG_DAN_MODEL.md) - Hướng dẫn chi tiết về model
- [scripts/train_models.py](scripts/train_models.py) - Source code training

---

## 👥 Team

- **Dự án**: Heart Disease Prediction
- **Mục đích**: Giáo dục và nghiên cứu

---

## 📄 License

Educational Project - 2025
