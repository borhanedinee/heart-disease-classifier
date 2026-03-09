# ❤️ Heart Disease Classifier

A production-ready machine learning API that predicts the presence of heart disease from clinical patient data.

Trained on the UCI Heart Disease dataset, served via FastAPI, and containerized with Docker.

---

## 🧠 Model

| Property | Detail |
|----------|--------|
| Algorithm | Random Forest Classifier |
| Trees | 100 estimators |
| Max depth | 6 (prevents overfitting) |
| Seed | 42 (full reproducibility) |
| Task | Binary classification (0 = healthy, 1 = disease) |

---

## 📊 Results (Validation Set)

| Metric | Score |
|--------|-------|
| F1 Score | 0.8077 |
| AUROC | 0.8880 |
| Accuracy | 0.78 |

---

## 📁 Dataset

- **Name**: UCI Heart Disease Dataset (Cleveland)
- **Samples**: 303 patients
- **Features**: 13 clinical features (age, sex, chest pain type, cholesterol, etc.)
- **Source**: UCI Machine Learning Repository
- **License**: Public domain
- **Split**: 70% train / 15% val / 15% test (stratified)

---

## 🧹 Data Cleaning

- Replaced missing values encoded as `?` with NaN
- Dropped rows with missing values (6 rows removed)
- Binarized target column: 0 = no disease, 1 = disease present

---

## 📂 Project Structure

```
heart-disease-classifier/
├── train.py              # Model training & evaluation
├── app.py                # FastAPI prediction endpoint
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── model.pkl             # Saved trained model
├── scaler.pkl            # Saved StandardScaler
├── confusion_matrix.png  # Validation confusion matrix
├── feature_importance.png# Feature importance chart
└── data/
    └── heart.csv         # Dataset
```

---

## 🔌 API Usage

### Health Check
```bash
GET /
→ { "status": "Heart Disease Classifier API is running 🚀" }
```

### Predict
```bash
POST /predict
Content-Type: application/json

{
  "age": 67,
  "sex": 1,
  "cp": 2,
  "trestbps": 160,
  "chol": 286,
  "fbs": 0,
  "restecg": 0,
  "thalach": 108,
  "exang": 1,
  "oldpeak": 1.5,
  "slope": 1,
  "ca": 3,
  "thal": 2
}

→ {
    "prediction": 1,
    "result": "Disease likely 🔴",
    "confidence": 0.87
  }
```

---

## ⚙️ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/borhanedinee/heart-disease-classifier.git
cd heart-disease-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```

### 4. Start the API
```bash
python -m uvicorn app:app --reload
```

### 5. Open docs
```
http://localhost:8000/docs
```

---

## 🐳 Run with Docker

```bash
docker build -t heart-disease-classifier .
docker run -d -p 8000:8000 heart-disease-classifier
```

---

## ☁️ Deployment

The app is fully containerized and can be deployed on any cloud provider (GCP, AWS, Azure) using Docker.

```bash
# On any VM or cloud instance
git clone https://github.com/borhanedinee/heart-disease-classifier.git
cd heart-disease-classifier
docker build -t heart-disease-classifier .
docker run -d -p 8000:8000 heart-disease-classifier
```

---

## 🔁 Reproducibility

- `numpy` seed set to 42
- `scikit-learn` random_state set to 42
- Stratified train/val/test split
- `model.pkl` and `scaler.pkl` saved for consistent inference

---

## 👤 Author

**BOUSSAHA Borhanedine**
[GitHub](https://github.com/borhanedinee) · [LinkedIn](https://www.linkedin.com/in/borhanedine-boussaha-a02045251)