# Heart Disease Classifier

Binary classification model to predict the presence of heart disease using a Random Forest classifier trained on the UCI Heart Disease dataset.

## Dataset
- **Name**: UCI Heart Disease Dataset (Cleveland)
- **Samples**: 303 patients
- **Features**: 13 clinical features (age, sex, chest pain type, resting blood pressure, cholesterol, etc.)
- **Target**: Binary — 0 = no disease, 1 = disease
- **Source**: UCI Machine Learning Repository
- **License**: Public domain

## Model
- **Algorithm**: Random Forest Classifier
- **Trees**: 100 estimators
- **Max depth**: 6 (to prevent overfitting)
- **Seed**: 42 (for reproducibility)

## Data Split
| Set        | Samples |
|------------|---------|
| Train      | 212     |
| Validation | 45      |
| Test       | 46      |

## Results (Validation Set)
| Metric   | Score  |
|----------|--------|
| F1 Score | 0.8077 |
| AUROC    | 0.8880 |
| Accuracy | 0.78   |

## Data Cleaning
- Replaced missing values encoded as `?` with NaN
- Dropped rows with missing values
- Binarized target column (0 = no disease, 1 = disease presence)

## Reproducibility
- NumPy seed set to 42
- Scikit-learn random_state set to 42 in all stochastic components
- Fixed train/val/test split via stratified sampling

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python train.py
```

### Output
- `confusion_matrix.png` — validation set confusion matrix
- `feature_importance.png` — feature importance chart

## Requirements
See `requirements.txt` for full list of dependencies.

## Checkpoint
Final model checkpoint reference: `RandomForest_seed42_depth6`