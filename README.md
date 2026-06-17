# Day 11 — Titanic Survival Prediction: Random Forest & XGBoost

Part of the [`ai-engineer-journey`](https://github.com/parma-nand/ai-engineer-journey) series — a day-by-day log of hands-on ML/AI engineering practice.

---

## 🎯 Objective

Build and compare two ensemble classifiers — **Random Forest** and **XGBoost** — on the classic Titanic survival dataset. Covers the full supervised learning workflow: data cleaning, feature engineering, model training, cross-validation, and evaluation.

---

## 📄 File

```
ai-engineer-journey/
└── day11.py    ← Single-file ML experiment (self-contained)
```

---

## 🔬 What This Script Does

### 1. Data Loading & Cleaning
- Loads Titanic dataset directly from GitHub (DataScienceDojo raw CSV)
- Fills missing `Age` values with column mean
- Drops non-predictive columns (`Name`, `Ticket`, `Cabin`, `PassengerId`)
- Encodes categorical columns (`Sex`, `Embarked`) using `pd.get_dummies`

### 2. Feature Engineering
- Target variable: `Survived`
- Features: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex_male`, `Embarked_Q`, `Embarked_S`

### 3. Train/Test Split
- 80/20 stratified split using `train_test_split`

### 4. Model Training
| Model | Key Hyperparameters |
|-------|-------------------|
| `RandomForestClassifier` | `n_estimators=100`, `random_state=42` |
| `XGBClassifier` | `use_label_encoder=False`, `eval_metric='logloss'` |

### 5. Evaluation
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1 per class)
- **Confusion Matrix** (visualised with Seaborn heatmap)
- **K-Fold Cross-Validation** (`cv=5`) for both models

---

## 📊 Sample Results

| Metric | Random Forest | XGBoost |
|--------|:---:|:---:|
| Accuracy | ~83% | ~84% |
| CV Mean (5-fold) | ~81% | ~82% |

> Results may vary slightly based on random seed and data version.

---

## 🛠️ Tech Stack

```
numpy        — numerical operations
pandas       — data loading, cleaning, feature encoding
matplotlib   — plotting
seaborn      — confusion matrix heatmap
scikit-learn — train_test_split, cross_val_score, KFold,
               RandomForestClassifier, accuracy_score,
               classification_report, confusion_matrix
xgboost      — XGBClassifier
```

---

## 🚀 Run It

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

# Run the script
python day11.py
```

---

## 🧠 Key Concepts Practiced

- **Ensemble methods** — Bagging (Random Forest) vs Boosting (XGBoost)
- **Cross-validation** — Proper model evaluation beyond a single train/test split
- **Overfitting check** — Comparing CV score vs hold-out test accuracy
- **Feature encoding** — One-hot encoding categorical variables with `pd.get_dummies`
- **Missing value imputation** — Mean imputation for continuous features
- **Confusion matrix analysis** — Understanding False Positives vs False Negatives

---

## 📌 Interview Talking Points

> Useful for AI/ML Engineer interviews at product companies (Sarvam AI, Observe.AI, etc.)

- **Why XGBoost over Random Forest?** XGBoost uses gradient boosting (sequential correction of errors) vs Random Forest's parallel bagging. XGBoost generally achieves better accuracy at the cost of more hyperparameter tuning.
- **Why cross-validation?** A single train/test split can be lucky or unlucky. K-Fold CV gives a more stable estimate of generalisation performance.
- **OOB Score** — Random Forest can compute Out-of-Bag score as a free internal CV estimate without needing a separate val set.
- **Feature Importance** — Both models expose `.feature_importances_` to explain which features drive predictions.

---

## 🗂️ Part of `ai-engineer-journey`

| Day | Topic |
|-----|-------|
| ... | ... |
| Day 11 | ✅ Titanic — Random Forest + XGBoost + Cross-Validation |
| Day 12 | Coming up... |
