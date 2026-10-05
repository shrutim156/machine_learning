# Breast Cancer Prediction using Logistic Regression

## Project Overview
This beginner-friendly machine learning project predicts whether a breast tumor is **malignant (cancerous)** or **benign (non-cancerous)** from measurements of cell nuclei. It uses **Logistic Regression** and walks through a complete workflow: downloading data from Kaggle, cleaning it, training a model and evaluating it with standard classification metrics.

The full, explained code lives in `logistic_regression_breast_cancer.ipynb`.

## Dataset Information
- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** Kaggle (`uciml/breast-cancer-wisconsin-data`), downloaded in code with `kagglehub`
- **Size:** 569 samples, 30 numeric features
- **Target:** `diagnosis` (M = Malignant, B = Benign)
- **Class split:** 357 Benign (62.7%) and 212 Malignant (37.3%)
- **Features:** 10 measurements of each cell nucleus (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each given as the **mean**, **standard error (se)** and **worst** value, which gives 30 features.
- **Columns removed:** `id` (just an identifier) and `Unnamed: 32` (an entirely empty column).

## Problem Statement
Given the measurements of a tumor sample, build a model that classifies it as **malignant (1)** or **benign (0)**. This is a **binary classification** problem. Because missing a cancer case is costly, **recall** is treated as an especially important metric.

## Why Logistic Regression is Suitable
- The target has **two classes**, which is exactly what Logistic Regression is designed for.
- It outputs a **probability**, which is useful in medical settings.
- It is **simple, fast** and works well on small, mostly numeric datasets like this one.
- It is **interpretable**: each feature has a coefficient showing whether it pushes the prediction towards malignant or benign.
- It makes a strong, easy-to-explain **baseline** before trying complex models.

## Technologies / Libraries Used
- Python 3
- kagglehub
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- Jupyter Notebook

## Project Workflow
1. Import libraries
2. Download the dataset from Kaggle with `kagglehub`
3. Load the CSV into a pandas DataFrame
4. Explore the data (shape, types, statistics, class balance)
5. Clean the data (check missing values, drop useless columns, check duplicates)
6. Encode the target (M = 1, B = 0)
7. Select features (X) and target (y)
8. Train-test split (80% / 20%, stratified)
9. Scale features with `StandardScaler`
10. Train the Logistic Regression model
11. Predict on the test set
12. Evaluate and visualize the results
13. Interpret the most influential features

## Data Preprocessing
- **Missing values:** the only column with missing values was `Unnamed: 32` (100% empty), so it was dropped.
- **Irrelevant column:** `id` was dropped because it carries no predictive information.
- **Duplicates:** checked, none found.
- **Target encoding:** `M` mapped to `1`, `B` mapped to `0`.
- **Train-test split:** 80% training (455 samples) and 20% testing (114 samples) using `stratify=y` and `random_state=42`.
- **Feature scaling:** `StandardScaler` was fitted on the training data only and then applied to the test data, which avoids data leakage.

## Model Implementation
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
```

## Evaluation Metrics
| Metric | What it tells us |
|---|---|
| Accuracy | Share of all predictions that were correct |
| Precision | When the model predicts malignant, how often it is right |
| Recall | Share of real malignant tumors the model detected |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Counts of TN, FP, FN and TP |
| Classification Report | Precision, recall and F1 for each class |
| ROC Curve / ROC-AUC | How well the model separates the two classes across all thresholds |

## Results
Test-set results with `random_state=42` (your numbers should match if you run the notebook as-is):

| Metric | Score |
|---|---|
| Accuracy | 96.49% |
| Precision (Malignant) | 97.50% |
| Recall (Malignant) | 92.86% |
| F1-score (Malignant) | 95.12% |
| ROC-AUC | 0.996 |

**Confusion matrix** (rows = actual, columns = predicted):

|  | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 71 | 1 |
| **Actual Malignant** | 3 | 39 |

The model made only 4 mistakes out of 114 test samples: 1 false positive and 3 false negatives.

## Conclusion
Logistic Regression performs very well on this dataset, reaching about 96% accuracy and a ROC-AUC of about 0.996. It correctly identified most malignant cases while producing very few false alarms. The 3 false negatives show where improvement is most valuable, since missing a cancer case is the most serious error.

**Possible improvements:** cross-validation, hyperparameter tuning with `GridSearchCV`, lowering the decision threshold to increase recall, and comparing against Random Forest, SVM or KNN.

> **Disclaimer:** This project is for educational purposes only and must not be used for real medical diagnosis.

## How to Run the Project
1. **Clone or download** this project and open a terminal in its folder.
2. **Install the requirements:**
   ```bash
   pip install kagglehub pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. **Start Jupyter:**
   ```bash
   jupyter notebook logistic_regression_breast_cancer.ipynb
   ```
4. **Run all cells** (Kernel > Restart & Run All). The dataset is downloaded automatically by `kagglehub`.

An internet connection is required for the download. If Kaggle asks for authentication, create an API token in your Kaggle account settings (Settings > API) and either place `kaggle.json` in `~/.kaggle/` or set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

## Kaggle Dataset Source
[Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) by UCI Machine Learning. The data originates from the UCI Machine Learning Repository.
