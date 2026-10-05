# Medical Insurance Cost Prediction using Linear Regression

## Overview

This is a small, beginner-friendly machine learning project that predicts **medical insurance charges** from basic personal information. It walks through a complete regression workflow – data loading, exploratory analysis, preprocessing, model training, regularization, hyperparameter tuning, cross-validation and evaluation – using only **linear models**: Linear Regression, Ridge and Lasso.

## Problem Statement

The goal is to predict the individual medical insurance cost (`charges`) of a person based on demographic and lifestyle-related features such as age, BMI, number of children, smoking status, sex and region.

## Dataset

**Medical Cost Personal Dataset** (`insurance.csv`) – 1,338 rows, 7 columns.

| Column | Description |
|--------|-------------|
| `age` | Age of the person |
| `sex` | Gender |
| `bmi` | Body mass index |
| `children` | Number of children covered by insurance |
| `smoker` | Smoking status (yes / no) |
| `region` | Residential region in the US |
| `charges` | Medical insurance cost (**target variable**) |

The original dataset comes from Kaggle: <https://www.kaggle.com/datasets/mirichoi0218/insurance>

The notebook downloads the same `insurance.csv` from a **public raw/mirror URL** on GitHub, so **no Kaggle authentication, API key or manual download is required**.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Machine Learning Concepts

- **Linear Regression** – fits a straight-line (linear) relationship between the features and the target by minimizing squared errors.
- **Ridge Regression** – Linear Regression with an L2 penalty that shrinks coefficients to reduce overfitting.
- **Lasso Regression** – Linear Regression with an L1 penalty that can shrink some coefficients exactly to zero.
- **L1 Regularization** – penalizes the sum of the absolute values of the coefficients (used by Lasso).
- **L2 Regularization** – penalizes the sum of the squared coefficients (used by Ridge).
- **Train-Test Split** – the data is divided into training data (to learn from) and test data (to evaluate on unseen examples).
- **Cross Validation** – the training data is split into several folds; the model is trained and validated on different folds to get a more reliable performance estimate.
- **GridSearchCV** – tries every value in a hyperparameter grid with cross-validation and keeps the best one.
- **MAE, RMSE, R²** – regression metrics (see [Evaluation Metrics](#evaluation-metrics)).

## Project Workflow

```text
Dataset
   ↓
EDA
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
Linear Regression
   ↓
Ridge
   ↓
Lasso
   ↓
Hyperparameter Tuning
   ↓
Cross Validation
   ↓
Model Comparison
   ↓
Final Evaluation
```

All preprocessing (imputation, scaling, one-hot encoding) is done **inside scikit-learn `Pipeline`s** so that no information from the test set leaks into training.

## Models

| Model | Description | Tuned hyperparameter |
|-------|-------------|----------------------|
| Linear Regression | Baseline model with no regularization | – |
| Ridge Regression | L2-regularized linear model | `alpha` ∈ {0.01, 0.1, 1, 10, 50, 100} |
| Lasso Regression | L1-regularized linear model | `alpha` ∈ {0.0001, 0.001, 0.01, 0.1, 1} |

Ridge and Lasso are tuned with `GridSearchCV` using 5-fold cross-validation and `scoring="neg_root_mean_squared_error"`.

## Evaluation Metrics

### MAE

Mean Absolute Error – the average absolute prediction error, in the same unit as `charges`. Lower is better.

### RMSE

Root Mean Squared Error – like MAE, but it penalizes larger errors more strongly. Lower is better.

### R²

R-squared – measures how much of the variation in the target is explained by the model. Closer to 1 is better.

## Results

This README intentionally contains **no hardcoded results**. The notebook calculates and prints the real results when you run it, including:

- a model comparison table (MAE, RMSE, R²) sorted by RMSE and an RMSE bar chart,
- the best model and its best hyperparameter,
- 5-fold cross-validation performance of the best model,
- actual-vs-predicted and residual plots,
- the top 10 feature coefficients,
- a final results summary and an automatically generated conclusion.

## How to Run

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
linear_regression_insurance.ipynb
```

Run all cells. The dataset downloads automatically (internet connection required).

## Project Structure

```text
linear-regression-insurance/
│
├── linear_regression_insurance.ipynb
└── README.md
```

## Future Improvements

- Log transformation of `charges`
- More feature engineering (e.g. `smoker` × `bmi` interaction)
- Elastic Net
- Outlier handling
- Larger hyperparameter search
- Comparison with non-linear models
