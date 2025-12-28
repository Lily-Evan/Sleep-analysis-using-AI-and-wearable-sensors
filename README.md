# Sleep Stage Classification using AI and Wearable Sensors

## Overview
This project investigates the use of machine learning for sleep stage classification based on data collected from wearable devices. By leveraging physiological and contextual features (e.g., sleep duration, body temperature, movement-related indicators), the goal is to predict sleep stages such as **Awake, Light, Deep, REM** and explore the potential of AI-driven sleep analysis for digital health applications.

---

## Dataset
The dataset used is the **Wearable Tech Sleep Quality** dataset:

- Source: Wearable sleep tracking data (Kaggle: `wearable_tech_sleep_quality.csv`)
- Includes:
  - Sleep-related features (e.g., duration, efficiency/quality indicators)
  - Physiological proxies (e.g., body temperature)
  - Environmental/contextual features (e.g., room temperature)
  - Sleep stage labels: **Awake, Light, Deep, REM**

Data preprocessing steps include:
- Handling missing values
- Type conversion and basic cleaning
- Encoding categorical variables
- Splitting into features (`X`) and labels (`y`)
- Train–test split with a test set of **1000 samples**

---

## Problem Formulation

- **Task:** Multiclass classification of sleep stages  
- **Classes:** `Awake`, `Light`, `Deep`, `REM`  
- **Input:** Engineered features from wearable sleep quality dataset  
- **Output:** Predicted sleep stage for each sample

This is a **challenging** multiclass prediction problem, as the classes are often overlapping from a feature perspective and labels may include noise or coarse granularity.

---

## Machine Learning Pipeline

### Models Evaluated
The following supervised learning models were trained and compared:

- **K-Nearest Neighbors (KNN)**
- **Random Forest** (with hyperparameter tuning via `GridSearchCV`)
- **Logistic Regression**
- **XGBoost Classifier**

Feature scaling and dimensionality reduction techniques (e.g., `StandardScaler`, PCA if used) are applied where appropriate to stabilize training and improve performance.

---

## Evaluation Strategy

- Train/test split with **1000 samples** in the test set
- Multiclass evaluation using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
  - **Macro and weighted averages**

Classification reports are computed using `sklearn.metrics.classification_report`.

---

## Results

### Random Forest (Best Model)

Best hyperparameters (from GridSearchCV):

```python
{'max_depth': 10, 'n_estimators': 100}
