# ⚙️ Industrial Predictive Maintenance System

An end-to-end machine learning system for predicting industrial machine failures from operating and sensor parameters.

The project explores multiple machine learning approaches for highly imbalanced failure-prediction data, including Logistic Regression, Random Forest, SMOTE-based approaches, and XGBoost. The final system uses a tuned XGBoost classifier and provides an interactive Streamlit dashboard for machine failure-risk prediction.

---

## 📌 Overview

Unexpected machine failures can result in:

- Production downtime
- Maintenance costs
- Equipment damage
- Reduced operational efficiency
- Safety risks

Predictive maintenance aims to identify machines that are likely to fail **before an actual failure occurs**, allowing maintenance teams to intervene proactively.

This project develops a complete predictive-maintenance pipeline:

```text
Raw Machine Data
       ↓
Data Cleaning & EDA
       ↓
Feature Preprocessing
       ↓
Class Imbalance Analysis
       ↓
Multiple ML Models
       ↓
Model Comparison
       ↓
XGBoost Hyperparameter Tuning
       ↓
Model Interpretation
       ↓
Model Serialization
       ↓
Streamlit Deployment
       ↓
Machine Failure Prediction
```

---

# 🎯 Problem Statement

Given a machine's operating conditions and sensor measurements, predict whether the machine is likely to experience a failure.

The model performs binary classification:

```text
0 → No Failure
1 → Failure
```

A major challenge is the severe class imbalance in the dataset. Machine failures are much less frequent than normal operating conditions.

Therefore, simply maximizing accuracy is not sufficient.

The project focuses particularly on:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

to properly evaluate minority-class failure detection.

---

# 📊 Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains machine operating measurements along with failure information.

### Input Features

| Feature | Description |
|---|---|
| `Type` | Machine/product type |
| `Air temperature [K]` | Ambient air temperature |
| `Process temperature [K]` | Process temperature |
| `Rotational speed [rpm]` | Machine rotational speed |
| `Torque [Nm]` | Applied machine torque |
| `Tool wear [min]` | Accumulated tool wear |

### Target

```text
0 → No Failure
1 → Failure
```

---

# ⚠️ Class Imbalance

The dataset contains significantly fewer failure cases than normal operating cases.

In the training set:

```text
No Failure : 7729
Failure    : 271
```

Approximately:

```text
No Failure → 96.6%
Failure    → 3.4%
```

This makes the problem highly imbalanced.

A model could achieve very high accuracy while still missing most actual failures.

For example, the initial Logistic Regression model achieved approximately:

```text
Accuracy = 96.75%
Recall   = 10.29%
```

This means the model was missing most actual machine failures despite its high accuracy.

Therefore, class imbalance became a major focus of the modeling process.

---

# 🔬 Exploratory Data Analysis

The exploratory analysis investigated:

- Feature distributions
- Target imbalance
- Relationships between sensor variables
- Failure patterns
- Correlations between numerical features
- Machine type distribution
- Potential predictive features

Important operating variables investigated included:

- Torque
- Rotational speed
- Tool wear
- Air temperature
- Process temperature

---

# ⚙️ Data Preprocessing

The preprocessing pipeline handles both numerical and categorical variables.

### Numerical Features

Numerical features are standardized using `StandardScaler`.

```text
Numerical Features
        ↓
StandardScaler
```

### Categorical Feature

`Type` is transformed using one-hot encoding:

```text
Type
 ↓
OneHotEncoder
 ↓
Type_H
Type_L
Type_M
```

The final processed dataset contains 8 features.

The preprocessing pipeline is saved along with the trained model so that new data receives exactly the same transformations during inference.

---

# 🤖 Machine Learning Models

The following approaches were evaluated:

### 1. Logistic Regression

Used as the initial baseline model.

### 2. Balanced Logistic Regression

Used:

```python
class_weight="balanced"
```

to give more importance to failure observations.

### 3. SMOTE + Logistic Regression

Synthetic Minority Over-sampling Technique (SMOTE) was applied only to the training data.

The test set remained completely untouched.

### 4. Random Forest

A nonlinear ensemble model capable of learning feature interactions and nonlinear relationships.

### 5. Balanced Random Forest

Random Forest with:

```python
class_weight="balanced"
```

### 6. SMOTE + Random Forest

Random Forest trained on the SMOTE-balanced training dataset.

### 7. XGBoost

Gradient-boosted decision trees were used as the main high-performance model.

### 8. Balanced XGBoost

XGBoost was additionally tested using:

```python
scale_pos_weight
```

### 9. Tuned XGBoost

Randomized hyperparameter search was used to optimize the XGBoost model.

---

# 📈 Model Comparison

Results obtained on the held-out test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 96.75% | 63.64% | 10.29% | 17.72% | 89.94% | 42.34% |
| Balanced Logistic Regression | 82.40% | 14.14% | 82.35% | 24.14% | 90.69% | 38.18% |
| SMOTE + Logistic Regression | ~82.4% | ~15.0% | 82.35% | ~25.0% | — | — |
| Random Forest | 98.15% | 89.74% | 51.47% | 65.42% | 97.06% | 79.18% |
| Balanced Random Forest | 98.05% | 89.19% | 48.53% | 62.86% | 96.26% | 77.72% |
| SMOTE + Random Forest | ~96.65% | ~50.5% | 73.53% | ~59.9% | ~97.2% | ~72.8% |
| XGBoost | 98.70% | 90.38% | 69.12% | 78.33% | 97.53% | 85.18% |
| Balanced XGBoost | 97.70% | 62.79% | 79.41% | 70.13% | 96.74% | 81.97% |
| **Tuned XGBoost** | **98.70%** | **90.38%** | **69.12%** | **78.33%** | **97.66%** | **84.00%** |

---

# 🏆 Final Model

The final selected model is **Tuned XGBoost**.

Approximate held-out test performance:

```text
Accuracy  : 98.70%
Precision : 90.38%
Recall    : 69.12%
F1 Score  : 78.33%
ROC-AUC   : 97.66%
PR-AUC    : 84.00%
```

### Confusion Matrix

```text
                    Predicted
                 No Failure  Failure

Actual No Failure    1927       5
Actual Failure         21      47
```

Therefore:

```text
True Negatives  = 1927
False Positives = 5
False Negatives = 21
True Positives  = 47
```

The model detected approximately:

```text
47 / 68 ≈ 69.1%
```

of the actual failures in the held-out test set while producing only 5 false alarms.

---

# 🧪 Handling Class Imbalance

Several imbalance-handling strategies were experimentally compared.

## Class Weighting

Class weighting substantially increased failure recall for Logistic Regression, but precision dropped considerably.

For Random Forest, class weighting did not improve the baseline model.

## SMOTE

SMOTE was applied only to the training data.

The test data was never oversampled.

SMOTE improved failure recall for Random Forest:

```text
51.5% → 73.5%
```

but precision decreased substantially:

```text
89.7% → ~50.5%
```

Therefore, the SMOTE-based Random Forest was not selected as the final model.

## XGBoost Class Weighting

Using `scale_pos_weight` increased recall:

```text
69.1% → 79.4%
```

but reduced precision:

```text
90.4% → 62.8%
```

and also reduced F1 and PR-AUC.

Therefore, standard XGBoost provided the better overall precision-recall balance for this project.

---

# 🌲 Why XGBoost?

XGBoost was selected because the problem is structured/tabular data with nonlinear relationships and feature interactions.

Compared with Logistic Regression, XGBoost can naturally capture relationships between operating variables without requiring manual nonlinear feature construction.

The final XGBoost model substantially outperformed the linear baseline and Random Forest in F1 score and PR-AUC.

---

# 🔍 Model Interpretability

Model interpretability was investigated using:

1. XGBoost feature importance
2. SHAP analysis

## XGBoost Feature Importance

The most influential features included:

1. Torque
2. Rotational speed
3. Tool wear
4. Air temperature
5. Process temperature

Categorical machine-type features had comparatively lower individual importance.

## SHAP Analysis

SHAP was used to investigate how individual features influenced model predictions.

The analysis indicated that:

- Higher torque generally pushed predictions toward failure.
- Higher tool wear generally pushed predictions toward failure.
- Air temperature had a noticeable contribution.
- Lower rotational speed showed a tendency toward higher predicted failure risk in the trained model.
- Process temperature showed a comparatively different contribution pattern.
- Machine type had lower overall impact than the major numerical sensor features.

These results represent **model-learned associations and not causal relationships**.

---

# 💻 Streamlit Application

The trained model is integrated into an interactive Streamlit dashboard.

The user can enter:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The application performs:

```text
User Input
    ↓
Saved Preprocessing Pipeline
    ↓
Tuned XGBoost Model
    ↓
Failure Prediction
    ↓
Failure Probability
```

The dashboard displays:

- Machine failure prediction
- Failure probability
- Risk level
- Machine operating parameters
- Maintenance recommendation
- Model information

---

# 🖥️ Application Workflow

```text
┌──────────────────────────────┐
│       User Input             │
│                              │
│ Machine Type                 │
│ Air Temperature              │
│ Process Temperature          │
│ Rotational Speed             │
│ Torque                       │
│ Tool Wear                    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Preprocessing          │
│                              │
│ StandardScaler               │
│ OneHotEncoder                │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Tuned XGBoost          │
│                              │
│ Binary Classification        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Prediction             │
│                              │
│ Failure / No Failure         │
│ Failure Probability         │
└──────────────────────────────┘
```

---

# 📁 Project Structure

```text
industrial-predictive-maintenance/
│
├── data/
│   └── ai4i2020.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── models/
│   ├── xgb_model.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── __init__.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Tech Stack

### Programming
- Python 3.10

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- XGBoost
- imbalanced-learn
- SMOTE

### Model Interpretation
- SHAP
- XGBoost Feature Importance

### Deployment / Application
- Streamlit
- Pickle

### Development
- Jupyter Notebook
- VS Code
- Git
- GitHub

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/industrial-predictive-maintenance.git

cd industrial-predictive-maintenance
```

## 2. Create a Conda environment

```bash
conda create -n venv python=3.10
```

Activate it:

```bash
conda activate venv
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

From the project root:

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 📓 Run the Notebook

The complete model-development process is documented in:

```text
notebooks/01_eda.ipynb
```

The notebook covers:

```text
Data Loading
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Preprocessing
     ↓
Train-Test Split
     ↓
Baseline Logistic Regression
     ↓
Class Imbalance Experiments
     ↓
Random Forest
     ↓
SMOTE
     ↓
XGBoost
     ↓
Hyperparameter Tuning
     ↓
Feature Importance
     ↓
SHAP Analysis
     ↓
Model Evaluation
```

---

# 📊 Evaluation Strategy

The dataset was divided into training and test sets.

The test set was kept isolated during model development and class-imbalance experiments.

When using SMOTE:

```text
Training data
     ↓
SMOTE
     ↓
Balanced training data
```

while:

```text
Test data
     ↓
UNCHANGED
```

This prevents synthetic samples derived from test information from contaminating the final evaluation.

---

# 🔐 Model Serialization

The final model and preprocessing pipeline are serialized using Python Pickle.

```text
models/
├── xgb_model.pkl
└── preprocessor.pkl
```

During inference, the saved preprocessing pipeline transforms new machine data in exactly the same way as during training.

---

# 🔮 Future Improvements

Potential future improvements include:

### 1. Real-Time Sensor Integration

Connect the system to actual industrial IoT sensors.

```text
Industrial Sensors
       ↓
Streaming Data
       ↓
Prediction API
       ↓
Failure Alert
```

### 2. Time-Series Modeling

The current model treats observations independently.

Future versions could use:

- LSTM
- GRU
- Temporal CNN
- Transformer-based time-series models

to capture temporal patterns in machine behavior.

### 3. Real-Time Monitoring

Build a dashboard that continuously monitors:

- Temperature
- Torque
- Rotational speed
- Tool wear
- Failure probability

### 4. Alert System

Generate alerts when failure probability exceeds a predefined operational threshold.

Possible integrations:

- Email
- SMS
- Slack
- Webhooks

### 5. Model Monitoring

Add production monitoring for:

- Data drift
- Feature drift
- Prediction drift
- Model performance
- Failure-rate changes

### 6. MLOps

Future versions could integrate:

- MLflow
- Docker
- CI/CD
- Cloud deployment
- Model versioning

### 7. Advanced Explainability

Extend SHAP analysis to provide explanations for individual machine predictions.

---

# ⚠️ Limitations

This project is primarily a machine-learning demonstration using a publicly available dataset.

The model's performance on this dataset should not be interpreted as guaranteed performance on real industrial equipment.

Real industrial deployment would require:

- Sensor validation
- Domain-specific failure definitions
- Time-series data
- Real operational costs
- Safety validation
- Continuous monitoring
- Model retraining
- Human maintenance oversight

The predictions represent statistical model outputs and should support, rather than independently replace, engineering and maintenance decisions.

---

# 📌 Key Learnings

This project provided practical experience with:

- End-to-end tabular ML
- Exploratory data analysis
- Feature preprocessing
- Binary classification
- Severe class imbalance
- SMOTE
- Class weighting
- Logistic Regression
- Random Forest
- XGBoost
- Hyperparameter optimization
- Precision-recall tradeoffs
- ROC-AUC and PR-AUC
- Feature importance
- SHAP explainability
- Model serialization
- Streamlit deployment
- Building a reusable inference pipeline

---

# ⭐ Project Highlights

- End-to-end predictive-maintenance ML pipeline
- Compared multiple modeling and imbalance-handling strategies
- Addressed severe class imbalance using class weighting and SMOTE
- Achieved approximately **98.7% accuracy**
- Achieved **90.4% precision** for failure prediction
- Achieved **69.1% recall** for actual failures
- Achieved **78.3% F1 score**
- Achieved approximately **97.7% ROC-AUC**
- Achieved approximately **84.0% PR-AUC**
- Used **XGBoost + SHAP** for model interpretability
- Built an interactive **Streamlit prediction dashboard**
- Implemented reusable inference using serialized model and preprocessing artifacts

---

# 👨‍💻 Author

**Krishna Srivastava**

Electrical Engineering Student  
NIT Hamirpur

GitHub: https://github.com/Arvik07  
LinkedIn: https://www.linkedin.com/in/krishna-srivastava-/
