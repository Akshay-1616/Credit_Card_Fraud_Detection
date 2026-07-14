# 🛡️ Credit Card Fraud Detection

This project applies **machine learning models** to detect fraudulent credit card transactions using the Kaggle dataset.  
It demonstrates end‑to‑end workflow: preprocessing, oversampling, training, evaluation, visualization, and performance tracking.

---

## 📂 Project Structure

Credit_Card_Fraud_Detection/
│
├── src/
│   ├── preprocess.py          # Data cleaning, SMOTE oversampling, exploratory plots
│   ├── train_test_model.py    # Model training, evaluation, metrics logging, trend plots
│
├── results/                   # Saved models (.pkl) and scaler
├── plots/                     # ROC curves, PR curves, confusion matrices, feature importance, distributions
├── creditcard.csv             # Dataset (Kaggle Credit Card Fraud Detection)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation



---

## ⚙️ Features
### 🔹 Preprocessing
- Drop NaN values in target column (`Class`)
- Convert `Class` to integer
- Scale features with `StandardScaler`
- Handle imbalance with **SMOTE**
- Save cleaned dataset (`cleaned_resampled.csv`)
- Generate exploratory plots:
  - Class distribution (bar + pie)
  - Transaction amount distribution (histogram + boxplot)
  - Fraud vs Non‑Fraud transaction time distribution
  - Correlation heatmap

### 🔹 Models
- **Logistic Regression**
- **Random Forest** (with feature importance)
- **XGBoost** (with `scale_pos_weight`)
- **RandomizedSearchCV** with StratifiedKFold for hyperparameter tuning

### 🔹 Evaluation
- Metrics: Accuracy, Precision, Recall, F1, AUC
- ROC curves and Precision‑Recall curves
- Confusion matrices (heatmaps)
- Feature importance plots
- Trend plots over time (Accuracy, F1, Precision, Recall, AUC)

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Credit_Card_Fraud_Detection.git
   cd Credit_Card_Fraud_Detection


2.Install dependencies:
--pip install -r requirements.txt

3.Run preprocessing:
--python src/preprocess.py
    -Cleans dataset
    -Applies SMOTE
    -Generates exploratory plots
    -Saves cleaned_resampled.csv

4.Train and evaluate models:
--python src/train_test_model.py
    -Trains Logistic Regression, Random Forest, XGBoost
    -Performs hyperparameter tuning
    -Saves models in results/
    -Generates evaluation plots in plots/
    -Logs metrics to results.csv


## 📊 Output Plots

### 📌 Preprocessing & Data Exploration
- [Class Distribution Bar Chart](plots/Class_Distribution_Barchart.png)
- [Class Distribution Pie Chart](plots/Class_Distribution_PieChart.png)
- [Transaction Amount Histogram](plots/Transaction_Amount_Distribution_Histogram.png)
- [Transaction Amount Boxplot](plots/Transaction_Amount_by_Class_Boxplot.png)
- [Transaction Time Distribution](plots/Transaction_Time_Distribution.png)
- [Correlation Heatmap](plots/Correlation_Features_Heatmap.png)

### 📌 Model Evaluation
- ROC Curves:
  - [Logistic Regression ROC](plots/Logistic_Regresion_ROC_Curve.png)
  - [Random Forest ROC](plots/Random_Forest_ROC_Curve.png)
  - [XGBoost ROC](plots/XGBoost_ROC_Curve.png)
- Precision‑Recall Curves:
  - [Logistic Regression PR](plots/PR_LogisticRegression.png)
  - [Random Forest PR](plots/PR_RandomForest.png)
  - [XGBoost PR](plots/PR_XGBoost.png)
- Confusion Matrices:
  - [Logistic Regression Confusion Matrix](plots/LR_ConfusionMatrix_heatmap.png)
  - [Random Forest Confusion Matrix](plots/RF_ConfusionMatrix_heatmap.png)
  - [XGBoost Confusion Matrix](plots/XG_ConfusionMatrix_heatmap.png)
  - [Tuned Random Forest Confusion Matrix](plots/GS_ConfusionMatrix_heatmap.png)
- Feature Importance:
  - [Random Forest Feature Importance](plots/feature_importance_RF.png)
  - [XGBoost Feature Importance](plots/feature_importance_XG.png)

### 📌 Trends & Comparisons
- [Model Comparison Bar Chart](plots/model_comparision.png)
- [Accuracy Trends](plots/Accuracy_trends_overtime.png)
- [F1 Score Trends](plots/F1Score_trends_overtime.png)
- [Precision Trends](plots/Precision_trends_overtime.png)
- [Recall Trends](plots/Recall_trends_overtime.png)
- [AUC Trends](plots/AUC_trends_overtime.png)

### 📌 Time Analysis
- [Fraud vs Non‑Fraud by Hour](plots/Fraud_vs_NonFraud_ByHour.png)
- [Transaction Time Density](plots/Transaction_Time_Density.png)

## 📊 Sample Outputs

### ROC Curve (Logistic Regression)
![ROC Curve - Logistic Regression](plots/Logistic_Regresion_ROC_Curve.png)

### Confusion Matrix (Random Forest)
![Confusion Matrix - Random Forest](plots/RF_ConfusionMatrix_heatmap.png)

### Feature Importance (XGBoost)
![Feature Importance - XGBoost](plots/feature_importance_XG.png)


📦 Dataset
The dataset used is the Kaggle Credit Card Fraud Detection dataset (kaggle.com in Bing)(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
It contains anonymized features (V1–V28), Time, Amount, and Class (fraud or not fraud).

🛠 Requirements
See requirements.txt for dependencies.

📌 Usage Example
Load a saved model and make predictions:

python
import joblib
import pandas as pd

# Load scaler and model
scaler = joblib.load("results/scaler.pkl")
model = joblib.load("results/Random_Forest.pkl")

# Load new transaction data
new_data = pd.DataFrame([[0.1, -1.2, 0.3, ...]], columns=[...])  # same feature order

# Scale features
new_data_scaled = scaler.transform(new_data)

# Predict fraud probability
prob = model.predict_proba(new_data_scaled)[:,1]
print("Fraud probability:", prob[0])

📈 Future Improvements
    -Add deep learning models (e.g., LSTMs, Autoencoders)
    -Deploy as a REST API or web app
    -Integrate real‑time fraud detection pipeline
    -Experiment with anomaly detection methods

👨‍💻 Author
Developed by [Akshay kumar]  (https://github.com/Akshay-1616)
Feel free to fork, contribute, or open issues to improve the project.


