import preprocess as pre
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
import seaborn as sns
from datetime import datetime
from sklearn.metrics import precision_recall_curve, average_precision_score

# split data into traning and testing
X_train, X_test, y_train, y_test = train_test_split(pre.X, pre.y, test_size=0.2, random_state=42)

# Scaling all the numerical values
Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)

# training model using Logistic regresion
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# testing the model
y_pred = model.predict(X_test)

# Evaluating the model
print("Classification Report for Logistic regression")
print(classification_report(y_test,y_pred))
report = classification_report(y_test, y_pred, output_dict=True)
lr_precision = report["weighted avg"]["precision"]
lr_recall = report["weighted avg"]["recall"]
lr_f1 = report["weighted avg"]["f1-score"]
print("F1Score: ",lr_f1)
print("Recall: ",lr_recall)
print("Precision: ",lr_precision)
print("Confusion Metrix")
conf_matric1 = confusion_matrix(y_test,  y_pred)
print(conf_matric1)

print("Accuracy Score")
lr_acc = accuracy_score(y_test,  y_pred)
print(lr_acc)

# training the model again with Random forest
model2 = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight="balanced")
model2.fit(X_train,y_train)

# testing model
y_pred2 = model2.predict(X_test)

#Evaluating model
print("Classification Report for randon forest")
print(classification_report(y_test,y_pred2))
report = classification_report(y_test, y_pred2, output_dict=True)
rf_precision = report["weighted avg"]["precision"]
rf_recall = report["weighted avg"]["recall"]
rf_f1 = report["weighted avg"]["f1-score"]
print("F1Score: ",rf_f1)
print("Recall: ",rf_recall)
print("Precision: ",rf_precision)


print("Confusion Metrix")
conf_matric2 = confusion_matrix(y_test,  y_pred2)
print(conf_matric2)

print("Accuracy Score")
rf_acc = accuracy_score(y_test,  y_pred2)
print(rf_acc)

# finding important columns
imp_columns = model2.feature_importances_
feature_imp = pd.DataFrame(imp_columns, index=pre.df.drop("Class", axis=1).columns, columns=["Important"]).sort_values(by="Important", ascending=False)
print("Important Features")
print(feature_imp[:6])

# training the model with XGBoost
model3 = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=578,  # adjust based on your dataset
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
# Training model
model3.fit(X_train,y_train)
# testing model
y_pred3 = model3.predict(X_test)

# Evaluating model
print("Classification Report Xgboost")
print(classification_report(y_test,y_pred3))
report = classification_report(y_test, y_pred3, output_dict=True)
xg_precision = report["weighted avg"]["precision"]
xg_recall = report["weighted avg"]["recall"]
xg_f1 = report["weighted avg"]["f1-score"]
print("F1Score: ",xg_f1)
print("Recall: ",xg_recall)
print("Precision: ",xg_precision)


print("Confusion Metrix")
conf_matric3 = confusion_matrix(y_test,  y_pred3)
print(conf_matric3)

print("Accuracy Score")
xg_acc = accuracy_score(y_test,  y_pred3)
print(xg_acc)
# Finding important features
imp_columns2 = model3.feature_importances_
feature_imp2 = pd.DataFrame(imp_columns2, index=pre.df.drop("Class", axis=1).columns, columns=["Important"]).sort_values(by="Important", ascending=False)
print("Important Features")
print(feature_imp2[:6])

# Dictionary for tuning using Random Forest
param_dist = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}


# StratifiedKfold reduce the split
cv_strategy = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Tuning the model
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_distributions=param_dist,
    n_iter=5,
    cv=cv_strategy,
    n_jobs=-1,
    verbose=2
)

# training model
random_search.fit(X_train, y_train)
print("Best Parameters:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)
best_model = random_search.best_estimator_

# testing model
y_pred_tuned = best_model.predict(X_test)

# evaluating model
print("Classification Report")
print(classification_report(y_test,y_pred_tuned))
report = classification_report(y_test, y_pred_tuned, output_dict=True)
rs_precision = report["weighted avg"]["precision"]
rs_recall = report["weighted avg"]["recall"]
tuned_f1 = report["weighted avg"]["f1-score"]
print("F1Score: ",tuned_f1)
print("Recall: ",rs_recall)
print("Precision: ",rs_precision)

print("Confusion Metrix")
conf_matric4 = confusion_matrix(y_test,  y_pred_tuned)
print(conf_matric4)

print("Accuracy Score")
tuned_acc = accuracy_score(y_test,  y_pred_tuned)
print(tuned_acc)

# storing the dataa using joblib
joblib.dump(model, "Credit_Card_Fraud_Detection/results/Logistic_Regression.pkl")
joblib.dump(model2, "Credit_Card_Fraud_Detection/results/Random_Forest.pkl")
joblib.dump(model3, "Credit_Card_Fraud_Detection/results/XGboost.pkl")
joblib.dump(Scaler, "Credit_Card_Fraud_Detection/results/scaler.pkl")

# calculating prbability of models
y_prob1 = model.predict_proba(X_test)[:,1]  # probability of class 1 (fraud)
y_prob2 = model2.predict_proba(X_test)[:,1]
y_prob3 = model3.predict_proba(X_test)[:,1]
y_prob_tuned = best_model.predict_proba(X_test)[:,1]

# Roc curve
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_prob1)
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_prob2)
fpr3, tpr3, thresholds3 = roc_curve(y_test, y_prob3)

# calculating AUC score for models
auc_score1 = roc_auc_score(y_test, y_prob1)
print("AUC1:", auc_score1)
auc_score2 = roc_auc_score(y_test, y_prob2)
print("AUC2:", auc_score2)
auc_score3 = roc_auc_score(y_test, y_prob3)
print("AUC3:", auc_score3)
best_auc = roc_auc_score(y_test,y_prob_tuned)

# adding timestamp to knoe when it is done
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# checks if the path exists
os.makedirs("Credit_Card_Fraud_Detection/plots", exist_ok=True)


plt.figure(figsize=(18,16))
# ROC for logistic Regression
plt.subplot(2,2,1)
plt.plot(fpr1, tpr1, color="blue", label=f"ROC curve (AUC = {auc_score1:.2f})")
plt.plot([0,1], [0,1], color="red", linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) for Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("Credit_Card_Fraud_Detection/plots/Logistic_Regresion_ROC_Curve.png", dpi = 300)

# ROC for Random Forest
plt.subplot(2,2,2)
plt.plot(fpr2, tpr2, color="blue", label=f"ROC curve (AUC = {auc_score2:.2f})")
plt.plot([0,1], [0,1], color="red", linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) for Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("Credit_Card_Fraud_Detection/plots/Random_Forest_ROC_Curve.png", dpi = 300)

# ROC for XGBoost
plt.subplot(2,2,3)
plt.plot(fpr3, tpr3, color="blue", label=f"ROC curve (AUC = {auc_score3:.2f})")
plt.plot([0,1], [0,1], color="red", linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) for XGBoost")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("Credit_Card_Fraud_Detection/plots/XGBoost_ROC_Curve.png", dpi = 300)
plt.tight_layout()
plt.show()

# calculating precision recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob1)  # RF example
ap_score1 = average_precision_score(y_test, y_prob1)

precision2, recall2, thresholds2 = precision_recall_curve(y_test, y_prob2)  # RF example
ap_score2 = average_precision_score(y_test, y_prob2)

precision3, recall3, thresholds3 = precision_recall_curve(y_test, y_prob3)  # RF example
ap_score3 = average_precision_score(y_test, y_prob3)

# precision recall curve for Logistic regression
plt.figure(figsize=(18,16))
plt.subplot(2,2,1)
plt.plot(recall, precision, color="blue", label=f"PR curve (AP = {ap_score1:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Logistic Regression")
plt.legend()
plt.grid(True)

# precision recall curve for random forest
plt.subplot(2,2,2)
plt.plot(recall2, precision2, color="blue", label=f"PR curve (AP = {ap_score2:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Random Forest")
plt.legend()
plt.grid(True)

# precision recall curve for XGBoost
plt.subplot(2,2,3)
plt.plot(recall3, precision3, color="blue", label=f"PR curve (AP = {ap_score3:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve XGBoost")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# confusion matrix for models
plt.figure(figsize=(18,16))

plt.subplot(2,2,1)
sns.heatmap(conf_matric1, annot=True, cmap="Blues", fmt="d", linecolor="black",
            xticklabels=["Fraud[1]", "Not Fraud[0]"], yticklabels=["Fraud[1]", "Not Fraud[0]"])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Credit_Card_Fraud_Detection/plots/LR_ConfusionMatrix_heatmap.png", dpi=300)

plt.subplot(2,2,2)
sns.heatmap(conf_matric2, annot=True, cmap="Blues", fmt="d", linecolor="black",
            xticklabels=["Fraud[1]", "Not Fraud[0]"], yticklabels=["Fraud[1]", "Not Fraud[0]"])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Credit_Card_Fraud_Detection/plots/RF_ConfusionMatrix_heatmap.png", dpi=300)

plt.subplot(2,2,3)
sns.heatmap(conf_matric3, annot=True, cmap="Blues", fmt="d", linecolor="black",
            xticklabels=["Fraud[1]", "Not Fraud[0]"], yticklabels=["Fraud[1]", "Not Fraud[0]"])
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Credit_Card_Fraud_Detection/plots/XG_ConfusionMatrix_heatmap.png", dpi=300)

plt.subplot(2,2,4)
sns.heatmap(conf_matric4, annot=True, cmap="Blues", fmt="d", linecolor="black",
            xticklabels=["Fraud[1]", "Not Fraud[0]"], yticklabels=["Fraud[1]", "Not Fraud[0]"])
plt.title("Grid Search CV Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Credit_Card_Fraud_Detection/plots/GS_ConfusionMatrix_heatmap.png", dpi=300)
plt.tight_layout()
plt.show()

plt.figure(figsize=(18,16))
# important features using random forest
plt.subplot(2,1,1)
sns.barplot(x=feature_imp["Important"][:8], y=feature_imp.index[:8], hue=feature_imp.index[:8], palette="viridis")
plt.title("Top 8 Important Features by Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.savefig("Credit_Card_Fraud_Detection/plots/feature_importance_RF.png", dpi=300)


# important features using XGBoost
plt.subplot(2,1,2)
sns.barplot(x=feature_imp2["Important"][:8], y=feature_imp2.index[:8], hue=feature_imp2.index[:8], palette="viridis")
plt.title("Top 8 Important Features by XGBoost")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.savefig("Credit_Card_Fraud_Detection/plots/feature_importance_XG.png", dpi=300)
plt.tight_layout()
plt.show()
# dictionary for model comparision
summary = {
    "timestamp": [timestamp, timestamp, timestamp, timestamp],
    "model": ["Logistic_Regression", "Random_Forest", "XGBoost", "Tuned_RandomForest"],
    "accuracy": [lr_acc, rf_acc, xg_acc, tuned_acc],
    "F1Score": [ lr_f1, rf_f1, xg_f1, tuned_f1],
    "precision": [lr_precision, rf_precision, xg_precision, rs_precision],
    "recall": [lr_recall, rf_recall, xg_recall, rs_recall],
    "AUC": [auc_score1, auc_score2, auc_score3, best_auc]
}
df = pd.DataFrame(summary)

# melting data to plot the graphs
df_melted = df.melt(id_vars=["timestamp", "model"], value_vars=["accuracy","F1Score","precision","recall", "AUC"],
                    var_name="metric", value_name="score")

results_path = r"C:\Users\akhia\OneDrive\Documents\bank_mngmt\New folder\ML_projects\Credit_Card_Fraud_Detection\results.csv"


# saving data to csv file
df.to_csv(
    results_path,
    mode="a",
    header= not os.path.exists(results_path) or os.path.getsize(results_path) == 0,
    index=False
)
results = pd.read_csv(results_path)

# adding name timestamp to column 1
expected_cols = ["timestamp","model","accuracy","F1Score","precision","recall","AUC"]
if list(results.columns) != expected_cols:
    results.columns = expected_cols


#  barplt that shows the comparision between the Models
plt.figure(figsize=(10,6))
sns.barplot(data=df_melted, x="model", y="score", hue="metric", palette="viridis")
plt.xlabel("Model Names")
plt.ylabel("model Scores")
plt.title("Logistic Regression vs Random Forest vs XGBoost vs  Tuned Random Forest Comparision")
plt.grid(visible=True)
plt.savefig("Credit_Card_Fraud_Detection/plots/model_comparision.png", format = "png", dpi = 300, bbox_inches = "tight")
plt.show()


# Plot accuracy trends over time
plt.figure(figsize=(18,16))
plt.subplot(2,3,1)
sns.lineplot(data=results, x="timestamp", y="accuracy", hue="model", markers="o")
plt.title("Accuracy Trends Over Time")
plt.xlabel("Run Timestamp")
plt.ylabel("Accuracy")
plt.savefig("Credit_Card_Fraud_Detection/plots/Accuracy_trends_overtime.png", format = "png", dpi = 300, bbox_inches = "tight")


# F1 trend
plt.subplot(2,3,2)
sns.lineplot(data=results, x="timestamp", y="F1Score", hue="model", markers="o")
plt.title("F1Score Trends Over Time")
plt.xlabel("Run Timestamp")
plt.ylabel("F1Score")
plt.savefig("Credit_Card_Fraud_Detection/plots/F1Score_trends_overtime.png", format = "png", dpi = 300, bbox_inches = "tight")


# Precision
plt.subplot(2,3,3)
sns.lineplot(data=results, x="timestamp", y="precision", hue="model", markers="o")
plt.title("Precision Trends Over Time")
plt.xlabel("Run Timestamp")
plt.ylabel("Precision")
plt.savefig("Credit_Card_Fraud_Detection/plots/Precision_trends_overtime.png", format = "png", dpi = 300, bbox_inches = "tight")


# recall
plt.subplot(2,3,4)
sns.lineplot(data=results, x="timestamp", y="recall", hue="model", markers="o")
plt.title("Recall Trends Over Time")
plt.xlabel("Run Timestamp")
plt.ylabel("Recall")
plt.savefig("Credit_Card_Fraud_Detection/plots/Recall_trends_overtime.png", format = "png", dpi = 300, bbox_inches = "tight")


# AUC
plt.subplot(2,3,5)
sns.lineplot(data=results, x="timestamp", y="AUC", hue="model", markers="o")
plt.title("AUC Trends Over Time")
plt.xlabel("Run Timestamp")
plt.ylabel("AUC")
plt.savefig("Credit_Card_Fraud_Detection/plots/AUC_trends_overtime.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()


plt.figure(figsize=(18,16))
plt.subplot(1,2,1)
plt.hist(pre.fraud_times, bins=24, alpha=0.7, color="salmon", label="Fraud")
plt.hist(pre.nonfraud_times, bins=24, alpha=0.7, color="skyblue", label="Non-Fraud")
plt.xlabel("Hour of Day")
plt.ylabel("Transaction Count")
plt.title("Fraud vs Non-Fraud Transactions by Hour")
plt.legend()
plt.savefig("Credit_Card_Fraud_Detection/plots/Fraud_vs_NonFraud_ByHour.png", dpi=300)


plt.subplot(1,2,2)
sns.kdeplot(pre.fraud_times, color="salmon", label="Fraud", fill=True)
sns.kdeplot(pre.nonfraud_times, color="skyblue", label="Non-Fraud", fill=True)
plt.xlabel("Time (Hours)")
plt.ylabel("Density")
plt.title("Transaction Time Density: Fraud vs Non-Fraud")
plt.legend()
plt.savefig("Credit_Card_Fraud_Detection/plots/Transaction_Time_Density.png", dpi=300)
plt.tight_layout()
plt.show()
