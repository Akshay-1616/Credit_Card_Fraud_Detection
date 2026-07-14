import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os

save_path = r"C:\Users\akhia\OneDrive\Documents\bank_mngmt\New folder\ML_projects\Credit_Card_Fraud_Detection\creditcard.csv"

# load data from csv file
data = pd.read_csv(r"C:\Users\akhia\OneDrive\Documents\bank_mngmt\New folder\ML_projects\Credit_Card_Fraud_Detection\creditcard.csv"
, encoding="utf-8", low_memory=False)

# drops nan values
data = data.dropna(subset=["Class"])
data["Class"] = data["Class"].astype(int)
df = pd.DataFrame(data)

# target variables x and y
X = df.iloc[:,:-1]
y = df["Class"]

# oversampling using smote
oversampling = SMOTE()
X_resampled, y_resampled = oversampling.fit_resample(X, y)

# Combine oversampled features and labels
df_resampled = pd.DataFrame(X_resampled, columns=df.drop("Class", axis=1).columns)
df_resampled["Class"] = y_resampled

# Ensure Time column is numeric
df_resampled["Time"] = pd.to_numeric(df_resampled["Time"], errors="coerce")

# Drop rows where Time could not be converted
df_resampled = df_resampled.dropna(subset=["Time"])

# save data to csv file
df_resampled.to_csv("Credit_Card_Fraud_Detection/cleaned_resampled.csv", index=False)



class_counts = df_resampled["Class"].value_counts()

os.makedirs("Credit_Card_Fraud_Detection/plots", exist_ok=True)

# Class distribution using bar chart
plt.figure(figsize=(8,6))
plt.bar(class_counts.index, class_counts.values, color=['skyblue','salmon'])
plt.xticks(class_counts.index, ['Non-Fraud (0)', 'Fraud (1)'])
plt.ylabel("Number of Samples")
plt.title("Class Distribution After Oversampling Using Bar Chart")
plt.savefig("Credit_Card_Fraud_Detection/plots/Class_Distribution_Barchart.png", dpi = 300)
plt.show()

#Class distribution using pie chart
plt.figure(figsize=(8,6))
plt.pie(class_counts.values, autopct="%1.2f%%", labels=['Non-Fraud (0)', 'Fraud (1)'])
plt.title("Class Distribution After Oversampling using Pie Chart")
plt.legend()
plt.savefig("Credit_Card_Fraud_Detection/plots/Class_Distribution_PieChart.png", dpi = 300)
plt.show()

# histogram for distribution of transaction amounts
plt.figure(figsize=(8,6))
plt.hist(df_resampled["Amount"], bins=50, color="skyblue", edgecolor="black")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Transaction Amounts")
plt.savefig("Credit_Card_Fraud_Detection/plots/Transaction_Amount_Distribution_Histogram.png", dpi = 300)
plt.show()

# amount by class boxplot
plt.figure(figsize=(8,6))
sns.boxplot(x="Class", y="Amount", data=df_resampled, palette=["skyblue","salmon"], hue="Class")
plt.xticks([0,1], ["Non-Fraud (0)", "Fraud (1)"])
plt.ylabel("Transaction Amount")
plt.title("Transaction Amounts by Class")
plt.savefig("Credit_Card_Fraud_Detection/plots/Transaction_Amount_by_Class_Boxplot.png", dpi = 300)
plt.show()


# Separate fraud and non-fraud transactions
fraud_times = df_resampled[df_resampled["Class"] == 1]["Time"] / 3600   # convert seconds to hours
nonfraud_times = df_resampled[df_resampled["Class"] == 0]["Time"] / 3600

plt.figure(figsize=(10,6))

# Plot histograms
plt.hist(nonfraud_times, bins=50, alpha=0.6, color="skyblue", label="Non-Fraud")
plt.hist(fraud_times, bins=50, alpha=0.6, color="salmon", label="Fraud")
plt.xlabel("Time (Hours)")
plt.ylabel("Frequency")
plt.title("Transaction Time Distribution: Fraud vs Non-Fraud")
plt.legend()
plt.savefig("Credit_Card_Fraud_Detection/plots/Transaction_Time_Distribution.png", dpi = 300)
plt.show()

# corelation heatmap
plt.figure(figsize=(12,10))
corr = df_resampled.corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Features")
plt.savefig("Credit_Card_Fraud_Detection/plots/Correlation_Features_Heatmap.png", dpi = 300)
plt.show()


