import pandas as pd
import requests
import random

# Load dataset
df = pd.read_csv("creditcard.csv", low_memory=False)



# Only keep features the model expects (drop target column)
X = df.drop("Class", axis=1)
y = df["Class"]

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Sample 100 random rows for testing
sample = df.sample(100, random_state=42)

correct = 0
total = len(sample)

for _, row in sample.iterrows():
    # Build JSON payload (all features except Class)
    transaction = row.drop("Class").to_dict()

    # Send to FastAPI
    response = requests.post(url, json=transaction)
    pred = response.json()["prediction"]

    # Compare with ground truth
    if pred == row["Class"]:
        correct += 1

print(f"Tested {total} rows")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {correct/total:.2f}")
