import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1) Load data
data = pd.read_csv(r"C:\Users\shara\OneDrive\Desktop\week2-churn-project\telco.csv")

# 2) Basic preprocessing
# Convert 'Churn' column to binary (Yes=1, No=0)
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Remove missing values (simple approach)
data = data.dropna()

# 3) Select features (X) and target (y)
# Use only numeric columns for simplicity
X = data.select_dtypes(include=['int64', 'float64'])
y = data['Churn']

# 4) Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6) Predict
y_pred = model.predict(X_test)

# 7) Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
