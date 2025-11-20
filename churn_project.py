import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv(r"C:\Users\shara\OneDrive\Desktop\week2-churn-project\telco.csv")


data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})


data = data.dropna()

X = data.select_dtypes(include=['int64', 'float64'])
y = data['Churn']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
