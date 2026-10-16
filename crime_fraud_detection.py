import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=== AI Crime & Fraud Detection System ===")
np.random.seed(42)
n_samples = 1000
data = {
    'transaction_amount': np.random.randint(10, 10000, n_samples),
    'time_of_day': np.random.randint(0, 24, n_samples),
    'location_risk_score': np.random.randint(1, 10, n_samples),
    'previous_fraud_attempts': np.random.randint(0, 5, n_samples),
    'device_change': np.random.randint(0, 2, n_samples),
    'is_fraud': np.random.randint(0, 2, n_samples)
}
df = pd.DataFrame(data)
X = df[['transaction_amount', 'time_of_day', 'location_risk_score', 'previous_fraud_attempts', 'device_change']]
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Trained! Accuracy: {accuracy*100:.2f}%")
print(classification_report(y_test, y_pred))
new_case = [[8500, 2, 9, 3, 1]]
pred = model.predict(new_case)
prob = model.predict_proba(new_case)
if pred[0] == 1:
    print(f"ALERT: Fraud Detected! Risk: {prob[0][1]*100:.1f}%")
else:
    print(f"Low Risk: {prob[0][0]*100:.1f}%")
