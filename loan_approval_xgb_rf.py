
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('dataset/loan_approval_dataset.csv')



data.columns = data.columns.str.strip()

data['education'] = data['education'].str.strip()
data['self_employed'] = data['self_employed'].str.strip()
data['loan_status'] = data['loan_status'].str.strip()

data = pd.get_dummies(data, columns=['education', 'self_employed'], drop_first=False)

data['loan_status'] = data['loan_status'].replace({'Approved': 1, 'Rejected': 0})

X = data.drop('loan_status', axis=1)
y = data['loan_status']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

model_xgb = XGBClassifier(n_estimators=100, learning_rate=1.1, random_state=42)
model_xgb.fit(X_train, y_train)

predictions_rf = model_rf.predict(X_val)

accuracy_score_rf = accuracy_score(predictions_rf, y_val)
confusion_matrix_rf = confusion_matrix(predictions_rf, y_val)
classification_report_rf = classification_report(predictions_rf, y_val)

print("Random Forest: ")
print(f"Accuracy Score: {accuracy_score_rf}")
print(f"Confusion Matrix: {confusion_matrix_rf}")
print(f"Classification Report: {classification_report_rf}")

predictions_xgb = model_xgb.predict(X_val)

accuracy_score_xbg = accuracy_score(predictions_xgb, y_val)
confusion_matrix_xgb = confusion_matrix(predictions_xgb, y_val)
classification_report_xgb = classification_report(predictions_xgb, y_val)

print("XGBoost: ")
print(f"Accuracy Score: {accuracy_score_xbg}")
print(f"Confusion Matrix: {confusion_matrix_xgb}")
print(f"Classification Report: {classification_report_xgb}")

plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Approved', 'Rejected'], yticklabels=['Approved', 'Rejected'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Random Forest Confusion Matrix')


plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Approved', 'Rejected'], yticklabels=['Approved', 'Rejected'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('XGBoost Confusion Matrix')

plt.show()