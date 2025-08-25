import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/Frederick-Stein/Data-Science-Playground/refs/heads/main/Models/Random_Forest/loan_approval_dataset.csv"
df = pd.read_csv(url)

# EDA
print(df.shape)

df.head()

df.info()

df.columns

# features and labels
X = df.drop(columns=['loan_id', ' loan_status'], axis = 1)
y = df[' loan_status']
X.info()

# encode categorical variables
X[' education'] = X[' education'].map({' Not Graduate': 0, ' Graduate': 1})
X[' self_employed'] = X[' self_employed'].map({' No': 0, ' Yes': 1})
X.head()
y = y.map({' Rejected': 0, ' Approved': 1})
y.head()

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .4f}")
print(classification_report(y_test, y_pred))

# feature importance
feature_scores = pd.Series(rf_classifier.feature_importances_, index=X.columns)
feature_scores = feature_scores.sort_values(ascending=False)
print(feature_scores)

# visualize
sns.barplot(x=feature_scores, y=feature_scores.index)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.show()

# drop least importance features
X = df.drop(['loan_id', ' loan_status', ' education'], axis = 1)
X[' self_employed'] = X[' self_employed'].map({' No': 0, ' Yes': 1})
y = df[' loan_status']
y = y.map({' Rejected': 0, ' Approved': 1})
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .4f}")
print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
