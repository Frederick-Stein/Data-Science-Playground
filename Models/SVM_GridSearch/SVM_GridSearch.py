from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns


## get data
cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
print(df.shape)
df.head()


df.isna().sum()
cancer_data.target_names

## prepare data
X = df
y = cancer_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## train the model
model0 = SVC(kernel='linear', C=1) # higher C -> harder margin (fewer misclassifications)
model0.fit(X_train, y_train)


## test results
y_pred = model0.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .4f}")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


## different kernel and C
# polynomial kernel
model1 = SVC(kernel='poly', C=1) # higher C -> harder margin (fewer misclassifications)
model1.fit(X_train, y_train)
## test results
y_pred = model1.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .4f}")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# different C
model2 = SVC(kernel='poly', C=3) # higher C -> harder margin (fewer misclassifications)
model2.fit(X_train, y_train)
## test results
y_pred = model2.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .4f}")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


## GridSearch
param_grid = {'kernel':['rbf'],'C':[0.1,1,10,100], 'gamma':[1,0.1, 0.01,0.001,0.0001]}
grid_model = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid_model.fit(X_train, y_train)

grid_model.best_params_

grid_model.best_estimator_

## best model in GridSearchCV
y_pred = grid_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .4f}")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
