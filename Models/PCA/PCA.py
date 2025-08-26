import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


## get data
cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df.info())
print(df.shape)
df.head()

## EDA
X = df
labels = pd.DataFrame({'label': cancer['target']})
y = labels.label
x = StandardScaler().fit_transform(X)
X = pd.DataFrame(x, columns=X.columns)
X.head()


## PCA
pca = PCA(n_components=2)
X_pca = pca.fit(X)
print('Explained variability per principal component: {}'.format(X_pca.explained_variance_ratio_))
df_pca = pd.DataFrame(X_pca.transform(X), columns=['PC1', 'PC2'])
df_pca.head()


## plot graph
colors = ['r', 'g']
plt.figure()
labels = ['benign', 'malignant']
for i, label in enumerate(labels):
    plt.scatter(df_pca.loc[y.label == i, 'PC1'], df_pca.loc[y.label == i, 'PC2'], c=colors[i], label=label)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


## fit model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(df_pca, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Original Dataset')
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

print('\nPCA Dataset')
model_pca = LogisticRegression()
model_pca.fit(X_pca_train, y_pca_train)
y_pca_pred = model_pca.predict(X_pca_test)
print(confusion_matrix(y_pca_test, y_pca_pred))
print(accuracy_score(y_pca_test, y_pca_pred))
