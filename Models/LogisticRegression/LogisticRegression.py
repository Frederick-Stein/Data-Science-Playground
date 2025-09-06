import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random
import pandas as pd

## load iris data
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.shape)
print(data.target_names)
df.head()


## prepare data
random.seed(42)
X = data.data
y = data.target
indices = list(range(len(X)))
random.shuffle(indices)
X = X[indices]
y = y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## fit the model
model = LogisticRegression(max_iter = 500)
model.fit(X_train, y_train)


## evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
print(f'Accuracy: {model.score(X_test, y_test): .4f}')
for i in range(len(y_pred)):
    print(f'{data.target_names[y_pred[i]]} {np.max(y_prob[i]):.2f}              |True label: {data.target_names[y_test[i]]}')
