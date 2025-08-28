import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import TextVectorization

from collections import Counter


seed = 42
## get data
url = 'https://raw.githubusercontent.com/Frederick-Stein/Data-Science-Playground/refs/heads/main/Projects/Spam_Detection/spam_ham_dataset.csv'
data = pd.read_csv(url)
print(data.shape)
data.head()


## EDA
sns.countplot(data['label'])
plt.show()
data.info()


# balance (undersampling) (optional)
data_ham = data[data['label'] == 'ham']
data_spam = data[data['label'] == 'spam']

data_ham_undersampled = data_ham.sample(n=len(data_spam), random_state=seed)
data_balanced = pd.concat([data_spam, data_ham_undersampled]).reset_index(drop=True)

sns.countplot(data_balanced['label'])
plt.show()


## base model
# unbalanced data
X = data['text']
y = data['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# encoding
vec = TfidfVectorizer(stop_words = 'english', lowercase = True)
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# fit baseline model
baseline_model = MultinomialNB()
baseline_model.fit(X_train,y_train)
print(f"Accuracy of baseline model with unbalanced data: {accuracy_score(y_test,baseline_model.predict(X_test)): .4f}")
print(classification_report(y_test,baseline_model.predict(X_test)))


# balanced data
X = data_balanced['text']
y = data_balanced['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# emcoding
vec = TfidfVectorizer(stop_words = 'english', lowercase = True)
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# fit baseline model
baseline_model = MultinomialNB()
result = baseline_model.fit(X_train,y_train)
print(f"Accuracy of baseline model with balanced data: {accuracy_score(y_test,baseline_model.predict(X_test)): .4f}")
print(classification_report(y_test,baseline_model.predict(X_test)))


## deep learning with LSTM using tensorflow

# get the length of text
length = [len(x.split()) for x in data['text']]
output_length = round(np.percentile(length, 80))
print("80th percentile of lengths:", output_length)
# get number of tokens
s = set()
for text in data['text']:
  for word in text.split():
    s.add(word.lower())
num_tokens=len(s)
print("Number of tokens:", num_tokens)
# get tokens that appear more than once
tokens = " ".join(data['text']).lower().split()
freqs = Counter(tokens)
freqs_1 = {k:v for k,v in freqs.items() if v > 1}
num_tokens = len(freqs_1)
print("Number of tokens that appear more than once:", num_tokens)


# balanced data
X = data_balanced['text']
y = data_balanced['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# encoding using TextVectorization
text_vec = TextVectorization(max_tokens=num_tokens, 
                        standardize="lower", 
                        output_mode='int',
                        output_sequence_length=300
                        )
text_vec.adapt(X_train)
X_train = text_vec(X_train)
X_test = text_vec(X_test)


# fit a base model
base_model = keras.Sequential([
    layers.Embedding(input_dim=num_tokens, output_dim=32, mask_zero=True),
    layers.GlobalAveragePooling1D(), # compress sequence
    layers.Flatten(),
    # layers.LSTM(32),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
base_model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

# evaluation
result_1 = base_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

plt.plot(result_1.history['accuracy'])
plt.plot(result_1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# fit a LSTM model
lstm_model = keras.Sequential([
    layers.Embedding(input_dim=num_tokens, output_dim=32, mask_zero=True),
    layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Flatten(),
    layers.Dropout(0.1),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

# evaluation
result_2 = lstm_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

plt.plot(result_2.history['accuracy'])
plt.plot(result_2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

cm = confusion_matrix(y_test, lstm_model.predict(X_test).round())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
