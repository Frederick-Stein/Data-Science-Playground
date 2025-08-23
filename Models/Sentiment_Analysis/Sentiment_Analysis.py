import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm ## progress bar

## get data from github
df = pd.read_csv("https://raw.githubusercontent.com/Frederick-Stein/Data-Science-Playground/refs/heads/main/ToyModels/Sentiment_Analysis/preprocessed_data.csv")
df.head(10)
sentences = df['able play youtube alexa']
labels = df['0.5']
labels = torch.tensor(labels.values, dtype=torch.float32)
labels.shape


torch.manual_seed(42)
## prepare encoded data
def prepare_data(sentences):

    words = set()
    for sentence in sentences:
        for word in sentence.split():
            words.add(word)

    token_to_id = {"<PAD>": 0, "<EOS>": 1}
    id_to_token = {0: "<PAD>", 1: "<EOS>"}
    for i, word in enumerate(words, start = 2):
        token_to_id[word] = i
        id_to_token[i] = word
    
    vocab_size = len(token_to_id)

    def encode(sentence):
        ids = []
        for word in sentence.split():
            ids.append(token_to_id[word])
        return ids
    
    encoded_sentences = [torch.tensor(encode(sentence)) for sentence in sentences]
    padded_sentences =  nn.utils.rnn.pad_sequence(encoded_sentences, batch_first=True, padding_value=0)
    return vocab_size, padded_sentences, token_to_id, id_to_token


## prepare data
vocab_size, input, token_to_id, id_to_token = prepare_data(sentences)
output = labels.unsqueeze(1)
dataset = TensorDataset(input, output)

## get train_data and test_data
train_size = int(0.8 * len(dataset))
test_size  = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

## set dataloader
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)



## construct function
class EmotionPredictor(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.linear = nn.Linear(d_model, 1)
        self.tanch = nn.Tanh()
    
    def forward(self, x):

        embedding = self.embedding(x)
        averaged = torch.mean(embedding, dim = 1)
        full_layer = self.linear(averaged)
        output = self.tanch(full_layer)
        return output


## training process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionPredictor(vocab_size, 256).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 1e-4)

train_error = []
# train_accuracy = []
test_error = []
# test_accuracy = []

Epochs = 100
for epoch in tqdm(range(1, Epochs + 1)):
    model.train()
    train_loss = 0

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)    
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_error.append(train_loss)

    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_error.append(test_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")


## plot error
plt.plot(train_error, label="train_error")
plt.plot(test_error, label="test_error")
plt.legend()
plt.show()


## test
test1 = "echo dot not work pls provide service center detail"
test2 = 'best thing ever'
test3 = "i dont like this movie"
test4 = "weird but funny movie"
test = [test1, test2, test3, test4]

encoded_test = []
for sentence in test:
    ids = []
    for word in sentence.split():
        if word in token_to_id:
            ids.append(token_to_id[word])
        else:
            ids.append(0)
    encoded_test.append(torch.tensor(ids))

padded_test = nn.utils.rnn.pad_sequence(encoded_test, batch_first=True, padding_value=0)
padded_test = padded_test.to(device)
with torch.inference_mode():
    output = model(padded_test)

print(output.tolist())




