import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import TensorDataset, DataLoader



## generalized sentence encode
sentence1 = "what is the weather today sunny"
sentence2 = "please tell me the weather tomorrow rainy"
sentences = [sentence1, sentence2]

words = set()
for sentence in sentences:
    for word in sentence.split():
        words.add(word)

token_to_id = {"<PAD>": 0, "<EOS>": 1}
id_to_token = {0: "<PAD>", 1: "<EOS>"}
for i, word in enumerate(words, start = 2):
    token_to_id[word] = i
    id_to_token[i] = word
print(token_to_id)
print(id_to_token)

def encode(sentence):
    ids = []
    for word in sentence.split():
        ids.append(token_to_id[word])
    return ids

encoded_sentences = []
for sentence in sentences:
    encoded = encode(sentence)
    encoded.append(token_to_id["<EOS>"])
    encoded_sentences.append(torch.tensor(encoded))

input_ids = nn.utils.rnn.pad_sequence(encoded_sentences, batch_first=True, padding_value=0)
output_ids = input_ids.clone()
output_ids[:, :-1] = input_ids[:, 1:]
output_ids[:, -1] = 0
print(input_ids)
print(output_ids)

train_data = TensorDataset(input_ids, output_ids)
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)



## construct transformer
class PositionEncoding(nn.Module):

    def __init__(self, d_model = 2, vocab_size = 6):

        ## d_model is the dim word embeddings
        ## max_len is length of longest sentence we can generate

        super().__init__()

        pe = torch.zeros(vocab_size , d_model)

        pos = torch.arange(0, vocab_size , step = 1, dtype = torch.float).unsqueeze(1) # (max_len, 1)
        embedding_idx = torch.arange(0, d_model, step = 2, dtype = torch.float) # (d/2,)

        div_term = 1 / torch.pow(10000, (2 * embedding_idx) / d_model)

        pe[:, 0::2] = torch.sin(pos * div_term) # # (max_len, d/2)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0) # (1, max_len, d)

        self.register_buffer("pe", pe)

    def forward(self, word_embeddings):
        B, L, d = word_embeddings.shape # expects shape (B, L, d)
        return word_embeddings + (self.pe[:, :L, :]).requires_grad_(False)

class Attention(nn.Module):

    def __init__(self, d_model = 2):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_model, bias = False)
        self.W_k = nn.Linear(d_model, d_model, bias = False)
        self.W_v = nn.Linear(d_model, d_model, bias = False)

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask = None):
        # encodings_* shapes: (B, Lq, d), (B, Lk, d), (B, Lk, d)

        Q = self.W_q(encodings_for_q) # (B, Lq, d)
        K = self.W_k(encodings_for_k) # (B, Lk, d)
        V = self.W_v(encodings_for_v) # (B, Lv, d)

        ## softmax(Q * K^T / sqrt(d_k) + M) * V
        similarity = Q @ K.transpose(-1, -2) # Q * K^T

        # d_k
        d_k = K.size(-1)

        # scaled_simalrity = simalrity / torch.sqrt(torch.tensor(K.shape(-1))) # Q * K^T / sqrt(d_k)
        scaled_similarity = similarity / math.sqrt(d_k)# Q * K^T / sqrt(d_k)

        if mask is not None:
            scaled_similarity = scaled_similarity.masked_fill(mask, float("-inf")) # Q * K^T / sqrt(d_k) + M

        attention_weights = F.softmax(scaled_similarity, dim = -1) # softmax(Q * K^T / sqrt(d_k) + M)

        attention_scores = attention_weights @ V # softmax(Q * K^T / sqrt(d_k) + M) * V

        return attention_scores


class MiniTransformer(nn.Module):

    def __init__(self, d_model = 2, num_tokens = 7, vocab_size  = 7):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings = num_tokens, embedding_dim = d_model)

        self.pe = PositionEncoding(d_model = d_model, vocab_size  = vocab_size )

        self.attention = Attention(d_model = d_model)

        self.fc_layer = nn.Linear(d_model, num_tokens)

    def forward(self, inputs):
        # inputs: (B, L) of token ids
        B, L = inputs.shape

        word_embeddings = self.embedding(inputs)

        pos_encoded = self.pe(word_embeddings)

        device = inputs.device

        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0)

        self_attention_values = self.attention(pos_encoded, pos_encoded, pos_encoded, mask)

        x = pos_encoded + self_attention_values # residual

        outputs = self.fc_layer(x) # logits

        return outputs



## test model
model = MiniTransformer(num_tokens = len(token_to_id), d_model = 4, vocab_size  = 8)
test = "what is the weather today <EOS>"
input = torch.tensor(encode(test))

input_length = input.size(0)
predictions = model(input.unsqueeze(0)).squeeze(0)
print(predictions)
predicted_id = torch.tensor([torch.argmax(predictions[-1, :]).detach()])

predicted_ids = predicted_id


max_length = 8
for i in range(input_length, max_length):
    if (predicted_id == token_to_id["<EOS>"]):
        break
    input = torch.cat((input, predicted_id))
    predictions = model(input.unsqueeze(0)).squeeze(0)
    predicted_id = torch.tensor([torch.argmax(predictions[-1, :]).detach()])
    predicted_ids = torch.cat((predicted_ids, predicted_id))

for id in predicted_ids:
    print(id_to_token[id.item()])

for id in input_ids[0]:
    print(id_to_token[id.item()])



## train model
model = MiniTransformer(num_tokens = len(token_to_id), d_model = 16, vocab_size  = 8)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-3)


epochs = 100

model.train()
for epoch in range(epochs):

    for X, y in train_loader:

        y_pred = model(X)
        B, L, d = y_pred.shape
        loss = loss_fn(y_pred.view(B*L, d), y.view(B*L))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# X, y = next(iter(train_loader))
# print(X.shape, y.shape)


## evaluate result
model.eval()

test1 = "what is the weather today"
test2 = "please tell me the weather tomorrow"
test3 = "what is the weather tomorrow"
test4 = "weather today is"

test = torch.tensor(encode(test4))
input_length = len(test)

predictions = model(input.unsqueeze(0)).squeeze(0)
predicted_id = torch.tensor([torch.argmax(predictions[-1, :]).detach()])
predicted_ids = predicted_id

max_length = 8
for i in range(input_length, max_length):
    if (predicted_id == token_to_id["<EOS>"]):
        break
    input = torch.cat((input, predicted_id))
    predictions = model(input.unsqueeze(0)).squeeze(0)
    predicted_id = torch.tensor([torch.argmax(predictions[-1, :]).detach()])
    predicted_ids = torch.cat((predicted_ids, predicted_id))

for id in predicted_ids:
    print(id_to_token[id.item()])
