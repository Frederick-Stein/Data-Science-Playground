import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import TensorDataset, DataLoader



## mini sentence
token_to_id = {
    "what": 0,
    "is": 1,
    "the": 2,
    "weather": 3,
    "today": 4,
    "sunny": 5,
    'rainy': 6,
    'tomorrow': 7,
    "<EOS>": 8,
    }

id_to_token = {v: k for k, v in token_to_id.items()}

# input_sentence = "what is the weather today <EOS> sunny"
# output_sentence = "is the weather today <EOS> sunny <EOS>"

# input_tokens = input_sentence.split()
# output_tokens = output_sentence.split()

# input_ids = torch.tensor([token_to_id[token] for token in input_tokens])
# output_ids = torch.tensor([token_to_id[token] for token in output_tokens])


input_ids = torch.tensor([[token_to_id["what"], token_to_id["is"], token_to_id["the"], token_to_id["weather"], token_to_id["today"], token_to_id["<EOS>"], token_to_id["sunny"]],
                       [token_to_id["the"], token_to_id['weather'], token_to_id['tomorrow'], token_to_id['is'], token_to_id['what'], token_to_id['<EOS>'], token_to_id['rainy']]
                       ])

output_ids = torch.tensor([
    [token_to_id['is'], token_to_id['the'], token_to_id["weather"], token_to_id["today"], token_to_id["<EOS>"], token_to_id["sunny"], token_to_id['<EOS>']],
    [token_to_id['weather'], token_to_id['tomorrow'], token_to_id['is'], token_to_id['what'], token_to_id['<EOS>'], token_to_id['rainy'], token_to_id['<EOS>']],
])

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
input = torch.tensor([token_to_id["what"], token_to_id["is"], token_to_id["the"], token_to_id["weather"], token_to_id["today"], token_to_id["<EOS>"]])

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

# input = torch.tensor([token_to_id["what"], token_to_id["is"], token_to_id["the"], token_to_id["weather"], token_to_id["today"], token_to_id["<EOS>"]])
# input = torch.tensor([token_to_id["the"], token_to_id["weather"], token_to_id["is"], token_to_id["what"], token_to_id["tomorrow"], token_to_id["<EOS>"]])
# input = torch.tensor([token_to_id["what"], token_to_id["is"], token_to_id["the"], token_to_id["weather"], token_to_id["tomorrow"], token_to_id["<EOS>"]])
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
