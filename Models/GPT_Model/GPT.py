import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math

#----------------------------------------------------------------------------------------------------------------------------------------------------------
### prepare context
SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
def build_vocab(sentences, min_freq=1, lowercase=True, max_vocab=None):
    if lowercase:
        sentences = [s.lower() for s in sentences]
    counter = Counter()
    for s in sentences:
        counter.update(s.split())
    # Stable, frequency-sorted vocab (ties broken lexicographically)
    words = sorted([w for w, c in counter.items() if c >= min_freq])
    if max_vocab is not None:
        words = words[: max(0, max_vocab - len(SPECIALS))]

    token_to_id = {tok: i for i, tok in enumerate(SPECIALS)}
    start = len(SPECIALS)
    for i, w in enumerate(words, start=start):
        token_to_id[w] = i
    id_to_token = {i: t for t, i in token_to_id.items()}
    return token_to_id, id_to_token

def encode_sentence(sentence, token_to_id, add_bos=True, add_eos=True, lowercase=True):
    if lowercase:
        sentence = sentence.lower()
    ids = []
    if add_bos:
        ids.append(token_to_id["<BOS>"])
    for w in sentence.split():
        ids.append(token_to_id.get(w, token_to_id["<UNK>"]))
    if add_eos:
        ids.append(token_to_id["<EOS>"])
    return torch.tensor(ids, dtype=torch.long)

def pad_batch(encoded, pad_id=0, max_len=None, batch_first=True):
    if max_len is not None:
        # truncate or pad to max_len
        clipped = []
        for t in encoded:
            if t.numel() > max_len:
                clipped.append(t[:max_len])
            else:
                clipped.append(t)
        encoded = clipped
    padded = nn.utils.rnn.pad_sequence(encoded, batch_first=batch_first, padding_value=pad_id)
    # attention mask: 1 for tokens, 0 for pad
    mask = (padded != pad_id).to(torch.bool)
    lengths = mask.sum(dim=1 if batch_first else 0) # length of real tokens in each sentence
    return padded, mask, lengths

def Encode(sentences, *, min_freq=1, lowercase=True, max_vocab=None, add_bos=True, add_eos=True, max_len=None, batch_first=True):

    token_to_id, id_to_token = build_vocab(
        sentences, min_freq=min_freq, lowercase=lowercase, max_vocab=max_vocab
    )
    encoded = [
        encode_sentence(s, token_to_id, add_bos=add_bos, add_eos=add_eos, lowercase=lowercase) for s in sentences
    ]
    padded_ids, attn_mask, lengths = pad_batch(
        encoded, pad_id=token_to_id["<PAD>"], max_len=max_len, batch_first=batch_first
    )
    vocab_size = len(token_to_id)
    return vocab_size, padded_ids, attn_mask, lengths, token_to_id, id_to_token


#----------------------------------------------------------------------------------------------------------------------------------------------------------

### Embedding function
class Embedding(nn.Module):

    def __init__(self, vocab_size: int, max_context_length: int, embedding_dim: int, dropout: float = 0.1, pad_idx: int | None = None):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.position_embeddings = nn.Embedding(max_context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Cache [0, 1, 2, ..., max_context_len-1] as a buffer
        self.register_buffer(
            "position_ids",
            torch.arange(max_context_length, dtype=torch.long).unsqueeze(0),  # (1, max_len)
            persistent=False
        )


    def forward(self, context: torch.Tensor):
        # context (B, L) B : batch size, L: context length
        assert context.dtype == torch.long
        B, L = context.shape
        # check if input context longer than max_context_length
        if L > self.max_context_length:
            raise ValueError(f"Sequence length {L} exceeds max_context_length {self.max_context_length}")

        word_embeddings = self.word_embeddings(context) * math.sqrt(self.embedding_dim)# (B, L, embedding_dim)
        positions = self.position_ids[:, :L] # (1, L)
        position_embeddings = self.position_embeddings(positions) # (1，L, embedding_dim)
        output = word_embeddings + position_embeddings
        return self.dropout(output)



### Multi-head attention
class MultiHeadedSelfAttention(nn.Module):
     
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()

        assert embedding_dim % num_heads == 0 # divisible
        self.head_dim = embedding_dim // num_heads

        # Bulid head
        self.att_heads = nn.ModuleList()
        for i in range(num_heads):
            self.att_heads.append(self.SingleHeadAttention(embedding_dim, self.head_dim, dropout, causal))

        # Output projection back to embedding_dim
        self.W_output = nn.Linear(num_heads * self.head_dim, embedding_dim, bias=False)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, embedded: torch.Tensor, attn_mask: torch.Tensor | None = None):
        # embedded: (B, L, embedding_dim)

        head_outputs = []
        for head in self.att_heads:
            head_outputs.append(head(embedded, attn_mask = attn_mask)) # (B, L, head_dim)
        concat_heads= torch.cat(head_outputs, dim = 2) # (B, L, num_heads * head_dim)
        output = self.W_output(concat_heads) # (B, L, embedding_dim)

        return self.proj_dropout(output)



    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, head_dim: int, dropout: float = 0.1, causal: bool = True):
            super().__init__()

            self.causal = causal
            self.head_dim = head_dim
            self.W_q = nn.Linear(embedding_dim, head_dim, bias=False)
            self.W_k = nn.Linear(embedding_dim, head_dim, bias=False)
            self.W_v = nn.Linear(embedding_dim, head_dim, bias=False)
            self.attn_dropout = nn.Dropout(dropout)
        
        def forward(self, embedded: torch.Tensor, attn_mask: torch.Tensor | None = None):
            # embedded: (B, L, embedding_dim)
            B, L, _ = embedded.shape

            Q = self.W_q(embedded) # (B, L, head_dim)
            K = self.W_k(embedded)
            V = self.W_v(embedded)

            scores = Q @ K.transpose(-2, -1) 
            scaled_scores = scores / (self.head_dim ** 0.5)

            # Causal mask
            if self.causal:
                mask = torch.triu(torch.ones(L, L, device=scores.device, dtype=torch.bool), diagonal=1)
                scaled_scores = scaled_scores.masked_fill(mask.unsqueeze(0), float('-inf'))

            # padding mask: attn_mask expected shape (B, L) with True for keep/1 for tokens
            if attn_mask is not None:
                key_keep = attn_mask.to(torch.bool).unsqueeze(1) # (B, 1, L)
                scaled_scores = scaled_scores.masked_fill(~key_keep, float('-inf'))
    
            attention_weights = F.softmax(scaled_scores, dim = -1) # (B, L, L)
            attention_weights = self.attn_dropout(attention_weights)
            attention_out = attention_weights @ V # (B, L, head_dim)

            return attention_out


### Transformer block
class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        self.attention = MultiHeadedSelfAttention(embedding_dim, num_heads, dropout=dropout, causal=causal)
        self.ffn = FFN(embedding_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        # optional residual dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        # embedded: (B, L, embedding_dim)
        x = x + self.attn_dropout(self.attention(self.norm1(x), attn_mask = attn_mask))
        x = x + self.ffn_dropout(self.ffn(self.norm2(x)))
        return x


### feed forward network
class FFN(nn.Module):

    def __init__(self, embedding_dim: int, dropout: float = 0.1, expansion_factor: int = 4):
        super().__init__()
        hidden_dim = expansion_factor * embedding_dim
        self.block = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), # up-projection
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim), # down-projection
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


### GPT model
class GPT(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, max_context_length: int, num_blocks: int, num_heads: int, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        self.embedding = Embedding(vocab_size, max_context_length, embedding_dim, dropout=dropout, pad_idx=0)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, dropout=dropout, causal=causal) for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.vocab_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        # tie projection weight with embedding
        self.vocab_projection.weight = self.embedding.word_embeddings.weight
        
    def forward(self, context: torch.Tensor, attn_mask: torch.Tensor | None = None):
        # context: (B, L)
        # attn_mask (B, L)
        x = self.embedding(context) # (B, L, d)
        for block in self.transformer_blocks:
            x = block(x, attn_mask = attn_mask) # (B, L, d)
        x = self.final_norm(x) # (B, L, d)
        raw_output = self.vocab_projection(x) # logits

        return raw_output
