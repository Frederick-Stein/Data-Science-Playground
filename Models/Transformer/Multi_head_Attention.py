import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

class MultiHeadedSelfAttention(nn.Module):
     
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()

        # torch.manual_seed(42)
        assert embedding_dim % num_heads == 0 # divisible
        self.head_dim = embedding_dim // num_heads

        # Bulid head
        self.att_heads = nn.ModuleList()
        for i in range(num_heads):
            self.att_heads.append(self.SingleHeadAttention(embedding_dim, self.head_dim))

        # Output projection back to embedding_dim
        self.W_output = nn.Linear(num_heads * self.head_dim, embedding_dim, bias=False)

    def forward(self, embedded: torch.Tensor):
        # embedded: (B, L, embedding_dim)

        head_outputs = []
        for head in self.att_heads:
            head_outputs.append(head(embedded)) # (B, L, head_dim)
        concatenated = torch.cat(head_outputs, dim = 2) # (B, L, num_heads * head_dim)
        output = self.W_output(concatenated) # (B, L, embedding_dim)

        return output



    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, head_dim: int):
            super().__init__()
    
            # torch.manual_seed(42)
            self.head_dim = head_dim
            self.W_q = nn.Linear(embedding_dim, head_dim, bias=False)
            self.W_k = nn.Linear(embedding_dim, head_dim, bias=False)
            self.W_v = nn.Linear(embedding_dim, head_dim, bias=False)
        
        def forward(self, embedded: torch.Tensor):
            # embedded: (B, L, embedding_dim)
            B, L, _ = embedded.shape

            Q = self.W_q(embedded) # (B, L, head_dim)
            K = self.W_k(embedded)
            V = self.W_v(embedded)

            scores = Q @ K.transpose(1, 2) 
            scaled_scores = scores / (self.head_dim ** 0.5)

            # Causal mask
            mask = torch.triu(torch.ones(L, L, device=scores.device, dtype=torch.bool), diagonal=1)
            scaled_scores = scaled_scores.masked_fill(mask.unsqueeze(0), float('-inf'))
            attention_weights = F.softmax(scaled_scores, dim = 2) # (B, L, L)
            attention_out = attention_weights @ V # (B, L, head_dim)

            return attention_out
