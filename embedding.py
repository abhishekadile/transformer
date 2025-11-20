"""
Token + positional encoding

"""
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.module):
  """
  Converting token indices to dense vectors
  Each token will gets its own learnable embedding vector
  It will help to encode the input data
  """
  def __init__(self, vocab_size, d_model):
    """
    Args:
         vocab_size: Size of the vocabbulary(number of unique tokens)
         d_model: Dimesions of embedding(typically 512 in original paper)
    """
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.d_model= d_model
  def forward(self, x):
    """
    Args:
         x: TOken indices, shape  (batch_size, seq_len)
    Returns:
         Embedding scaled by sqrt(d_model), shape (batch_size, seq_len, d_model)
    """
    return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
  """
  Adds positional information using sin/cos functions.
  This lets the model know the position of each token in the sequence
  This component help the model understand the position of the token in the sequence
  """
  def __init__(self, d_model, max_len=5000, dropout=0.1):
    """
    Args:
         d_model: Dimension of embeddings
         max_len: Maximum sequence length to precompute
         dropout: Dropout rate to apply after adding positional encoding
    """
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    
    #Create the matrix of the shape(max_len, d_model) for positional encoding
    pe = torch.zeros(max_len, d_model)
    
    #Create postion indices [0, 1, 2, ..., max_len-1]
    postion = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    #Create the div_term for the sinusodial functions
    #This creates the different frequencies for each dimension
    dive_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    #Apply sin to even indices
    pe[:, 0::2] = torch.sin(position * div_term)

    #Apply cost to odd indices
    pe[:, 1::2] = torch.cos(position * div_term)

    #Add batch dimension: (max_len, d_model) -> (1, max_len, d_model)
    pe = pe.unsqueeze(0)

    #Register as buffer (not a parameter, but should be saved with model)
    self.register_buffer('pe', pe)
  
  def forward(self, x):
    """
    Args:
        x:Embedding, shape (batch_size, seq_len, d_model)
    Return:
        EMbedding with a positional encoding added, same shape
    """
    #Add positional encoding to input embeddings
    #We slice pe to mach the sequence length of x
    x = x + self.pe[:, :x.size(1), :]
    return self.dropout(x)
    

