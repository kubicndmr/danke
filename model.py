import sys
import math
import torch
import torch.nn.functional as F

from PIL import Image
from torch import nn
from collections import OrderedDict

# Transformer
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
        
        
class SLPEncoder(nn.Module):
    def __init__(self, model_dim, num_heads, dropout_prob):
        super().__init__()
        
        # Attention layer
        self.self_attn = MultiheadAttention(model_dim, model_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout_prob),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, model_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask) # (B, L, model_dim)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x) # (B, L, model_dim)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x
"""
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        '''
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
""" 
def save_tensor_as_image(tensor, file_path):
    """
    Save a PyTorch tensor as an image file.

    Args:
    - tensor (torch.Tensor): The tensor to save as an image.
    - file_path (str): The path where the image will be saved.
    """
    # Normalize the tensor to the range [0, 255]
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)  # Scale to [0, 1]
    tensor = tensor * 255  # Scale to [0, 255]
    tensor = tensor.byte()  # Convert to byte type

    # Convert the tensor to a numpy array and then to a PIL image
    np_array = tensor.cpu().numpy()
    image = Image.fromarray(np_array)

    # Save the image
    image.save(file_path)
    
class SLPNet(nn.Module):
    def __init__(self, model_dim, num_head, num_encoder, num_classes, dropout_prob):
        super().__init__()
        self.model_dim = model_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.embed_in = nn.Sequential(
            nn.Conv1d(
                in_channels=1024,
                out_channels=model_dim, 
                kernel_size=1
            ),
            nn.BatchNorm1d(model_dim)
        )
        
        encoder_stack = OrderedDict()
        for n in range(num_encoder):
            encoder_stack[f'Encoder{n}'] = SLPEncoder(
                model_dim, 
                num_head, 
                dropout_prob=dropout_prob
            )
        self.encoder = nn.Sequential(encoder_stack)
        
        self.classfier = nn.Linear(
            in_features=model_dim,
            out_features=num_classes
        )
    
    def positional_encoding(self, idx):
        pe = torch.zeros((self.model_dim, idx.shape[0])).to(self.device)
        position = torch.arange(0, self.model_dim, dtype=torch.float).unsqueeze(1).to(self.device)
        div_term = torch.pow(10000, (position // 2 * 2) / self.model_dim).to(self.device)

        pe[0::2, :] = torch.sin(idx / div_term[0::2])
        pe[1::2, :] = torch.cos(idx / div_term[1::2])

        return pe.T
      
    def forward(self, x, t):
        t = self.positional_encoding(t) # (B, model_dim)
        t = t.unsqueeze(-1) # (B, model_dim, L)
        x = self.embed_in(x) # (B, 1024, L) -> (B, model_dim, L)
        x = x + t # (B, model_dim, L)
        x = self.encoder(torch.transpose(x, 1, 2)) # -> (B, L, model_dim)
        x = self.classfier(x) # -> (B, L, num_classes)
        return x.squeeze() # -> (B, num_classes)
    
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps