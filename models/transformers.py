"""
gpt4o
"""
from conf import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .DCTRescalingLayer import *

class TransformerLayer(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads_sa, num_heads_ca, ff_dim, dropout=DROPOUT):
        super(TransformerLayer, self).__init__()
        
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads_sa, 
                                               dropout=dropout,  batch_first=True,)
        
        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim_q, kdim=dim_kv, vdim=dim_kv, num_heads=num_heads_ca, 
                                                dropout=dropout,  batch_first=True,)
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(dim_q, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim_q)
        )
        if 0:
            ff_dim_4SA = dim_q*4 # _dimB
            self.feedforward2 = nn.Sequential(
                nn.Linear(dim_q, ff_dim_4SA),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim_4SA, dim_q)
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)
        self.ctx_norm = nn.LayerNorm(dim_kv)
        self.norm3 = nn.LayerNorm(dim_q)
        if 0:  self.norm4 = nn.LayerNorm(dim_q)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # scaling elements
        if DS_SA:
            self.se_sa  = DCTRescalingLayer(dim_q)
        if DS_CA:
            self.se_ca  = DCTRescalingLayer(dim_q)
        if DS_FFN:
            self.se_ffn = DCTRescalingLayer(dim_q)
        if DS_ctx:
            self.se_ctx = DCTRescalingLayer(dim_kv)

    def forward(self, x, context, 
                dct_scale_4_dim2=None,
                src_mask=None, tgt_mask=None, memory_mask=None):
        # Self-attention
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask)
        if dct_scale_4_dim2 is not None:
            attn_output = attn_output * dct_scale_4_dim2[None,None,:]
        if DS_SA:
            attn_output = self.se_sa(attn_output)
        x = x + self.dropout(attn_output)
        
        if 0:
            # Feedforward2
            x_norm = self.norm4(x)
            ff_output = self.feedforward2(x_norm)
            if dct_scale_4_dim2 is not None:
                ff_output = ff_output * dct_scale_4_dim2[None,None,:]
            x = x + self.dropout(ff_output)
        
        # Cross-attention
        x_norm = self.norm2(x)
        if DS_ctx and DS_ctx_beforeLN:
            context = self.se_ctx(context)
        c_norm = self.ctx_norm(context) # cLN
        if DS_ctx and DS_ctx_afterLN:
            c_norm = self.se_ctx(c_norm)
        attn_output, _ = self.cross_attn(query=x_norm, key=c_norm, value=c_norm, attn_mask=memory_mask)
        if dct_scale_4_dim2 is not None:
            attn_output = attn_output * dct_scale_4_dim2[None,None,:]
        if DS_CA:
            attn_output = self.se_ca(attn_output)
        x = x + self.dropout(attn_output)
        
        # Feedforward
        x_norm = self.norm3(x)
        ff_output = self.feedforward(x_norm)
        if dct_scale_4_dim2 is not None:
            ff_output = ff_output * dct_scale_4_dim2[None,None,:]
        if DS_FFN:
            ff_output = self.se_ffn(ff_output)
        x = x + self.dropout(ff_output)
        
        return x

class Transformer(nn.Module):
    def __init__(self, L, dim_q, dim_kv, num_heads_sa, num_heads_ca, ff_dim, dropout):
        super(Transformer, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerLayer(dim_q, dim_kv, num_heads_sa, num_heads_ca, ff_dim, dropout)
            for _ in range(L)
        ])
        
    def forward(self, tgt, src, src_mask=None, tgt_mask=None, memory_mask=None):
        """
         src is kv (context), tgt is q (x).  transformer(x, context,)
        """
        # Assume src is the memory and tgt is the target sequence
        memory = src
        x = tgt
        
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, memory_mask)
        
        return x

class SA_Transformer(nn.Module):
    """
    only SA
    ( self attention
    """
    def __init__(self, L,dim_q,  num_heads_sa,  ff_dim, dropout):
        super(SA_Transformer, self).__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'sa': nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads_sa, 
                                            dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(dim_q),
                'ffn': nn.Sequential(
                    nn.Linear(dim_q, ff_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, dim_q)
                ),
                'norm2': nn.LayerNorm(dim_q)
            })
            for _ in range(L)
        ])
        # Dropout
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_mask=None,  ):
        for layer in self.layers:
            # Self-Attention
            attn_output, _ = layer['sa'](x, x, x, attn_mask=src_mask)
            x = x + self.dropout(attn_output)
            x = layer['norm1'](x)
            # Feed Forward
            ff_output = layer['ffn'](x)
            x = x + self.dropout(ff_output)
            x = layer['norm2'](x)
        return x
            

class CA_Transformer(nn.Module):
    """
    only CA
    """
    def __init__(self, L, dim_q, dim_kv, num_heads_ca, ff_dim, dropout):
        super(CA_Transformer, self).__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ca': nn.MultiheadAttention(embed_dim=dim_q, kdim=dim_kv, vdim=dim_kv,
                                          num_heads=num_heads_ca, dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(dim_q),
                'ffn': nn.Sequential(
                    nn.Linear(dim_q, ff_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, dim_q)
                ),
                'norm2': nn.LayerNorm(dim_q)
            })
            for _ in range(L)
        ])
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context, memory_mask=None):
        for layer in self.layers:
            # Cross-Attention
            attn_output, _ = layer['ca'](query=x, key=context, value=context, attn_mask=memory_mask)
            x = x + self.dropout(attn_output)
            x = layer['norm1'](x)
            # Feed Forward
            ff_output = layer['ffn'](x)
            x = x + self.dropout(ff_output)
            x = layer['norm2'](x)
        return x

if __name__=='__main__':
    # Example usage:
    # Initialize transformer
    transformer = Transformer(
        L=6,               # Number of layers
        dim_q=512,         # Dimension of query
        dim_kv=512,        # Dimension of key/value
        num_heads_sa=8,    # Number of self-attention heads
        num_heads_ca=8,    # Number of cross-attention heads
        ff_dim=2048,       # Feedforward dimension
        dropout=0.1        # Dropout rate
    )

    # Example input (random tensors)
    src = torch.rand(10, 32, 512)  # (sequence length, batch size, dim)
    tgt = torch.rand(20, 32, 512)  # (sequence length, batch size, dim)

    # Forward pass
    output = transformer(tgt, src)
    print(output.shape)  # Output shape