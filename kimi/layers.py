import torch 
from torch import nn
from torch.nn import functional as F

class SWiGLU(nn.Module):
    def __init__(self, dd_in,intermidate_dim,bias=False):
        super().__init__()
        self.up= nn.Linear(dd_in,intermidate_dim,bias=bias)
        self.down= nn.Linear(dd_in,intermidate_dim,bias=bias)
        self.gate= nn.Linear(intermidate_dim,dd_in,bias=bias)
    
    def forward(self,x):
        up= self.up(x)
        down=self.down(x)
        return self.gate(F.silu(up)*down)
    

class RMSNorm(nn.Module):
    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)
    
class RotaryEmbeddings(nn.Module):
    def __init__(self, dim,seq_len):
        super().__init__()
        inv_freq=1.0/(10000**(torch.arange(0,dim,2).float()/dim))
        self.register_buffer("freq",inv_freq)
        self.seq_len= seq_len

    def forward(self,x,seq_len=None):
        if seq_len is None:
            seq_len=x.size(-2)
        t=torch.arange(seq_len,dtype=x.dtype)
        freq=torch.outer(t,self.freq)
        return freq.cos(),freq.sin()
    

def apply_rope(x,cos,sin):
    f,s= x.chunk(2,dim=-1)
    return torch.cat([s*cos-f*sin,f*sin+s*cos],dim=-1)