
import torch
from torch import nn



class LayerNorm(nn.Module):
    def __init__(self, embd, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embd))
        self.bias = nn.Parameter(torch.zeros(embd))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var= x.var(dim=-1, keepdim=True)
        norm= (x-mean)/torch.sqrt(var+self.eps)
        return norm*self.weight + self.bias


class GLUE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x=0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
        return x

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super(FeedForward, self).__init__()
        self.nn= nn.Sequential(
            nn.Linear(cfg["emb_dim"],cfg["emb_dim"]*4),
            GLUE(),
            nn.Linear(cfg["emb_dim"]*4,cfg["emb_dim"])
        )
    def forward(self, x):
        return self.nn(x)