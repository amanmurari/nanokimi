from kimi.layers import RMSNorm
from kimi.moe import MoELayer
from kimi.attention import MultiHeadLatentAttention
import torch 
from torch import nn

from torch.nn import functional as F



class KimiBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
    
        self.attn=MultiHeadLatentAttention(cfg)
        self.norm1=RMSNorm(cfg.n_embd)
        self.norm2=RMSNorm(cfg.n_embd)
        self.moe= MoELayer(cfg)

    def forward(self,x):
        x=x+self.attn(self.norm1(x))
        x=x+self.moe(self.norm2(x))
        return x
    

class KiMik2(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.emb= nn.Embedding(cfg.vocab_size,cfg.n_embd)
        
        self.drop= nn.Dropout(cfg.dropout)
        self.trf= nn.ModuleList([KimiBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm= nn.RMSNorm(cfg.n_embd)
        self.lm_head= nn.Linear(cfg.n_embd,cfg.vocab_size,bias=False)
    

    def forward(self,x):
        x= self.emb(x)
        x=self.drop(x)
        for block in self.trf:
            x=block(x)
        x= self.norm(x)
        x=self.lm_head(x)
        return x
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits= self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx