
import torch
from torch import nn
from torch.nn import functional as F
from gpt.attention import MultiHeadAttention
from gpt.layers import LayerNorm,FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ln1= LayerNorm(embd=cfg["emb_dim"])
        self.ln2= LayerNorm(embd=cfg["emb_dim"])
        self.ff= FeedForward(cfg)
        self.dropout= nn.Dropout(cfg["drop_rate"])
        
    def forward(self, x):
        sortcut=x
        x= self.ln1(x)
        x=self.attn(x)
        x=self.dropout(x)
        x= x+sortcut
        sortcut =x
        x=self.ln2(x)
        x=self.ff(x)
        x=self.dropout(x)
        x=x+sortcut
        return x
    

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg= cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["context_length"]:]
            logits= self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx