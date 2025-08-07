from kimi.layers import SWiGLU
import torch 
from torch import nn
from torch.nn import functional as F

class MoELayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        self.num_experts_per_tok = cfg.n_experts_per_token
        self.num_experts = cfg.n_experts
        self.gate = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)

        # meta device to reduce memory pressure when initializing the model before loading weights
       
        self.fc1 = nn.ModuleList([
            nn.Linear(
                cfg.n_embd, cfg.expert_intermediate_size,
                bias=False)
            for _ in range(cfg.n_experts)]
        )
        self.fc2 = nn.ModuleList([
             nn.Linear(
                cfg.n_embd, cfg.expert_intermediate_size,
                bias=False)
            for _ in range(cfg.n_experts)]
        )
        self.fc3 = nn.ModuleList([
            nn.Linear(
                cfg.expert_intermediate_size,cfg.n_embd,
                bias=False)
            for _ in range(cfg.n_experts)]
        )
        if cfg.use_shared_expert:
            self.shared_expert= SWiGLU(cfg.n_embd,cfg.shared_expert_intermediate_size,bias=cfg.bias )
    def forward(self, x):
        b, seq_len, embed_dim = x.shape
        scores = self.gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)
        
        expert_outputs = []
        for e in range(self.num_experts):
            hidden = torch.nn.functional.silu(self.fc1[e](x)) * self.fc2[e](x)
            out = self.fc3[e](hidden)
            expert_outputs.append(out.unsqueeze(-2))
        expert_outputs = torch.cat(expert_outputs, dim=-2)  # (b, t, num_experts, emb_dim)

        gating_probs = torch.zeros_like(scores)

        for i in range(self.num_experts_per_tok):
            indices = topk_indices[..., i:i+1]
            prob = topk_probs[..., i:i+1]
            gating_probs.scatter_(dim=-1, index=indices, src=prob)
        gating_probs = gating_probs.unsqueeze(-1)  # (b, t, num_experts, 1)
        
        # Weighted sum over experts
        y = (gating_probs * expert_outputs).sum(dim=-2)
        if self.cfg.use_shared_expert:
            shared=self.shared_expert(x)
            y+=shared
        return y
    

