from dataclasses import dataclass

@dataclass
class KimiConfig:
    # Model architecture
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 4

    # MLA configuration
    kv_lora_rank: int = 128
    q_lora_rank: int = 192
    rope_dim: int = 32

    # MoE configuration
    n_experts: int = 8
    n_experts_per_token: int = 2
    expert_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 768
    use_shared_expert: bool = True


    # Training parameters
    dropout: float = 0.1
    bias: bool = True
    


GPT_CONFIG_124M = {
    "vocab_size": 50257,   
    "context_length": 1024,
    "emb_dim": 768,        
    "n_heads": 12,        
    "n_layers": 12,         
    "drop_rate": 0.1,      
    "qkv_bias": False    
}