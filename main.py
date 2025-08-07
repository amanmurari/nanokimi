from kimi.model import KiMik2
import torch
from infrance import generate_text
from gpt.model import GPTModel
import os
from config import KimiConfig,GPT_CONFIG_124M
gpt= GPTModel(GPT_CONFIG_124M)
config=KimiConfig()
kimi=KiMik2(config)
if os.path.exists("models/kimik2.pt"):
    kimi.load_state_dict(torch.load("models/kimik2.pt"))
    print("loader kimi....")
if os.path.exists("models/gpt2.pt"):
    gpt.load_state_dict(torch.load("models/gpt2.pt"))
    print("loader gpt....")
print(generate_text(gpt,"tensorflow js is my faviorate",100,1))