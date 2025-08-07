import torch
import time
from torch import nn
from kimi.model import KiMik2
from gpt.model import GPTModel
from kimi.muon import MuonWithAuxAdam
from config import KimiConfig,GPT_CONFIG_124M
from training.training import train_model
from training.data_loader import train_loader,val_loader
from kimi.muon import MuonWithAuxAdam
import tiktoken
import torch.distributed as dt
import os
import gc
gc.collect()

os.environ["MASTER_ADDR"]="localhost"
os.environ["MASTER_PORT"]="12355"
device= "cuda" if torch.cuda.is_available() else "cpu"
dt.init_process_group(backend="nccl" if device=="cuda" else "gloo",rank=0,world_size=1,)

tokenizer= tiktoken.get_encoding("gpt2")

kimi_m= KiMik2(KimiConfig())
gpt_m= GPTModel(GPT_CONFIG_124M)

if os.path.exists("models/kimik2.pt"):
    kimi_m.load_state_dict(torch.load("models/kimik2.pt"))
    print("loader kimi....")
if os.path.exists("models/gpt2.pt"):
    gpt_m.load_state_dict(torch.load("models/gpt2.pt"))
    print("loader gpt....")

hid=[p for p in kimi_m.parameters() if p.ndim==2]
nhid=[p for p in kimi_m.parameters() if p.ndim!=2]
parms_group= [
    dict(params=hid,use_muon=True,lr=0.01,weight_decay=0.01),
    dict(params=nhid,use_muon=False,lr=3e-4,weight_decay=0.00001),
]

num_epochs=10
kimi_m=kimi_m.to(device)
gpt_m=gpt_m.to(device)
kimi_m.qk_pair= [(kimi_m.trf[i].attn.q_proj.weight,kimi_m.trf[i].attn.kv_proj.weight) for i in range(kimi_m.cfg.n_layer)]

k_optim= MuonWithAuxAdam(parms_group)
g_optim = torch.optim.AdamW(gpt_m.parameters(), lr=3e-4, weight_decay=0.0001)
start_time= time.time()
print("training...........")
train_losses, val_losses, tokens_seen=train_model(gpt_m,kimi_m,train_loader,val_loader,g_optim,k_optim,device,num_epochs, eval_freq=5, eval_iter=5,
     start_context="tensorflow js is a good", tokenizer=tokenizer)


torch.save([train_losses,val_losses,tokens_seen],"models/ckp.pt")

end_time = time.time()

execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.") 
