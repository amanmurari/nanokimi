import torch
from infrance import text_to_token_ids,token_ids_to_text,generate_text
import time
import tqdm

import tiktoken
from training.data_loader import val_loader,train_loader


def qk_clip(parms,tau=100.0,alpha=0.5):
    for w_q,w_k in parms:
        with torch.inference_mode():
            max_s=(w_q@w_k.T).abs().sum()
            if max_s>tau:
                eta= tau/max_s
                w_q.mul_(eta**alpha)
                w_k.mul_(eta**(1-alpha))

tokenizer= tiktoken.get_encoding("gpt2")      
start_time = time.time

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch=input_batch.to(device)
    target_batch=target_batch.to(device)
    model=model.to(device)
    logits= model(input_batch)
    loss= torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss






def gpt2_trainer(model,optimizer,input_ids,target_batch,device):
    
    loss= calc_loss_batch(input_ids,target_batch,model,device)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    

def kimik2_trainer(model,optimizer,input_ids,target_batch,device):
    
    loss= calc_loss_batch(input_ids,target_batch,model,device)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    qk_clip(model.qk_pair)
    


def train_model(gptm,kimim, train_loader, val_loader, g_optim,k_optim, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    
    train_losses, val_losses, track_tokens_seen = {"kimi":[],"gpt":[]}, {"kimi":[],"gpt":[]}, []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        gptm.train()  # Set model to training mode
        kimim.train()  # Set model to training mode
        for input_batch,target_batch in tqdm.tqdm(train_loader):
            
            gpt2_trainer(gptm,g_optim,input_batch,target_batch,device)
            kimik2_trainer(kimim,k_optim,input_batch,target_batch,device)
            
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0: 
                
                gtrain_loss, gval_loss = evaluate_model(
                    gptm, train_loader, val_loader, device, eval_iter)
                
                train_losses["gpt"].append(gtrain_loss)
                val_losses["gpt"].append(gval_loss)
                ktrain_loss, kval_loss = evaluate_model(
                    kimim, train_loader, val_loader, device, eval_iter)
                
                train_losses["kimi"].append(ktrain_loss)
                val_losses["kimi"].append(kval_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} of kimi (Step {global_step:06d}): "
                        f"Train loss {ktrain_loss:.3f}, Val loss {kval_loss:.3f}")
                print(f"Ep {epoch+1} of gpt (Step {global_step:06d}): "
                        f"Train loss {gtrain_loss:.3f}, Val loss {gval_loss:.3f}")

        # Print a sample text after each epoch
        torch.save([train_losses,val_losses,tokens_seen],f"models/losses/loss{epoch+4}.pt")
        torch.save(kimim.state_dict(),"models/kimik2.pt")

        torch.save(gptm.state_dict(),"models/gpt2.pt")
        print("gpt ........",end="\t")
        print(generate_text(
            gptm,start_context,20
        ))
        print("kimi ........",end="\t")
        print(generate_text(
            kimim,start_context,20
        ))
        

    return train_losses, val_losses, track_tokens_seen










# def trainer():
#     device= "cuda" if torch.cuda.is_available() else "cpu"
#     torch.manual_seed(123)
#     model = Llama2Model(LLAMA2_CONFIG_7B)
#     model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

#     num_epochs=10
#     train_losses, val_losses, tokens_seen = train_model(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=5, eval_iter=5,
#     start_context="Every effort moves you", tokenizer=tokenizer
#     )

#     torch.save(model.parameters(),"best_llama_v2.pt")
#     end_time = time.time()
#     execution_time_minutes = (end_time - start_time) / 60
#     print(f"Training completed in {execution_time_minutes:.2f} minutes.") 
#     return train_losses,val_losses,tokens_seen          
