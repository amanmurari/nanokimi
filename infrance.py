import torch
import tiktoken

from rich.console import Console
tokenizer= tiktoken.get_encoding("gpt2")
console=Console()
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())



def generate_text(model, prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
   
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    
    context = text_to_token_ids(prompt,tokenizer)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature, top_k)
    
    # Decode and return
    result = token_ids_to_text(generated,tokenizer)
    return result

