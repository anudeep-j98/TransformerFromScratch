import streamlit as st
import os
import math
import time
import inspect
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AdamW, get_scheduler
from model import GPT, GPTConfig

@st.cache_resource
def load_model(model_path):
  """Load the trained model"""
  try:
      torch.serialization.add_safe_globals({'GPTConfig': GPTConfig})
      checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
      config = GPTConfig()
      model = GPT(config)
      model.load_state_dict(checkpoint['model_state_dict'])
      model.eval()
      return model
  except Exception as e:
      print(f"Error loading model: {e}")
      return None

def generate_text(model, prompt, max_new_tokens=50, temperature=0.8, top_k=40):
    """Generate text based on a prompt
    Args:
        model: The GPT model
        prompt (str): Input text to continue from
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Higher values produce more diverse text (default: 0.8)
        top_k (int): Number of highest probability tokens to consider (default: 40)
    Returns:
        str: Generated text including the original prompt
    """
    try:
        # Initialize tokenizer and encode prompt
        enc = tiktoken.get_encoding("gpt2")
        input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
        
        # Move to same device as model
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Generate tokens
        with torch.no_grad():
            generated_tokens = []
            for _ in range(max_new_tokens):
                # Truncate if sequence length exceeds block size
                if input_ids.size(1) > model.config.block_size:
                    input_ids = input_ids[:, -model.config.block_size:]
                
                # Get predictions from model
                logits, _ = model(input_ids)
                logits = logits[:, -1, :]  # Get last token's logits
                
                # Apply temperature scaling
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append the token and continue generating
                generated_tokens.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token), dim=1)

            # Decode the generated tokens
            output_text = prompt + enc.decode(generated_tokens)
            return output_text

    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        return prompt

model = load_model('best_model.pt')

txt = st.text_area(
    "Input Text",
    "The quick brown fox"
)

Max_Tokens = st.slider("Max Tokens", 0, 200, 25, step = 5)
temperature = st.slider("Temperature", 0.0, 1.0, 0.9, step = 0.05)
top_k = st.slider("Top K", 0, 100, 10, step = 5)

if st.button("Generate text"):
	generated_text = generate_text(
	            model=model,
	            prompt=txt.strip(),
	            max_new_tokens=Max_Tokens,
	            temperature=temperature,
	            top_k=top_k
	        )

	st.write("-----------")
	st.write("Output")

	st.write(generated_text)