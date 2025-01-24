# Trainibg Transformer from scratch

- Model Params: ~124 Million
- Final Model Loss: 2.18e-06
- Target Loss: 0.099

## Trianing Log is pasted below
![alt text](image.png)


# Transformer and Multi-Head Attention

## Transformer Overview
The Transformer is a groundbreaking architecture introduced in the "Attention is All You Need" paper. This project focuses on a **decoder-only Transformer** architecture, which is widely used in autoregressive language models like GPT. This simplified architecture generates sequences by predicting the next token based on previously generated tokens.

### Key Components of the Decoder-Only Architecture:
1. **Self-Attention Mechanism**: Enables tokens to attend to earlier tokens in the sequence.
2. **Positional Encodings**: Adds information about token order to embeddings, as Transformers are inherently order-agnostic.
3. **Feedforward Layers**: Applies fully connected layers to refine token representations.
4. **Causal Masking**: Ensures the model attends only to preceding tokens during training, preserving the autoregressive property.

## Multi-Head Attention
Multi-Head Attention (MHA) is a crucial element of the decoder architecture. It allows the model to learn multiple types of relationships between tokens by employing multiple attention "heads."

### Steps in Multi-Head Attention:
1. **Linear Transformations**: Input embeddings are projected into query (Q), key (K), and value (V) vectors.
2. **Scaled Dot-Product Attention**: Attention scores are computed using:
   \[
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
   \]
   where \(d_k\) is the dimension of each key vector.
3. **Causal Masking**: During training, a triangular mask is applied to prevent future token predictions.
4. **Concatenation and Projection**: Outputs from all attention heads are concatenated and passed through a linear layer.

### Benefits of Multi-Head Attention:
- Allows the model to capture diverse relationships between tokens.
- Improves learning efficiency for long-range dependencies.

# Deployed this in streamlit app

Attached streamlit UI below
![alt text](image-1.png)"# TransformerFromScratch" 
