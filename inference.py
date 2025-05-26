import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

embedding_dim = 512

tokenized_lines = open("input.txt", "r")
tokenized_lines = tokenized_lines.readlines()

vocab = set()
special_tokens = ["<pad>", "<start>", "<end>"]
for sentence in tokenized_lines:
    vocab.update(sentence.split())
vocab = special_tokens + list(vocab)

vocab_to_index = {word:index for index, word in enumerate(vocab)}
vocab_size = len(vocab)

from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = "<pad>"
PAD_IDX = vocab_to_index[PAD_TOKEN]

special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]

vocab_to_index = {}

vocab = set()
for sentence in tokenized_lines:
    vocab.update(sentence.lower().split())      # lowercase here

vocab = special_tokens + sorted(vocab)          # sorted for reproducibility
vocab_to_index = {w:i for i,w in enumerate(vocab)}

PAD_IDX = vocab_to_index["<pad>"]
UNK_IDX = vocab_to_index["<unk>"]

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_layer = nn.Embedding(vocab_size, embedding_dim).to(device)

def positional_encodings(seq_len, embedding_dim, device):
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2, device=device).float() * (-math.log(10000.0) / embedding_dim))
    pe = torch.zeros(seq_len, embedding_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def create_padding_mask(input_ids, pad_idx):
    input_ids: (batch, seq_len)
    return (input_ids != pad_idx).unsqueeze(1)

class self_attention(nn.Module):
    def __init__(self):
        super(self_attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask=None):
        # Q, K, V shape: (batch, seq_len, dim)
        batch_size, seq_len, dim = Q.size()

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)  # (batch, seq_len, seq_len)

        # Causal mask (upper triangular)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device) * float('-inf'), diagonal=1)
        scores = scores + causal_mask

        # Padding mask (optional)
        if attn_mask is not None:
            # attn_mask: (batch, 1, seq_len), 1 for keep, 0 for mask
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        weights = self.softmax(scores)
        context = torch.matmul(weights, V)  # (batch, seq_len, dim)

        return context

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # creating the multi-headed attention block.
        self.self_attn1 = self_attention()
        self.self_attn2 = self_attention()
        self.self_attn3 = self_attention()
        self.self_attn4 = self_attention()
        self.self_attn5 = self_attention()
        self.self_attn6 = self_attention()
        self.self_attn7 = self_attention()
        self.self_attn8 = self_attention()

        self.self_attn9 = self_attention()
        self.self_attn10 = self_attention()
        self.self_attn11 = self_attention()
        self.self_attn12 = self_attention()
        self.self_attn13 = self_attention()
        self.self_attn14 = self_attention()
        self.self_attn15 = self_attention()
        self.self_attn16 = self_attention()


        # All the layers, we gonna need to make the decoder work.
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.softmax = nn.Softmax(-1)
        
        self.latent_downscale = nn.Linear(embedding_dim, 32)
        self.latent_upscale = nn.Linear(32, embedding_dim)

        self.final_linear_layer = nn.Linear(embedding_dim, vocab_size)


    def forward(self, Q, K, V, X, attn_mask=None):
        q = self.latent_downscale(Q)
        k = self.latent_downscale(K)
        v = self.latent_downscale(V)
        x = self.latent_downscale(X)

        context1 = self.self_attn1(q, k, v, attn_mask)
        context2 = self.self_attn2(q, k, v, attn_mask)
        context3 = self.self_attn3(q, k, v, attn_mask)
        context4 = self.self_attn4(q, k, v, attn_mask)
        context5 = self.self_attn5(q, k, v, attn_mask)
        context6 = self.self_attn6(q, k, v, attn_mask)
        context7 = self.self_attn7(q, k, v, attn_mask)
        context8 = self.self_attn8(q, k, v, attn_mask)

        context9 = self.self_attn1(q, k, v, attn_mask)
        context10 = self.self_attn2(q, k, v, attn_mask)
        context11 = self.self_attn3(q, k, v, attn_mask)
        context12 = self.self_attn4(q, k, v, attn_mask)
        context13 = self.self_attn5(q, k, v, attn_mask)
        context14 = self.self_attn6(q, k, v, attn_mask)
        context15 = self.self_attn7(q, k, v, attn_mask)
        context16 = self.self_attn8(q, k, v, attn_mask)

        combined = torch.cat((context1, context2, context3, context4, context5, context6, context7, context8, context9, context10, context11, context12, context13, context14, context15, context16), 2)
        final_encodings = combined + self.latent_upscale(x)
        final_encodings = self.layer_norm(final_encodings)
        logits = self.final_linear_layer(final_encodings)

        return logits

inference_model = Model()
inference_model.load_state_dict(torch.load("saved_model.pth", map_location=device))
inference_model.to(device)
inference_model.eval()

import torch

def generate_sequence(model, start_text, vocab_to_idx, idx_to_vocab, embedding_layer, device, max_len=50):
    model.eval()  # Setting the model to evaluation mode
    start_tokens = start_text.lower().split()

    # Convert words to indices
    input_ids = [vocab_to_idx.get(word, vocab_to_idx["<pad>"]) for word in start_tokens]
    generated = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)

    for _ in range(max_len):
        seq_len = generated.size(1)

        # Recalculate positional encodings each time
        pos = positional_encodings(seq_len, embedding_layer.embedding_dim, device)
        input_embed = embedding_layer(generated) + pos

        # Attention mask
        attn_mask = create_padding_mask(generated, vocab_to_idx["<pad>"]).to(device)

        with torch.no_grad():
            q = k = v = input_embed
            logits = model(q, k, v, input_embed, attn_mask)

        # Sample next token
        logits = logits[:, -1, :]  # Get last token's logits
        temperature = 0.7 # I added a temperature hyperparameter to check for repitition.
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: (1, 1)

        # Stop if end token
        token_id = next_token.item()
        if idx_to_vocab.get(token_id, "") == "<end>":
            break

        # Append next token
        generated = torch.cat((generated, next_token), dim=1)

    # Convert generated indices back to words
    generated_text = ' '.join([idx_to_vocab.get(idx.item(), "<unk>") for idx in generated.squeeze()])
    return generated_text

# Example of inference usage:
start_text = "<start>"  # Starting text for generation
generated_text = generate_sequence(
    model=inference_model, 
    start_text=start_text, 
    vocab_to_idx=vocab_to_index, 
    idx_to_vocab={index: word for word, index in vocab_to_index.items()}, 
    embedding_layer=embedding_layer, 
    device=device,
    max_len=50  # Limit generated sequence length
)

print("Generated Text:")
print(generated_text)