import streamlit as st
import torch
import math
import torch.nn as nn

# ---- Model + Helper Definitions (same as your training script) ----

embedding_dim = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load vocab
special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]
with open("input.txt", "r") as f:
    tokenized_lines = f.readlines()

vocab = set()
for sentence in tokenized_lines:
    vocab.update(sentence.lower().split())
vocab = special_tokens + sorted(vocab)

vocab_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_vocab = {idx: word for word, idx in vocab_to_index.items()}
vocab_size = len(vocab)

# Positional encodings
def positional_encodings(seq_len, embedding_dim, device):
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2, device=device).float() * (-math.log(10000.0) / embedding_dim))
    pe = torch.zeros(seq_len, embedding_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def create_padding_mask(input_ids, pad_idx):
    return (input_ids != pad_idx).unsqueeze(1)

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask=None):
        batch_size, seq_len, dim = Q.size()
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device) * float('-inf'), diagonal=1)
        scores = scores + causal_mask
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        weights = self.softmax(scores)
        return torch.matmul(weights, V)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.attn_layers = nn.ModuleList([SelfAttention() for _ in range(16)])
        self.latent_downscale = nn.Linear(embedding_dim, 32)
        self.latent_upscale = nn.Linear(32, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.final_linear_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, Q, K, V, X, attn_mask=None):
        q = self.latent_downscale(Q)
        k = self.latent_downscale(K)
        v = self.latent_downscale(V)
        x = self.latent_downscale(X)
        contexts = [attn(q, k, v, attn_mask) for attn in self.attn_layers]
        combined = torch.cat(contexts, dim=2)
        final_encodings = combined + self.latent_upscale(x)
        final_encodings = self.layer_norm(final_encodings)
        return self.final_linear_layer(final_encodings)

# Load model
embedding_layer = nn.Embedding(vocab_size, embedding_dim).to(device)
model = Model()
model.load_state_dict(torch.load("saved_model.pth", map_location=device))
model.to(device)
model.eval()

def generate_sequence(start_text, max_len=50, temperature=0.7):
    input_ids = [vocab_to_index.get(word, vocab_to_index["<unk>"]) for word in start_text.lower().split()]
    generated = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_len):
        seq_len = generated.size(1)
        pos = positional_encodings(seq_len, embedding_dim, device)
        input_embed = embedding_layer(generated) + pos
        attn_mask = create_padding_mask(generated, vocab_to_index["<pad>"]).to(device)

        with torch.no_grad():
            q = k = v = input_embed
            logits = model(q, k, v, input_embed, attn_mask)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        if index_to_vocab.get(token_id) == "<end>":
            break

        generated = torch.cat((generated, next_token), dim=1)

    return ' '.join([index_to_vocab.get(idx.item(), "<unk>") for idx in generated.squeeze()])

# ---- Streamlit App UI ----

st.set_page_config(page_title="Text Generator", layout="centered")

st.title("📜 Text Generator (Transformer)")
st.markdown("Type a prompt below and generate a continuation using your custom Transformer model.")

prompt = st.text_input("Start text:", value="<start>")
max_length = st.slider("Max length", min_value=10, max_value=100, value=50)
temperature = st.slider("Temperature", 0.1, 2.0, 0.7)

if st.button("Generate"):
    with st.spinner("Generating..."):
        output = generate_sequence(prompt, max_len=max_length, temperature=temperature)
        st.subheader("Generated Text:")
        st.write(output)
