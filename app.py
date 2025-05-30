from flask import Flask, render_template, request, jsonify
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

app = Flask(__name__)

# Your original model code
embedding_dim = 512

# Load tokenized lines
try:
    with open("input.txt", "r") as f:
        tokenized_lines = f.readlines()
except FileNotFoundError:
    # Fallback data if input.txt doesn't exist
    tokenized_lines = ["hello world this is a test", "artificial intelligence is fascinating", "transformers are powerful models"]

vocab = set()
special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]
for sentence in tokenized_lines:
    vocab.update(sentence.lower().split())

vocab = special_tokens + sorted(vocab)
vocab_to_index = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

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
    return (input_ids != pad_idx).unsqueeze(1)

class self_attention(nn.Module):
    def __init__(self):
        super(self_attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask=None):
        batch_size, seq_len, dim = Q.size()
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
        
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device) * float('-inf'), diagonal=1)
        scores = scores + causal_mask

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        weights = self.softmax(scores)
        context = torch.matmul(weights, V)
        return context

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Your 16 attention heads
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

        # Note: You had a bug here - using the same attention heads again
        context9 = self.self_attn9(q, k, v, attn_mask)
        context10 = self.self_attn10(q, k, v, attn_mask)
        context11 = self.self_attn11(q, k, v, attn_mask)
        context12 = self.self_attn12(q, k, v, attn_mask)
        context13 = self.self_attn13(q, k, v, attn_mask)
        context14 = self.self_attn14(q, k, v, attn_mask)
        context15 = self.self_attn15(q, k, v, attn_mask)
        context16 = self.self_attn16(q, k, v, attn_mask)

        combined = torch.cat((context1, context2, context3, context4, context5, context6, context7, context8, 
                             context9, context10, context11, context12, context13, context14, context15, context16), 2)
        final_encodings = combined + self.latent_upscale(x)
        final_encodings = self.layer_norm(final_encodings)
        logits = self.final_linear_layer(final_encodings)
        return logits

# Initialize model
inference_model = Model()

# Try to load saved model, otherwise use random weights
try:
    inference_model.load_state_dict(torch.load("saved_model.pth", map_location=device))
    print("Loaded saved model weights")
except FileNotFoundError:
    print("No saved model found, using random weights")

inference_model.to(device)
inference_model.eval()

def generate_sequence(model, start_text, vocab_to_idx, idx_to_vocab, embedding_layer, device, max_len=50, temperature=0.7):
    model.eval()
    start_tokens = start_text.lower().split()

    input_ids = [vocab_to_idx.get(word, vocab_to_idx["<unk>"]) for word in start_tokens]
    generated = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for wop in range(max_len):
        seq_len = generated.size(1)
        pos = positional_encodings(seq_len, embedding_layer.embedding_dim, device)
        input_embed = embedding_layer(generated) + pos
        attn_mask = create_padding_mask(generated, vocab_to_idx["<pad>"]).to(device)

        with torch.no_grad():
            q = k = v = input_embed
            logits = model(q, k, v, input_embed, attn_mask)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        if idx_to_vocab.get(token_id, "") == "<end>":
            break

        generated = torch.cat((generated, next_token), dim=1)

    generated_text = ' '.join([idx_to_vocab.get(idx.item(), "<unk>") for idx in generated.squeeze()])
    return generated_text, wop

# Create reverse vocabulary mapping
idx_to_vocab = {index: word for word, index in vocab_to_index.items()}

@app.route('/')
def index():
    return render_template('index.html', vocab_size=vocab_size, device=device)

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.json
        start_text = data.get('start_text', '<start>')
        max_length = int(data.get('max_length', 50))
        temperature = float(data.get('temperature', 0.7))
        
        # Clamp values for safety
        max_length = min(max(max_length, 1), 200)
        temperature = min(max(temperature, 0.1), 2.0)
        
        generated_text, word_count = generate_sequence(
            model=inference_model,
            start_text=start_text,
            vocab_to_idx=vocab_to_index,
            idx_to_vocab=idx_to_vocab,
            embedding_layer=embedding_layer,
            device=device,
            max_len=max_length,
            temperature=temperature
        )
        
        return jsonify({
            'success': True,
            'generated_text': generated_text,
            'word_count': word_count,
            'vocab_size': vocab_size,
            'device': str(device)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/model_info')
def model_info():
    return jsonify({
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'device': str(device),
        'attention_heads': 16,
        'special_tokens': special_tokens,
        'sample_vocab': list(vocab_to_index.keys())[:20]
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Write the HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Transformer Model - Live Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2196F3, #21CBF3);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; font-weight: 300; }
        .header p { opacity: 0.9; font-size: 1.1em; }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }
        .input-panel, .output-panel {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            border: 1px solid #e9ecef;
        }
        .section-title {
            font-size: 1.4em;
            color: #2c3e50;
            margin-bottom: 20px;
            font-weight: 600;
        }
        .form-group { margin-bottom: 20px; }
        label {
            display: block;
            margin-bottom: 8px;
            color: #495057;
            font-weight: 500;
        }
        input[type="text"], input[type="number"], input[type="range"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        input:focus {
            outline: none;
            border-color: #2196F3;
            box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1);
        }
        .slider-container { position: relative; }
        input[type="range"] {
            appearance: none;
            height: 8px;
            background: #ddd;
            border-radius: 4px;
        }
        input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            background: #2196F3;
            border-radius: 50%;
            cursor: pointer;
        }
        .slider-value {
            position: absolute;
            right: 0;
            top: -25px;
            color: #666;
            font-weight: 500;
        }
        .generate-btn {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 1.1em;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            font-weight: 600;
        }
        .generate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
        }
        .output-text {
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            min-height: 200px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        .stat-number {
            font-size: 1.5em;
            font-weight: bold;
            color: #2196F3;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .model-info {
            background: #e8f4fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 0 10px 10px 0;
        }
        .loading { display: none; text-align: center; padding: 20px; color: #666; }
        .loading.active { display: block; }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #2196F3;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { color: #dc3545; background: #f8d7da; padding: 10px; border-radius: 5px; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Your Transformer Model</h1>
            <p>Running on {{ device }} • Vocab Size: {{ vocab_size }} • 16 Attention Heads</p>
        </div>

        <div class="main-content">
            <div class="input-panel">
                <div class="section-title">⚙️ Model Controls</div>
                
                <div class="model-info">
                    <h4>Live PyTorch Model</h4>
                    <p>Your actual transformer with 16-head attention, 512-dim embeddings, and causal masking.</p>
                </div>

                <div class="form-group">
                    <label for="startText">Start Text:</label>
                    <input type="text" id="startText" value="<start>" placeholder="Enter starting text...">
                </div>

                <div class="form-group">
                    <label for="maxLength">Max Length:</label>
                    <input type="number" id="maxLength" value="50" min="1" max="200">
                </div>

                <div class="form-group">
                    <label for="temperature">Temperature: <span class="slider-value" id="tempValue">0.7</span></label>
                    <div class="slider-container">
                        <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="0.7">
                    </div>
                </div>

                <button class="generate-btn" onclick="generateText()">
                    🚀 Generate with Your Model
                </button>
            </div>

            <div class="output-panel">
                <div class="section-title">📝 Model Output</div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Running inference on your transformer...</p>
                </div>

                <div class="output-text" id="outputText">
                    Click "Generate" to see your model's output...
                </div>

                <div id="errorDiv"></div>

                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number" id="wordCount">0</div>
                        <div class="stat-label">Words Generated</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="vocabSize">{{ vocab_size }}</div>
                        <div class="stat-label">Vocabulary Size</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="deviceInfo">{{ device.upper() }}</div>
                        <div class="stat-label">Device</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('temperature').addEventListener('input', function() {
            document.getElementById('tempValue').textContent = this.value;
        });

        async function generateText() {
            const startText = document.getElementById('startText').value || '<start>';
            const maxLength = parseInt(document.getElementById('maxLength').value) || 50;
            const temperature = parseFloat(document.getElementById('temperature').value) || 0.7;
            
            document.getElementById('loading').classList.add('active');
            document.getElementById('outputText').style.display = 'none';
            document.getElementById('errorDiv').innerHTML = '';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        start_text: startText,
                        max_length: maxLength,
                        temperature: temperature
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('outputText').textContent = result.generated_text;
                    document.getElementById('wordCount').textContent = result.word_count;
                } else {
                    document.getElementById('errorDiv').innerHTML = 
                        '<div class="error">Error: ' + result.error + '</div>';
                }
                
            } catch (error) {
                document.getElementById('errorDiv').innerHTML = 
                    '<div class="error">Network error: ' + error.message + '</div>';
            }
            
            document.getElementById('loading').classList.remove('active');
            document.getElementById('outputText').style.display = 'block';
        }
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w') as f:
        f.write(html_template)
    
    print(f"🚀 Starting Flask app...")
    print(f"📊 Vocab size: {vocab_size}")
    print(f"🔥 Device: {device}")
    print(f"🌐 Open http://localhost:5000 to use your model!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)