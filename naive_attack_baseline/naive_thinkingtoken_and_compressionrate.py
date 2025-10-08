import os
import torch
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util

# ====================================
# NLTK Configuration
# ====================================
# Use local or default NLTK path
nltk.data.path.append('/path/to/nltk_data')

# Download required resources (only runs if missing)
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ====================================
# Device Setup
# ====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ====================================
# Load GPT-2 Model
# ====================================
print("🔧 Loading GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained("/path/to/gpt2/")
model = GPT2LMHeadModel.from_pretrained("/path/to/gpt2/").to(device)
model.eval()

# ====================================
# Load Sentence Transformer Model
# ====================================
print("🔧 Loading embedding model...")
embed_model = SentenceTransformer("/path/to/embedding_model/all-MiniLM-L6-v2", device=device)

# ====================================
# Thinking-token seed phrases
# ====================================
thinking_seeds = [
    "Hmm", "Wait", "So", "Therefore", "Now", "Maybe", "Since", "I think",
    "Hold on", "Okay", "appear", "However", "perhaps", "could"
]
seed_embeddings = embed_model.encode(thinking_seeds, convert_to_tensor=True, device=device)

# ====================================
# Function: Extract “thinking tokens”
# ====================================
def extract_thinking_tokens(text, threshold=0.8):
    """
    Identify tokens semantically similar to 'thinking-style' expressions.
    """
    tokens_set = set()
    for sent in sent_tokenize(text):
        words = [w for w in word_tokenize(sent) if w.isalpha()]
        for i in range(len(words)):
            w = words[i]
            if w.lower() not in stop_words:
                tokens_set.add(w.lower())
            if i + 1 < len(words):
                tokens_set.add(f"{words[i].lower()} {words[i+1].lower()}")
            if i + 2 < len(words):
                tokens_set.add(f"{words[i].lower()} {words[i+1].lower()} {words[i+2].lower()}")

    candidates = list(tokens_set)
    if not candidates:
        return Counter()

    candidate_embeddings = embed_model.encode(candidates, convert_to_tensor=True, device=device)
    sim_matrix = util.pytorch_cos_sim(candidate_embeddings, seed_embeddings)

    selected = []
    for i, row in enumerate(sim_matrix):
        max_sim = torch.max(row).item()
        if max_sim >= threshold:
            token = candidates[i]
            print(f"✅ Matched token: '{token}'  |  Similarity: {max_sim:.3f}")
            selected.append(token)

    return Counter(selected)

# ====================================
# Function: Compute Compression via GPT-2 Loss
# ====================================
def compute_compression(text, max_length=1024):
    """
    Compute negative log-likelihood (NLL) as a proxy for text compression.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"].squeeze(0).to(device)
    n_tokens = len(input_ids)
    total_nll = 0.0

    if n_tokens <= max_length:
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), labels=input_ids.unsqueeze(0))
            loss = outputs.loss
        total_nll = loss.item() * n_tokens
    else:
        stride = max_length
        for i in range(0, n_tokens, stride):
            chunk = input_ids[i:i + stride]
            if len(chunk) < 2:
                continue
            with torch.no_grad():
                outputs = model(chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
                loss = outputs.loss
            total_nll += loss.item() * len(chunk)

    return total_nll

# ====================================
# Load Input Excel
# ====================================
input_path = "/path/to/input_file.xlsx"
print(f"📄 Reading input file: {input_path}")
df = pd.read_excel(input_path)

texts = df["Reasoning_path_1"].astype(str)

# ====================================
# Initialize Storage Lists
# ====================================
compression_list = []
thinking_count_list = []

# ====================================
# Main Processing Loop
# ====================================
for text in tqdm(texts, desc="Processing rows"):
    compression = compute_compression(text)
    token_counter = extract_thinking_tokens(text)
    total_thinking_tokens = sum(token_counter.values())

    compression_list.append(compression)
    thinking_count_list.append(total_thinking_tokens)

# ====================================
# Save Results
# ====================================
df["ThinkingTokenCount"] = thinking_count_list
df["CompressionRate"] = compression_list

output_path = "/path/to/output_file.xlsx"
df.to_excel(output_path, index=False)
print(f"✅ Processing complete. Results saved to: {output_path}")