import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ====================================
# Device Setup
# ====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ====================================
# File Paths
# ====================================
path_generalization = "/.../anchordata/generalization_syn_prefixes_top.xlsx"
path_memorization = "/.../anchordata/memorized_prefixes.xlsx"
path_detect = "/path/to/detection_data.xlsx"
output_path = "/path/to/output_file.xlsx"

# ====================================
# Model Setup (SentenceTransformer)
# ====================================
model_name = "sentence-transformers/all-distilroberta-v1"

# "sentence-transformers/all-mpnet-base-v2"
# "sentence-transformers/all-MiniLM-L6-v2"
#  "sentence-transformers/all-distilroberta-v1"


model = SentenceTransformer(model_name, device=device)
hidden_size = model.get_sentence_embedding_dimension()

# ====================================
# Utility Functions
# ====================================
def encode_text(text, model):
    """Encode text to embedding vector."""
    return model.encode(text, convert_to_numpy=True, device=str(device))

def remove_item_info(R, I):
    """Remove the component of R that aligns with I."""
    if np.linalg.norm(I) < 1e-9:
        return R
    proj = (np.dot(R, I) / np.dot(I, I)) * I
    return R - proj

def get_avg_reasoning_embeddings_with_item_removed(df, model, use_all_paths=True):
    """Compute mean reasoning embedding per row, removing item info."""
    embeddings = []
    texts_combined = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing embeddings"):
        parts, part_embeddings = [], []
        cols = ["Reasoning_path_1", "Reasoning_path_2", "Reasoning_path_3"] if use_all_paths else ["Reasoning_path_2"]

        for col in cols:
            text = row.get(col)
            if pd.notna(text) and isinstance(text, str) and text.strip():
                parts.append(text.strip())
                emb = encode_text(text.strip(), model)
                part_embeddings.append(emb)

        if part_embeddings:
            R = np.mean(part_embeddings, axis=0)
        else:
            R = np.zeros(hidden_size)

        item_text = row.get("Item")
        if pd.notna(item_text) and isinstance(item_text, str) and item_text.strip():
            I = encode_text(item_text.strip(), model)
        else:
            I = np.zeros_like(R)

        R_clean = remove_item_info(R, I)
        embeddings.append(R_clean)
        texts_combined.append(" || ".join(parts))

    return np.array(embeddings), texts_combined

def remove_outliers_by_std(points, z_thresh=2):
    """Remove embedding outliers based on distance from centroid."""
    centroid = np.mean(points, axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    z_scores = (dists - np.mean(dists)) / np.std(dists)
    mask = z_scores < z_thresh
    print(f"🚫 Outlier removal: total {len(points)}, kept {np.sum(mask)}, removed {np.sum(~mask)}")
    return points[mask]

# ====================================
# Compute Embeddings
# ====================================
generalization_embeddings, _ = get_avg_reasoning_embeddings_with_item_removed(
    pd.read_excel(path_generalization), model, use_all_paths=True
)
memorization_embeddings, _ = get_avg_reasoning_embeddings_with_item_removed(
    pd.read_excel(path_memorization), model, use_all_paths=True
)
detect_embeddings, used_texts_combined = get_avg_reasoning_embeddings_with_item_removed(
    pd.read_excel(path_detect), model, use_all_paths=False
)
detect_df = pd.read_excel(path_detect)

# ====================================
# Compute Centers
# ====================================
gen_filtered = remove_outliers_by_std(generalization_embeddings, z_thresh=2.0)
mem_filtered = remove_outliers_by_std(memorization_embeddings, z_thresh=2.0)

gen_center = np.mean(gen_filtered, axis=0)
mem_center = np.mean(mem_filtered, axis=0)
center_unit = (gen_center - mem_center) / np.linalg.norm(gen_center - mem_center)

# ====================================
# Compute Only Two Metrics
# (CenterProjectionFromMemorization & CenterProjectionToGeneralization)
# ====================================
projection_from_mem_center = []
projection_to_gen_center = []

for vec in tqdm(detect_embeddings, desc="Computing projections"):
    projection_from_mem_center.append(np.dot(vec - mem_center, center_unit))
    projection_to_gen_center.append(np.dot(gen_center - vec, center_unit))

# ====================================
# Build Result DataFrame
# ====================================
result_df = detect_df[["ID", "Item"]].copy()
result_df["Reasoning_path_combined"] = used_texts_combined

result_df["CenterProjectionFromMemorization"] = projection_from_mem_center
result_df["CenterProjectionToGeneralization"] = projection_to_gen_center

result_df["ScoreMethod"] = "Center-based projection metrics (Item info removed)"

# ====================================
# Save Results
# ====================================
result_df.to_excel(output_path, index=False)
print(f"✅ Results saved to: {output_path}")