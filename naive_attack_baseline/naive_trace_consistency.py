import pandas as pd
import Levenshtein
from tqdm import tqdm

# ====================================
# Load Excel file
# ====================================
input_path = "/path/to/input_file.xlsx"
df = pd.read_excel(input_path)
print(f"✅ Successfully loaded data: {len(df)} rows.")

# ====================================
# Character-level edit distance consistency
# ====================================
def compute_edit_distance_consistency_char(paths):
    """
    Compute character-level edit distance consistency across multiple reasoning paths.
    """
    paths = [p for p in paths if pd.notna(p)]
    if len(paths) < 2:
        return 0.0

    sim_scores = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            s1, s2 = paths[i], paths[j]
            max_len = max(len(s1), len(s2))
            if max_len == 0:
                sim = 1.0
            else:
                dist = Levenshtein.distance(s1, s2)
                sim = 1 - dist / max_len
            sim_scores.append(sim)
    return sum(sim_scores) / len(sim_scores)


# ====================================
# Token-level edit distance consistency
# ====================================
def compute_edit_distance_consistency_token(paths):
    """
    Compute token-level edit distance consistency (based on token mapping).
    """
    paths = [p for p in paths if pd.notna(p)]
    if len(paths) < 2:
        return 0.0

    sim_scores = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            tokens1 = paths[i].split()
            tokens2 = paths[j].split()
            max_len = max(len(tokens1), len(tokens2))

            if max_len == 0:
                sim = 1.0
            else:
                # Build token mapping to ensure unique and consistent representation
                vocab = {tok: chr(65 + k) for k, tok in enumerate(set(tokens1 + tokens2))}
                s1 = "".join(vocab[t] for t in tokens1)
                s2 = "".join(vocab[t] for t in tokens2)
                dist = Levenshtein.distance(s1, s2)
                sim = 1 - dist / max_len

            sim_scores.append(sim)
    return sum(sim_scores) / len(sim_scores)


# ====================================
# Compute metrics for each row
# ====================================
char_edit_scores = []
token_edit_scores = []

print("⚙️ Computing character-level and token-level edit distance consistency...")
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    paths = [row.get("Reasoning_path_1"), row.get("Reasoning_path_2"), row.get("Reasoning_path_3")] # or more reasoning paths
    char_edit_scores.append(compute_edit_distance_consistency_char(paths))
    token_edit_scores.append(compute_edit_distance_consistency_token(paths))

# ====================================
# Store results in DataFrame
# ====================================
df["CharLevel_EditConsistency"] = char_edit_scores
df["TokenLevel_EditConsistency"] = token_edit_scores

# ====================================
# Save results
# ====================================
output_path = "/path/to/output_file.xlsx"
df.to_excel(output_path, index=False)
print(f"✅ Processing complete. Results saved to: {output_path}")