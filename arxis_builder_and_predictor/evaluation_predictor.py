import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import ttest_ind

# ====================================
# 1. Load datasets
# ====================================
path_train = "/path/to/member_dataset.xlsx"
path_untrain = "/path/to/nonmember_dataset.xlsx"

df_train = pd.read_excel(path_train)
df_untrain = pd.read_excel(path_untrain)

print(f"✅ Loaded {len(df_train) + len(df_untrain)} total samples.")

df_train["Label"] = 1  # Member samples
df_untrain["Label"] = 0  # Non-member samples
df_all = pd.concat([df_train, df_untrain], ignore_index=True)

# ====================================
# 2. Identify metric columns
# ====================================
exclude_cols = [
    "ID", "Item", "Reasoning_Path_1", "Reasoning_Path_2",
    "Reasoning_Path_3", "ScoreMethod", "Label"
]
metric_cols = [col for col in df_all.columns if col not in exclude_cols]

# ====================================
# 3. Evaluate each metric
# ====================================
results = []

for col in metric_cols:
    # Determine direction based on name
    if "Generalization" in col:
        score = df_all[col]
        direction = "+"
    elif "Memorization" in col:
        score = -df_all[col]
        direction = "-"
    else:
        continue

    # Fill missing values with group-wise mean
    score = score.fillna(df_all.groupby("Label")[col].transform("mean"))
    y_true = df_all["Label"]

    # Compute metrics
    auc_val = roc_auc_score(y_true, score)
    fpr, tpr, _ = roc_curve(y_true, score)
    balanced_accuracies = (tpr + (1 - fpr)) / 2
    max_bal_acc = np.max(balanced_accuracies)
    tpr_at_5fpr = np.interp(0.05, fpr, tpr)

    # Statistical significance test
    score_member = score[df_all["Label"] == 1]
    score_nonmember = score[df_all["Label"] == 0]
    _, p_value = ttest_ind(score_member, score_nonmember, equal_var=False)

    # Effect size (Cohen’s d)
    mean_diff = score_member.mean() - score_nonmember.mean()
    pooled_std = np.sqrt((score_member.var(ddof=1) + score_nonmember.var(ddof=1)) / 2)
    cohen_d = mean_diff / pooled_std if pooled_std > 0 else np.nan

    results.append({
        "Metric": col,
        "Direction": direction,
        "AUC": auc_val,
        "MaxBalancedAcc": max_bal_acc,
        "TPR@5%FPR": tpr_at_5fpr,
        "p-value": p_value,
        "EffectSize(Cohen_d)": cohen_d
    })

# ====================================
# 4. Format and print results
# ====================================
results_df = pd.DataFrame(results)

# Format p-values in scientific notation (3 significant digits)
results_df["p-value"] = results_df["p-value"].apply(
    lambda x: f"{x:.3e}" if pd.notnull(x) else "NaN"
)

# Round other numeric columns to 3 decimals
for col in ["AUC", "MaxBalancedAcc", "TPR@5%FPR", "EffectSize(Cohen_d)"]:
    results_df[col] = results_df[col].apply(lambda x: round(x, 3) if pd.notnull(x) else np.nan)

print("\n📊 Metric evaluation results (unsorted):\n")
print(results_df.to_string(index=False, float_format="%.3f"))