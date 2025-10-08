import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import ttest_ind

# ====================================
# Evaluation Function
# ====================================
def evaluate_metrics(df, exclude_cols=None, label_col="Label"):
    """
    Evaluate discriminative power of numeric metrics in a DataFrame.

    Computes:
      - AUC
      - Max Balanced Accuracy
      - TPR@5%FPR
      - p-value (t-test)
      - Effect size (Cohen’s d)
    """
    if exclude_cols is None:
        exclude_cols = []

    # Identify metric columns
    metric_cols = [col for col in df.columns if col not in exclude_cols + [label_col]]
    y_true = df[label_col]

    results = []

    for col in metric_cols:
        score = df[col].copy()

        # Reverse score direction for some metrics
        if any(key in col for key in ["CompressionRate", "ThinkingTokenCount", "LLMjudge"]):
            score = -score

        # Handle missing values using group means
        score = score.fillna(df.groupby(label_col)[col].transform("mean"))

        # ROC / AUC metrics
        auc_val = roc_auc_score(y_true, score)
        fpr, tpr, _ = roc_curve(y_true, score)
        balanced_accuracies = (tpr + (1 - fpr)) / 2
        max_bal_acc = np.max(balanced_accuracies)
        tpr_at_5fpr = np.interp(0.05, fpr, tpr)

        # Separate groups
        score_member = score[df[label_col] == 1]
        score_nonmember = score[df[label_col] == 0]

        # Independent t-test (unequal variance)
        _, p_value = ttest_ind(score_member, score_nonmember, equal_var=False)

        # Effect size (Cohen's d)
        mean_diff = score_member.mean() - score_nonmember.mean()
        pooled_std = np.sqrt(((score_member.var(ddof=1) + score_nonmember.var(ddof=1)) / 2))
        cohen_d = mean_diff / pooled_std if pooled_std > 0 else np.nan

        results.append({
            "Metric": col,
            "MaxBalancedAcc": max_bal_acc,
            "AUC": auc_val,
            "TPR@5%FPR": tpr_at_5fpr,
            "p-value": f"{p_value:.3e}",
            "EffectSize(Cohen_d)": cohen_d
        })

    results_df = pd.DataFrame(results)
    return results_df.round(6)

# ====================================
# 1. Load data
# ====================================
train_path = "/path/to/member_data.xlsx"
untrain_path = "/path/to/nonmember_data.xlsx"

df_train = pd.read_excel(train_path)
df_untrain = pd.read_excel(untrain_path)

df_train["Label"] = 1
df_untrain["Label"] = 0
df_all = pd.concat([df_train, df_untrain], ignore_index=True)
print(f"✅ Combined dataset loaded: {len(df_all)} rows")

# ====================================
# 2. Exclude non-metric columns
# ====================================
exclude_cols = [
    "ID", "Item",
    "Reasoning_path_1", 
    "Output_1", "Reasoning_path_2", "Output_2",
    "Reasoning_path_3", "Output_3"
]

# ====================================
# 3. Run evaluation
# ====================================
results_df = evaluate_metrics(df_all, exclude_cols, label_col="Label")

# ====================================
# 4. Display results
# ====================================
print("\n📊 Metric Evaluation Results (Original Order, No Sorting):\n")
print(results_df.to_string(index=False))