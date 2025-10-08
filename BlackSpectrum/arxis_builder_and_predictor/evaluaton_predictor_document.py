import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import ttest_ind

# ====================================
# 1. Load data
# ====================================
train_path = "/path/to/member_data.xlsx"
untrain_path = "/path/to/nonmember_data.xlsx"

df_train = pd.read_excel(train_path)
df_untrain = pd.read_excel(untrain_path)

# Assign binary labels
df_train["Label"] = 1
df_untrain["Label"] = 0

# Combine both datasets
df_all = pd.concat([df_train, df_untrain], ignore_index=True)
print(f"✅ Loaded combined dataset: {len(df_all)} rows")

# ====================================
# 2. Aggregate numeric metrics by ID and Label
# ====================================
numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
agg_cols = [col for col in numeric_cols if col != "Label"]

agg_df = df_all.groupby(["ID", "Label"])[agg_cols].mean().reset_index()

# ====================================
# 3. Identify metric columns (exclude ID & Label)
# ====================================
exclude_cols = ["ID", "Label"]
metric_cols = [col for col in agg_df.columns if col not in exclude_cols]

# ====================================
# 4. Evaluate each metric column
# ====================================
results = []

for col in metric_cols:
    # Define direction for scoring
    if "Generalization" in col:
        score = agg_df[col]
        direction = "+"
    elif "Memorization" in col:
        score = -agg_df[col]
        direction = "-"
    else:
        continue

    # Handle missing values
    score = score.fillna(agg_df.groupby("Label")[col].transform("mean"))
    y_true = agg_df["Label"]

    # Compute evaluation metrics
    auc_val = roc_auc_score(y_true, score)
    fpr, tpr, _ = roc_curve(y_true, score)
    balanced_accuracies = (tpr + (1 - fpr)) / 2
    max_bal_acc = np.max(balanced_accuracies)
    tpr_at_5fpr = np.interp(0.05, fpr, tpr)

    # Statistical significance (two-sample t-test)
    score_member = score[y_true == 1]
    score_nonmember = score[y_true == 0]
    _, p_value = ttest_ind(score_member, score_nonmember, equal_var=False)

    results.append({
        "Metric": col,
        "Direction": direction,
        "AUC": auc_val,
        "MaxBalancedAcc": max_bal_acc,
        "TPR@5%FPR": tpr_at_5fpr,
        "p-value": p_value
    })

# ====================================
# 5. Display results
# ====================================
results_df = pd.DataFrame(results)
print("\n📊 Metric evaluation results (aggregated by ID):\n")
print(results_df.round(6).to_string(index=False))