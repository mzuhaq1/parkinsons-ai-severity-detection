import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv("data/features_dataset.csv")

# Remove rows with missing labels
data = data.dropna(subset=["UPDRS_label"])

# Optional: remove rows with missing IDs
data = data.dropna(subset=["ID"])

# -----------------------------
# 2. Keep ID separately
# -----------------------------
ids = data["ID"]

# Features used for model
X = data.drop(columns=["ID", "UPDRS_label", "ML_prediction"], errors="ignore")
y = data["UPDRS_label"]

print("Total samples:", len(data))
print("\nClass distribution:")
print(y.value_counts().sort_index())

print("\nUnique IDs:")
print(ids.nunique())

# -----------------------------
# 3. Group-based split to avoid leakage
#    Same ID will not appear in both train and test
# -----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, test_idx = next(gss.split(X, y, groups=ids))

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

id_train = ids.iloc[train_idx].copy()
id_test = ids.iloc[test_idx].copy()

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining unique IDs:", id_train.nunique())
print("Testing unique IDs:", id_test.nunique())

# -----------------------------
# 4. Save clean split files WITH ID
# -----------------------------
train_data = pd.concat(
    [id_train.reset_index(drop=True),
     X_train.reset_index(drop=True),
     y_train.reset_index(drop=True)],
    axis=1
)
train_data.columns = ["ID"] + list(X.columns) + ["UPDRS_label"]
train_data.to_csv("data/train_data.csv", index=False)

# Test split without prediction first
test_data = pd.concat(
    [id_test.reset_index(drop=True),
     X_test.reset_index(drop=True),
     y_test.reset_index(drop=True)],
    axis=1
)
test_data.columns = ["ID"] + list(X.columns) + ["UPDRS_label"]

# -----------------------------
# 5. Train model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 6. Predict
# -----------------------------
y_pred = model.predict(X_test)

# Add ML prediction to test split
test_data["ML_prediction"] = y_pred
test_data.to_csv("data/test_data.csv", index=False)

# -----------------------------
# 7. Evaluate
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n")
print(cm)

# -----------------------------
# 8. Save model
# -----------------------------
joblib.dump(model, "model/parkinsons_severity_model.pkl")
print("\nModel saved as model/parkinsons_severity_model.pkl")
print("Saved train_data.csv and test_data.csv")

# -----------------------------
# 9. Confusion Matrix Plot
# -----------------------------
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("ML Confusion Matrix")
plt.savefig("results/ml_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# 10. Presentation summary image
# -----------------------------
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
weighted_f1 = report["weighted avg"]["f1-score"]

rows = []
for cls in sorted([k for k in report.keys() if str(k).isdigit()], key=lambda x: int(x)):
    rows.append([f"Class {cls}", f"{report[cls]['f1-score']:.2f}"])

summary_df = pd.DataFrame(rows, columns=["Severity Class", "F1 Score"])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.axis("off")

ax.text(
    0.5, 0.95,
    "Machine Learning Results",
    ha="center", va="top",
    fontsize=18, fontweight="bold",
    transform=ax.transAxes
)

ax.text(
    0.15, 0.78,
    f"Accuracy: {accuracy:.2f}",
    ha="left", va="center",
    fontsize=14, fontweight="bold",
    transform=ax.transAxes
)

ax.text(
    0.15, 0.70,
    f"Weighted F1 Score: {weighted_f1:.2f}",
    ha="left", va="center",
    fontsize=14, fontweight="bold",
    transform=ax.transAxes
)

table = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc="center",
    loc="center",
    bbox=[0.12, 0.15, 0.76, 0.38]
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.4)

plt.savefig("results/ml_results_summary.png", dpi=300, bbox_inches="tight")
plt.show()