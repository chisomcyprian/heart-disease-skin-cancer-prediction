from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/heart.csv")
out = Path("reports/figures")
out.mkdir(parents=True, exist_ok=True)

# Average values for healthy (0) vs disease (1)
cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
print(df.groupby("target")[cols].mean().round(1))

# Which features move with the target?
print("\nCorrelation with target:")
print(df.corr()["target"].drop("target").sort_values().round(2))

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature correlations")
plt.tight_layout()
plt.savefig(out / "correlation.png", dpi=150)
plt.close()

sns.countplot(data=df, x="cp", hue="target")
plt.title("Chest pain type vs heart disease")
plt.savefig(out / "chest_pain.png", dpi=150)
plt.close()

print("\nFigures saved to", out)