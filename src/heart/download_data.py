from pathlib import Path
from ucimlrepo import fetch_ucirepo

# Fetch the UCI Heart Disease dataset (id 45)
heart = fetch_ucirepo(id=45)
df = heart.data.features.copy()

# Raw target is 0-4; convert to 0 = no disease, 1 = disease
df["target"] = (heart.data.targets["num"] > 0).astype(int)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/heart.csv", index=False)

print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isna().sum())
print("\nTarget counts:\n", df["target"].value_counts())