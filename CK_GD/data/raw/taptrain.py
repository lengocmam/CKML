import numpy as np
import pandas as pd


INPUT_FILE = "heart_2020_processed.csv"   
OUTPUT_FILE = "train_balanced.csv"

SEED = 42
MAX_PER_CLASS = 20000

rng = np.random.default_rng(SEED)

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["HeartDisease"]).values
y = df["HeartDisease"].values

y = np.where(y == 1, 1, -1)

print("Original data shape:", X.shape)
print("Original class distribution:",
      "Pos =", np.sum(y == 1),
      "| Neg =", np.sum(y == -1))


pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == -1)[0]

pos_pick = rng.choice(
    pos_idx,
    size=min(MAX_PER_CLASS, len(pos_idx)),
    replace=False
)

neg_pick = rng.choice(
    neg_idx,
    size=min(MAX_PER_CLASS, len(neg_idx)),
    replace=False
)

pick_idx = np.concatenate([pos_pick, neg_pick])
rng.shuffle(pick_idx)

X_balanced = X[pick_idx]
y_balanced = y[pick_idx]

print("\nAfter Random Undersampling:")
print("Balanced shape:", X_balanced.shape)
print("Pos =", np.sum(y_balanced == 1),
      "| Neg =", np.sum(y_balanced == -1))


df_balanced = pd.DataFrame(
    X_balanced,
    columns=df.drop(columns=["HeartDisease"]).columns
)

df_balanced.insert(
    0,
    "HeartDisease",
    np.where(y_balanced == 1, 1, 0)  
)

df_balanced.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ DONE. Exported file: {OUTPUT_FILE}")
print("Final shape:", df_balanced.shape)
