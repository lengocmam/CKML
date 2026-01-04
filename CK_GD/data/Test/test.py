import pandas as pd

df = pd.read_csv("heart_2020_processed.csv")

df_20 = df.sample(frac=0.2, random_state=42)
df_20.to_csv("output_20_percent.csv", index=False)

print("Đã xuất 20% dữ liệu ra output_20_percent.csv")
