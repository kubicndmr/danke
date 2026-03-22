import os
import pandas as pd
from pathlib import Path

dataset_path = Path("TextualSPR-Dataset/SynPoCaP")

for f in dataset_path.glob("*.csv"):
    df = pd.read_csv(f, index_col=0)
    df_clean = df.dropna()

    if len(df) != len(df_clean):
        print(f"{f.name}: {len(df)} -> {len(df_clean)}")
        df_clean.to_csv(f)