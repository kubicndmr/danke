"""
This script adds sentence embeddings to SynPoCaP dataset
"""

import os
import torch
import pandas as pd
pd.options.mode.chained_assignment = None

from pathlib import Path
from sentence_transformers import SentenceTransformer

# Model
embedder_model = SentenceTransformer(
            'intfloat/multilingual-e5-large').to("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Path
    synpocap = Path('/DATA/kubi/Dataset/SynPoCaP/')
    print('Dataset path: ', synpocap)

    # OPs
    ops = synpocap.glob('*.csv')

    # Iter
    for op in ops:
        print('Processing: ', op)
        
        if not os.path.isfile(op.with_suffix('.pkl')):
        
            # Read dataset
            df = pd.read_csv(op, index_col = 0)
            
            # Filter only phyiscian
            df_physician = df[df['Person'] == 'Radiologe']
                        
            # Drop NaNs
            df_physician = df_physician.dropna()
            
            # Check Phases
            try:
                df_physician["Phase_Label"] = pd.to_numeric(df["Phase_Label"]).astype("Int64")
                phases = df_physician["Phase_Label"].unique()
                assert len(phases) == 8
            except:
                print("\tSkipping... ")
                continue
                
            # Add format
            df_physician.loc[:, 'Text'] = df_physician['Text'].apply(lambda x: f'query: {x}')
            
            # Add Embeddings
            embeddings = embedder_model.encode(
                df_physician['Text'].tolist(), 
                normalize_embeddings=True, 
                convert_to_tensor=True
            ).cpu().tolist()
            df_physician['Embeddings'] = embeddings
            
            # Save
            df_physician.to_pickle(op.with_suffix('.pkl'))
            
        else:
            print("\t.pkl file for this OP already exists!")