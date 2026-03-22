"""
This script adds sentence embeddings to SynPoCaP dataset
"""
import os
import dotenv
dotenv.load_dotenv()
os.environ["HF_HOME"] = os.getenv("HF_HOME")

import torch
import pandas as pd
pd.options.mode.chained_assignment = None

from pathlib import Path
from transformers import AutoTokenizer
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

LLM = "e5-large"

if __name__ == "__main__":
    # Path
    synpocap = Path('TextualSPR-Dataset/SynPoCaP')
    print('Dataset path: ', synpocap)

    # Tokenizer
    if LLM == "bert":
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    elif LLM == "e5-large":
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    else:
        raise NotImplementedError

    # Iter
    counter = 0
    for op in synpocap.glob('*.csv'):
        
        if not os.path.isfile(op.with_suffix('.pkl')):
            print('Processing: ', op)
        
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
                counter += 1
                continue
            
            # Remove transition label
            df = df[df['Phase_Label'] != 8].reset_index(drop=True)

            # Add format
            df_physician.loc[:, 'Text'] = df_physician['Text'].apply(lambda x: f'query: {x}')
            
            # Tokenize text
            inputs = tokenizer(
                df_physician['Text'].tolist(),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt',
                return_attention_mask=True
            )
            df_physician["input_ids"] = inputs['input_ids'].tolist()
            df_physician["attention_mask"] = inputs['attention_mask'].tolist()

            # Save
            df_physician.to_pickle(op.with_suffix('.pkl'))
        
            
        else:
            print(f"\t.pkl file for '{op}' already exists!")
        
    print(f"\tDone! {counter} files skipped due to missing phases.")