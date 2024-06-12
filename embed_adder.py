import sys
import torch
import utils
import pandas as pd
from sentence_transformers import SentenceTransformer

# Sentence Transformer
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
model.to(device)

def get_embedding(sentences):
    return model.encode(sentences, convert_to_tensor=True).T

for df_path in utils.listdir('SynPoCaP/', ending='.csv'):
    print(df_path[:-3]+'pkl')
    df = pd.read_csv(df_path)
    print(df)
    df['Embeddings'] = df['Text'].apply(get_embedding)
    df.to_pickle(df_path[:-3]+'pkl')