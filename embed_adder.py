import os
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

for df_path in utils.listdir('Data'):
    print(df_path)
    df = pd.read_pickle(df_path)
    df['ShuffledEmbeddings'] = df['ShuffledSentences'].apply(get_embedding)
    df.to_pickle(df_path)