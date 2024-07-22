import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    # Paths
    path_source = 'SynPoCaP/'
    path_target = '/DATA/kubi/SynPoCaP/'
    if not os.path.exists(path_target):
        os.mkdir(path_target)
    
    # find op names and substract already existing ones
    names_exist = [f[:-4] for f in os.listdir(path_target)]
    names_data = [f[:-4] for f in os.listdir(path_source)]
    names_new = list(set(names_data) - set(names_exist))
    
    # convert names to paths
    files_source = sorted([os.path.join(path_source, name+".csv") for name in names_new])
    files_target = sorted([os.path.join(path_target, name+".pkl") for name in names_new])
    
    # get model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    model.to(device)
    
    # iter
    for f_s, f_t in zip(files_source, files_target):
        print(f_s)
        df = pd.read_csv(f_s, index_col=0)
        df = df[df['Text'] != '<nicht verstanden>']
        text = df['Text'].tolist()
        df['Embeddings'] = model.encode(text, normalize_embeddings=True, convert_to_tensor=True).cpu().tolist()
        df.to_pickle(f_t)