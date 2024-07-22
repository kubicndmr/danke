import os
import shutil
import pandas  as pd

path_source = 's_temp/'
path_target = 'SynPoCaP/'

names_exist = [f[:-4] for f in os.listdir(path_target)]
names_data = [f[:-4] for f in os.listdir(path_source)]
names_new = list(set(names_data) - set(names_exist))

files_source = sorted([os.path.join(path_source, name+".csv") for name in names_new])
files_target = sorted([os.path.join(path_target, name+".csv") for name in names_new])

for f_s, f_t in zip(files_source, files_target):
    df = pd.read_csv(f_s, index_col=0)
    try:
        df['Phase_Label'] = df['Phase_Label'].apply(pd.to_numeric)
        if df['Phase_Label'].between(0, 7).all():
            shutil.copy(f_s, f_t)
    except:
        print('Skipping:\t', f_s)