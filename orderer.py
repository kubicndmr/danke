import os
import torch
import shutil
import argparse
import pandas as pd

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


surgical_phases = {
    0: 'Vorbereitung', 1: 'Punktion', 2: 'Führungsdraht',
    3: 'Pouchvorbereitung-und-Katheterplatzierung', 4: 'Katheterpositionierung', 5: 'Katheteranpassung',
    6: 'Katheterkontrolle', 7: 'Abschluss'
}


def prefix(id, name='', buffer=5):
    return name + str(id).zfill(buffer)


def listdir(path, ending=None):
    '''Returns dir with full path'''
    if ending == None:
        return sorted([os.path.join(path, f) for f in os.listdir(path)])
    else:
        return sorted([os.path.join(path, f) for f in os.listdir(path)
                       if f.endswith(ending)])


def split_and_copy(from_csv, to_csv):
    df = pd.read_csv(from_csv, index_col=0)
    rows_list = []

    for _, row in df.iterrows():
        sentences = sent_tokenize(row['Text'])
        for sentence in sentences:
            new_row = row.copy()
            new_row['Text'] = sentence
            rows_list.append(new_row)

    df_divided = pd.DataFrame(rows_list)
    df_divided = df_divided.reset_index(drop=True)
    df_divided['Phase_Label'] = df_divided['Phase_Label'].astype(int)
    df_divided.to_csv(to_csv)


def sentence_embeddings(file_name):
    df = pd.read_csv(file_name+'.csv', index_col=0)
    df = df[df['Text'] != '<nicht verstanden>']
    text = df['Text'].tolist()
    df['Embeddings'] = model.encode(
        text, normalize_embeddings=True, convert_to_tensor=True).cpu().tolist()
    df['Phase'] = df['Phase'].map(surgical_phases)
    df.to_pickle(file_name+'.pkl')


def get_files(target_path):
    files_csv = listdir(target_path, '.csv')
    files_txt = listdir(target_path, '.txt')

    # Check number of files
    assert len(files_csv) == len(
        files_txt), f"Existing number of csv and txt files dont match in {target_path}"

    # Check order of files and if they match
    match_flag = 1
    order_flag = 1
    unmatch_files = []
    unorder_files = []
    for i, (c, t) in enumerate(zip(files_csv, files_txt)):
        if not c[:-4] == t[:-4]:
            match_flag = 0
            unmatch_files.append([c, t])

        order_flag_csv = int(c[-9:-4]) == (i+1)
        order_flag_txt = int(t[-9:-4]) == (i+1)
        order_flag *= (order_flag_csv * order_flag_txt)
        if not order_flag_csv:
            unorder_files.append(c)
        if not order_flag_txt:
            unorder_files.append(t)

    assert match_flag == 1, f"following csv and txt file in {target_path} do not match {unmatch_files}"
    assert order_flag == 1, f"following files in {target_path} not ordered properly {unorder_files}"

    return files_csv, files_txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bumbdadumba")

    parser.add_argument('-m', '--mode', type=str,
                        default=None,
                        help='c for create, a for append')

    parser.add_argument('-f', '--from_path', type=str,
                        help='path to copy from')

    parser.add_argument('-t', '--to_path', type=str,
                        help='path to copy')

    parser.add_argument('-s', '--split', type=bool,
                        help='whether to create individual rows for each sentence in the text column')

    parser.add_argument('-e', '--embed', type=bool,
                        help='whether to create sentence embeddings')

    args = parser.parse_args()

    if args.embed:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        model.to(device)

    if args.mode == 'c':
        assert args.from_path != args.to_path, "Both paths cannot be same"
        assert not os.path.exists(
            args.to_path), f"Target path {args.to_path} already exists. Do you want to append instead?"

        os.mkdir(args.to_path)

        files_csv, files_txt = get_files(args.from_path)

        for i, (c, t) in enumerate(zip(files_csv, files_txt)):
            target_file = os.path.join(args.to_path, prefix(i+1, 'SynOP_'))

            print(f"{c} \t-> \t{target_file + '.csv'}")
            print(f"{t} \t-> \t{target_file + '.txt'}")

            if args.split:
                split_and_copy(c, target_file + '.csv')
            else:
                shutil.copy(c, target_file + '.csv')
            shutil.copy(t, target_file + '.txt')

            if args.embed:
                sentence_embeddings(target_file)

    if args.mode == 'a':
        assert args.from_path != args.to_path, "Both paths cannot be same"

        # get file pairs
        to_files_csv, _ = get_files(args.to_path)
        from_files_csv, from_files_txt = get_files(args.from_path)

        last_index = int(to_files_csv[-1][-9:-4])
        input(
            f"Last index of existing files is {last_index}. Just cross-checking, continue?")

        for f_i, t_i in enumerate(range(last_index+1, last_index+len(from_files_csv)+1)):
            target_file = os.path.join(args.to_path, prefix(t_i, 'SynOP_'))

            print(f"{from_files_csv[f_i]} \t-> \t{target_file + '.csv'}")
            print(f"{from_files_txt[f_i]} \t-> \t{target_file + '.txt'}")

            if args.split:
                split_and_copy(from_files_csv[f_i], target_file + '.csv')
            else:
                shutil.copy(from_files_csv[f_i], target_file + '.csv')

            shutil.copy(from_files_txt[f_i], target_file + '.txt')

            if args.embed:
                sentence_embeddings(target_file)
