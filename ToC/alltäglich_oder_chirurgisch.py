import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import huggingface_hub
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def get_answer(tokenizer, language_model, messages, max_new_tokens,
               do_sample=True, top_p=0.95, temperature=1, repetition_penalty=1.1):
    # Prepare the prompt from the chat history
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    # Encode the prompt to input tensor
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs['input_ids'] = inputs['input_ids'].to(language_model.device)

    # Generate the model's response
    outputs = language_model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )

    # Decode the model's output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


def get_tagged_block(text, start_tag, end_tag):
    """
    Extracts and returns the last block of text that is enclosed between specified start and end tags.

    Args:
        text (str): The input text from which to extract the block.
        start_tag (str): The tag that marks the beginning of the block.
        end_tag (str): The tag that marks the end of the block.

    Returns:
        list: A list of lines that are between the start_tag and end_tag.
    """
    block = []
    tag_flag = False
    for line in text.splitlines():
        line = line.strip()
        if start_tag in line:
            index = start_tag.find(line)
            if index != -1:
                line = line[index + len(start_tag):]
            block = []
            tag_flag = True
            continue
        if end_tag in line:
            index = end_tag.find(line)
            if index == -1:
                line = line[:len(end_tag)-1]
                block.append(line)
            tag_flag = False
        if tag_flag:
            if line != '':
                block.append(line)
    return block


def block_to_df(block):
    """
    Converts a list of strings into a pandas DataFrame

    Args:
        block   : list 
                    A list of strings where the first item contains column names

    Returns:
        df      : pandas.DataFrame
                    A DataFrame of the text
    """
    columns = block[0].split(';')

    rows = []
    for line in block[1:]:
        rows.append(line.split(';'))

    df = pd.DataFrame(rows, columns=columns)

    df[columns[0]] = df[columns[0]].apply(pd.to_numeric)
    df.set_index(columns[0], inplace=True)

    return df


def split_dataframe(df, n):
    """
    Splits a DataFrame into n smaller DataFrames.

    Parameters:
        df (pd.DataFrame): The original DataFrame to be split.
        n (int): The number of smaller DataFrames to return.

    Returns:
        list of pd.DataFrame: A list containing the smaller DataFrames.
    """
    # Calculate the size of each split
    split_size = len(df) // n
    remainder = len(df) % n

    # Create a list to hold the smaller DataFrames
    split_dfs = []

    start_idx = 0

    for i in range(n):
        end_idx = start_idx + split_size + (1 if i < remainder else 0)
        split_dfs.append(df.iloc[start_idx:end_idx])
        start_idx = end_idx

    return split_dfs


if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generation for SPR")

    parser.add_argument('-t', '--target_path', type=str,
                        default='ToC/Transcripts_ToC/',
                        help='path to save labeled data')

    args = parser.parse_args()

    # Target folder
    if not os.path.exists(args.target_path):
        os.mkdir(args.target_path)

    # Login huggingface environment
    load_dotenv()
    huggingface_hub.login(os.getenv('HF_TOKEN'), add_to_git_credential=False)

    # Model
    model_id = 'mistralai/Mistral-Large-Instruct-2407'
    auto_tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=os.getenv('HF_CACHE_DIR'))
    auto_language_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        cache_dir=os.getenv('HF_CACHE_DIR')
    )

    # Data
    data_path = 'Transcripts/'
    trainset = ['OP_040.csv', 'OP_005.csv', 'OP_023.csv', 'OP_027.csv', 'OP_032.csv', 'OP_035.csv', 'OP_017.csv', 'OP_002.csv', 'OP_006.csv', 'OP_038.csv',
                'OP_011.csv', 'OP_022.csv', 'OP_007.csv', 'OP_019.csv', 'OP_013.csv', 'OP_024.csv', 'OP_039.csv', 'OP_026.csv', 'OP_016.csv', 'OP_009.csv']

    # Compute labels
    role = """Du bist ein hilfsbereites Assistent, das die Gespräche der Chirurgen analysiert."""
    for op in trainset:
        print(op)
        df = pd.read_csv(data_path+op, index_col=0)
        df = df[df['Text'] != '<nicht verstanden>']
        df = df.drop(columns=['File_Name', 'End_Time', 'Phase_Label'])
        df_print = df.copy()

        dfs_to_concat = []

        for i, sub_df in enumerate(split_dataframe(df_print, 5)):
            print(f"\tStep : {i+1}")
            prompt = """Du hast einen Datensatz mit Gesprächen, die von einem Chirurgen während einer Portkatheter-Platzierungsoperation aufgezeichnet wurden. Deine Aufgabe ist es, jeden Satz in der Spalte 'Text' zu überprüfen und die Spalte 'Bezeichnung' auszufüllen. In der Spalte 'Bezeichnung', markiere die Sätze mit 'C', wenn sie für den chirurgischen Ablauf der Operation relevant sind, und mit 'T', wenn sie sich auf alltägliche Gespräche beziehen. Gib deine Antwort im CSV format, verwende die Vorlage im Abschnitt <Antwort 1> in die mit "Ausfüllen" markierte Spalte "Bezeichnung" ein. Verwende immer die Tags <Antwort 1> und </Antwort 1> am Anfang und Ende deiner Antwort."""
            prompt += f"\n<Daten>\n{sub_df.to_csv(index=True, sep=';', index_label='Index')}</Daten>\n"
            sub_df = sub_df.drop(columns=['Start_Time', 'Text'])
            sub_df['Bezeichnung'] = "*Ausfüllen*"
            prompt += f"\n<Antwort 1>\n{sub_df.to_csv(index=True, sep=';', index_label='Index')}</Antwort 1>\n"

            messages = [{"role": "system", "content": role},
                        {"role": "user", "content": prompt}]

            answer = get_answer(
                auto_tokenizer, auto_language_model, messages, max_new_tokens=500)

            block_answer = get_tagged_block(
                answer, '<Antwort 1>', '</Antwort 1>')

            dfs_to_concat.append(block_to_df(block_answer))

        df_answer = pd.concat(dfs_to_concat)
        df['Bezeichnung'] = df_answer['Bezeichnung']
        df.to_csv(args.target_path+op)

    # Plot bars
    data = [args.target_path+d for d in os.listdir(args.target_path)]

    percentages = np.zeros((len(data), 2))

    for i, d in enumerate(data):
        print('Read: ', d)
        df = pd.read_csv(d)
        percentages[i, :] = df['Bezeichnung'].value_counts(
            normalize=True) * 100

    print(np.mean(percentages[:, 0]))
    print(np.std(percentages[:, 0]))

    bar_width = 0.7
    yticks_values = [np.mean(percentages[:, 0]), 0, 20, 40, 60, 80, 100]
    yticks_labels = ['Avg', '0', '20', '40', '60', '80', '100']

    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(15, 9), dpi=600)
    plt.bar(np.arange(len(data)),
            percentages[:, 0], width=bar_width, label='Surgical', color='#1d3557')
    plt.bar(np.arange(len(data)), percentages[:, 1], width=bar_width,
            bottom=percentages[:, 0], label='Daily', color='#e63946')
    plt.grid(True, axis='y', color='gray', linestyle='--', linewidth=0.5)
    plt.xticks([])
    plt.yticks(yticks_values, yticks_labels, fontsize=24)
    plt.ylabel('Percentage', fontsize=28)
    plt.xlabel('Training Set', fontsize=28)
    plt.ylim([0, 100])
    plt.xlim([-0.5, 19.5])
    plt.legend(ncol=2, loc="lower right", fontsize=24)
    plt.savefig('ToC/bars.jpg', bbox_inches='tight')
