import os
import sys
import copy
import time
import torch
import random
import argparse
import subprocess
import numpy as np
import pandas as pd
import huggingface_hub

from dotenv import load_dotenv
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


PRINT_MODE = False


############################## Functions ##############################    
def prefix(id, name='', buffer=5):
    return name + str(id).zfill(buffer)


def get_answer(language_model, tokenizer, messages, max_new_tokens, 
               do_sample=True, top_p=0.95, temperature=1, repetition_penalty=1.05):
    """
    Generates a response from the model based on the provided chat history.

    Parameters:
    language_model: The language model to generate the response.
    tokenizer: The tokenizer to process the text.
    messages: A list of dictionaries containing the chat history.
    max_new_tokens: The maximum number of new tokens to generate (default: 4096).
    do_sample: Whether to sample the output (default: True).
    top_p: The cumulative probability for top-p sampling (default: 0.95).
    temperature: The temperature for sampling (default: 1).

    Returns:
    chat: Updated chat history with the assistant's response.
    response: The generated response from the model.
    """
    # Prepare the prompt from the chat history
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Encode the prompt to input tensor
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    
    #for debugging
    with open('prompt.txt', 'w') as f:
        f.write(prompt)
        f.write(f"\nPrompt has {len(inputs[0])} tokens")
        f.write(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
    
    # Generate the model's response
    outputs = language_model.generate(
        input_ids=inputs.to(language_model.device),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )
        
    # Decode the model's output and update the chat history
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    #for debugging
    with open('response.txt', 'w') as f:
        f.write(response)
        f.write(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
    
    # Extract the model's answer by splitting at the delimiter
    answer = remove_generation_tokens(response)
    answer = get_tagged_block(answer, '<start_of_turn>model', '<end_of_turn>')
    answer = "\n".join(answer) + '\n'
    messages.append({"role": "model", "content": answer})
    
    return messages, answer


def remove_generation_tokens(text, bos_token='<bos>', eos_token='<eos>'):
    clean_text = list()
    for line in text.splitlines():
        if bos_token in line:
            line = line.replace(bos_token, '')
        if eos_token in line:
            line = line.replace(eos_token, '')
        if line != '':
            clean_text.append(line)
    return "\n".join(clean_text)
    

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
        block       : list 
                        A list of strings where the first item contains column names
        
    Returns:
        pandas.DataFrame: A DataFrame of the text
    """
    columns = block[0].split(';')
    
    rows = []
    for line in block[1:]:
        rows.append(line.split(';'))

    df = pd.DataFrame(rows, columns=columns)
    
    # Convert 'Index' to numeric and set it as the index
    df[columns[0]] = df[columns[0]].apply(pd.to_numeric)
    df.set_index(columns[0], inplace=True)

    return df


def line_errors(block, true_len):
    if true_len == len(block) - 1:
        return block
    else:
        # csv input error
        tested_block = []
        for line in block:
            if "```csv" in line:
                line.replace("```csv", '')
            if "```" in line:
                line.replace("```", '')
            tested_block.append(line)
        # new line error
        block = copy.copy(tested_block)
        tested_block = [block[0]]
        shifted_block = block[2:] + ['0']
        for line, sline in zip(block[1:], shifted_block):
            if line[0].isdigit() and sline[0].isdigit():
                tested_block.append(line)
            if line[0].isdigit() and not sline[0].isdigit():
                tested_block.append(line + sline)
        return tested_block


def listdir(path, ending=None):
    '''Returns dir with full path'''
    if ending == None:
        return sorted([os.path.join(path, f) for f in os.listdir(path)])
    else:
        return sorted([os.path.join(path, f) for f in os.listdir(path) 
                       if f.endswith(ending)])
        

def get_stats(data):
    percentage_count = np.zeros((len(data), 8))
    
    for i, d in enumerate(data):
        array_count = np.zeros(8, dtype=int)
        df = pd.read_csv(d)
        phases = df['Phase_Label'].value_counts().drop(8, errors='ignore')
        array_count[phases.index] = phases.values
        percentage_count[i,:] = array_count / np.sum(array_count)
        
    lower_quantile = np.quantile(percentage_count, 0.25, axis=0)
    upper_quantile = np.quantile(percentage_count, 0.75, axis=0)
    
    return [lower_quantile, upper_quantile]


def generate_d_a(q_low, q_high, p, eta=0):
    if p <= q_low + eta:
        d = np.random.uniform(0.1, 0.2)
        a = np.random.uniform(0.2, 0.3)
    elif p >= q_high - eta:
        d = np.random.uniform(0.2, 0.3)
        a = np.random.uniform(0.1, 0.2)
    else:
        d = np.random.uniform(0.1, 0.2)
        a = np.random.uniform(0.12, 0.22)
    return d, a


def drop_and_add(df, qtile_low, qtile_high):
    '''
    This functions drops randomly selected rows from given dataframe,
    and adds empty rows in 'Text' column with '"Ausfüllen"' tag on randomly 
    selected positions. Randomness is controlled by p values.
    
    Parameters:
    df              : DataFrame
                        DataFrame with transcriptions, time and phase labels
    
    qtile_low       : np.array
                        0.25 percent quantile of phases computed from the
                        dataset
     
    qtile_high      : np.array
                        0.75 percent quantile of phases computed from the
                        dataset   
    Returns:
    df              : pd.DataFrame
                        DataFrame with rows dropped and empty rows added.
    df_to_fill      : pd.DataFrame
                        DataFrame with only empty rows.
    '''
    fill_tag = '*Ausfüllen*'
    
    # Remove unrelated parts
    df = df.drop(columns=['File_Name', 'End_Time'], errors='ignore')
    df = df[df['Text'] != '<nicht verstanden>']
    df = df[~df['Phase_Bezeichnung'].isin([8, '8'])]

    # Get phase distribution
    count = np.zeros(8, dtype=int)
    phases = df['Phase_Bezeichnung'].value_counts().drop(8, errors='ignore')
    count[phases.index] = phases.values
    percentage = count / np.sum(count)

    # Find number of lines to drop and add
    to_add = np.zeros(len(count), dtype=int)
    to_drop = np.zeros(len(count), dtype=int)
    for i, (p, c, q_l, q_u) in enumerate(zip(percentage, count, qtile_low, qtile_high)):
        if c == 0:
            pass
        if c > 0 and c < 3:
            to_drop[i] = 0
            to_add[i] = 1
        else:   
            d, a = generate_d_a(q_l, q_u, p)
            to_drop[i] = np.ceil(c * d)
            to_add[i] = np.ceil(c * a)

    # Drop lines
    for phase in range(len(to_drop)):
        if count[phase] - to_drop[phase] > 1:
            phase_indices = df[df['Phase_Bezeichnung'] == phase].index
            if len(phase_indices) >= to_drop[phase]:
                drop_indices = np.random.choice(phase_indices, to_drop[phase], replace=False)
                df = df.drop(drop_indices)
    
    # Add lines
    phase_dfs = []
    for phase in range(len(to_add)):
        phase_df = df[df['Phase_Bezeichnung'] == phase]
        for _ in range(to_add[phase]):
            empty_row = pd.DataFrame({
                'Start_Zeit': [np.nan],
                'Text': [fill_tag],
                'Phase_Bezeichnung': [np.nan]
            })
            
            if len(phase_df) == 1:
                phase_df = pd.concat([phase_df, empty_row]).reset_index(drop=True)
            else:
                idx = np.random.randint(1,len(phase_df))
                phase_df = pd.concat([phase_df.iloc[:idx], empty_row, phase_df.iloc[idx:]]).reset_index(drop=True)
                
        phase_dfs.append(phase_df)
        
    # Concat
    df = pd.concat(phase_dfs).reset_index(drop=True)
    
    # Fill the phase label column
    df['Phase_Bezeichnung'] = df['Phase_Bezeichnung'].apply(pd.to_numeric).ffill().astype(int)
    
    # Update the time column with randomly generated values
    df['Start_Zeit'] = df['Start_Zeit'].apply(pd.to_numeric).ffill()
    
    relaxation = 0.01
    last_phase_end = 0
    
    for phase in df['Phase_Bezeichnung'].unique():
        
        phase_df = df[df['Phase_Bezeichnung'] == phase].copy()
        
        start_time = phase_df['Start_Zeit'].iloc[0]
        end_time = phase_df['Start_Zeit'].iloc[-1]
        
        if last_phase_end == 0:
            start_time = np.random.uniform(start_time*(1-relaxation), start_time*(1+relaxation), 1)[0]
        else:
            if start_time > last_phase_end:
                start_time = np.random.uniform(last_phase_end, start_time*(1+relaxation), 1)[0]
            else:
                start_time = np.random.uniform(last_phase_end, last_phase_end*(1+relaxation), 1)[0]
        
        
        end_time = np.random.uniform(end_time*(1-relaxation), end_time*(1+relaxation), 1)[0]
        last_phase_end = end_time
        
        if len(phase_df) > 2:
            random_times = sorted(np.random.uniform(start_time + 1, end_time, len(phase_df) - 2))
            random_times = [start_time] + list(random_times) + [end_time]
            
            for i, new_time in zip(phase_df.index, random_times):
                df.at[i, 'Start_Zeit'] = new_time
        
        else:
            if len(phase_df) == 2:
                df.at[phase_df.index[0], 'Start_Zeit'] = start_time
                df.at[phase_df.index[1], 'Start_Zeit'] = end_time
            elif len(phase_df) == 1:
                df.at[phase_df.index[0], 'Start_Zeit'] = start_time
    
    df['Start_Zeit'] = df['Start_Zeit'].astype(float).round(3)
      
    # Copy also empty datatframe
    df_to_fill = df.copy()
    df_to_fill = df_to_fill[df_to_fill['Text'] == fill_tag]
    df_to_fill['Grund'] = fill_tag
    
    return df, df_to_fill


def df_splitter(df, num_sub_dfs=4):
    return np.array_split(df, num_sub_dfs)


def check_format(block, columns, generation_tag='*Ausfüllen*'):
    # Check column names
    columns_flag = block[0].split(';') == columns
    
    # Check row entries
    rows_flag = 1
    for row in block[1:]:
        length = len(row.split(';')) == len(columns)
        generation = not (generation_tag in row)
        empty = np.prod([item != '' for item in row.split(';')])
        rows_flag *= (length * generation * empty)       
          
    return columns_flag * rows_flag
        

def gen_data(language_model, tokenizer, sample_data, qtile_low, qtile_high):
    '''
    Generates synthetic data
                        
    language_model  : transformers.AutoModelForCausalLM
                        Gemma-2 model
                    
    tokenizer       : transformers.AutoTokenizer
                        Gemma-2 tokenizer
                        
    sample_data     : DataFrame
                        Example surgical operation transcript
    '''
    ############################## Step 1 ##############################
    # Variables
    limit_try = 3
    chat_container = []
    max_token_generation = 1000
    
    # Prepare Role Prompt
    role = """* Rolle: Du bist ein hilfsbereites künstlicher Assistent, der die Sprache der Chirurgen in der radiologischen Abteilung nachahmt, die Operationen zur Platzierung von Portkathetern durchführen. Die Phasen der Operation wurden im Abschnitt <Operation> angegeben. Das Ziel ist es, Gespräche eines Chirurgen mit dem medizinischen Assistenten und dem Patienten während einer Port-Katheter-Platzierung zu generieren. Die neu generierten Daten werden für das Training eines textbasierten Deep-Learning-Modells verwendet, das entwickelt wurde, um die chirurgischen Phasen der Port-Katheter-Placement-Operation zu erkennen. Alle Gespräche in den Daten müssen ausschließlich auf Deutsch geführt werden. Um eine neue Daten zu erstellen, befolge die Anweisungen und gib deine Antwort in den markierten Abschnitten ein."""
    role += """<Operation>
Phase_Bezeichnung;Phase_Name;Phase_Beschreibung
0;Vorbereitung;Der Patient wird auf die Operation vorbereitet, was die Desinfektion der Haut, die Auswahl der Punktionsstelle und die Bereitstellung der erforderlichen medizinischen Instrumente umfasst.
1;Punktion;Eine große Vene, typischerweise die Vena subclavia oder Vena jugularis interna, wird mit der Hilfe Ultraschall punktiert, um einen Zugang zum venösen System zu schaffen.
2;Positionierung des Führungsdrahtes;Ein Führungsdraht wird durch die Punktionsnadel in die Vene eingeführt und dient als Leitschiene für den Katheter.
3;Vorbereitung des Pouches und Platzierung des Katheters;Ein subkutanes Reservoir (Port-Pouch) wird vorbereitet, und der Katheter wird durch einen Tunnel unter der Haut zum Pouch geführt.
4;Positionierung des Katheters;Der Katheter wird über den Führungsdraht in die Vene vorgeschoben und bis in die gewünschte Position, meist nahe dem Herzen, platziert.
5;Anpassung des Katheters;Der Katheter wird auf die erforderliche Länge zugeschnitten und fest mit dem Port verbunden, um eine stabile Platzierung zu gewährleisten.
6;Kontrolle der Katheter;Die korrekte Positionierung und Funktion des Katheters werden durch bildgebende Verfahren (Digitale Subtraktionsangiographie) überprüft.
7;Abschluss;Die Hautschnitte werden vernäht, ein steriler Verband wird angelegt, und der Patient wird überwacht, bis er sich vollständig von der Narkose erholt hat.
</Operation>
"""
    # Prepare Data
    df_org = pd.read_csv(sample_data, index_col=0)
    df_org = df_org.rename(columns={'Start_Time': 'Start_Zeit', 'Phase_Label': 'Phase_Bezeichnung'})
    
    df_daa, df_to_fill = drop_and_add(df_org, qtile_low, qtile_high)
    
    # Prepare prompt iteratively
    dfs_to_concat = []
    tokens_per_row = 50
    num_sub_dfs = tokens_per_row*len(df_to_fill) // max_token_generation + 1
    
    # Sliding window
    df_to_print = df_daa.copy()
    df_to_print.drop(columns=['Start_Zeit'], inplace=True)
        
    for i, sub_df in enumerate(df_splitter(df_to_fill, num_sub_dfs)):
        max_new_tokens = tokens_per_row*len(sub_df)
        
        # Try limit_try times, if Gemma cant follow instructions
        n_try = 0
        while n_try < limit_try:
            try:
                print(f'\tStep 1: {int(i+1)}/{num_sub_dfs} max new tokens: {max_new_tokens}')
                
                prompt = """\n* Anweisung: Du erhältst einen Datensatz mit einer Reihe von Sätzen im Abschnitt <Daten>. Die Daten enthalten einen Index, die Startzeit der Rede, den gesprochenen Satz und eine Bezeichnung für die Operationsphase. Allerdings fehlen einige Daten in der Textspalte. 
* Aufgabe: Deine Aufgabe ist es, die Zeilen in der Spalte 'Text', die mit '"Ausfüllen"' markiert sind, in mehreren Schritten zu ergänzen. In diesem Schritt wirst du speziell die ausgewählten Teile der Daten ergänzen, die im Abschnitt <Antwort 1> aufgeführt sind. Stell sicher, dass die generierten Sätze mit der vorgegebenen Phasen im Abschinitt <Operation> übereinstimmen und auch den Nachbarsätzen anknüpfen, wobei der Kontext erhalten bleibt. Verwende die Spalte 'Grund' in der Vorlage <Antwort 1>, um mit 10 Wörtern zu erklären, warum du dich für diese Sätze entschieden hast.
* Still: Du sollst die Daten in einem konsistenten Stil mit dem unten angegebenen Daten erstellen. Verwende die gesamte Daten im <Daten> Abschnitt um die Kontext der Gescprähe zu erfahren. Antworte unbedingt im CSV-Format als in der Vorlage <Antwort 1>."""
                prompt += f"\n<Daten>\n{df_to_print.to_csv(index=True, sep=';', index_label='Index')}</Daten>\n"
                prompt += f"\n<Antwort 1>\n{sub_df.to_csv(index=True, sep=';', index_label='Index')}</Antwort 1>\n"
    
                # Generate Answer 1
                messages = [{"role": "user", "content": role + prompt}]
                messages, answer = get_answer(language_model, tokenizer, messages, max_new_tokens=max_new_tokens) 
                '''
                with open(f'step_1_{i+1}.txt', 'r', encoding='utf-8') as file:
                    answer = file.readlines()
                answer = "\n".join(answer) + '\n'
                '''
                
                # Extract tagged block
                block_answer = get_tagged_block(answer, '<Antwort 1>', '</Antwort 1>')
                
                # Check line shapes/errors
                block_answer = line_errors(block_answer, len(sub_df))
                
                # Check format
                correct_format = check_format(block_answer, ['Index', 'Start_Zeit', 'Text', 'Phase_Bezeichnung', 'Grund'])
                if not correct_format:
                    print('\t\tColumns or rows do not match')
                    raise ValueError(f"Columns or rows do not match")
                
                # Concat
                dfs_to_concat.append(block_to_df(block_answer))
                
                # Log for debugging
                if PRINT_MODE:
                    with open(f'step_1_{i+1}.txt', 'w') as f:
                        for m in messages:
                            f.write('*'*50+' <'+m['role']+'> '+'*'*50+'\n')
                            f.write(m['content'])
                            chat_container.append({"role": m['role'], "content": m['content']})
                
                # Exit loop
                n_try = limit_try
            except:
                # Increment
                print(f'\t\tConnot genreate step_1_{i+1}.txt, will try again')
                n_try += 1

    df_empty_rows = pd.concat(dfs_to_concat)
    df_empty_rows['Text'] = df_empty_rows['Text'].str.replace('"""', '')
    df_filled = df_daa.copy()
    df_filled.loc[df_empty_rows.index, 'Text'] = df_empty_rows['Text']
    
    ############################## Step 2 ##############################
    # Variables
    limit_try = 3
    dfs_to_concat = []
    tokens_per_row = 50
    max_token_generation = 1000
    num_sub_dfs = tokens_per_row*len(df_filled) // max_token_generation + 1
    
    # Sliding window
    df_to_print = df_filled.copy()
    df_to_print.drop(columns=['Start_Zeit'], inplace=True)
    
    for i, sub_df in enumerate(df_splitter(df_filled, num_sub_dfs)):
        # Work data
        sub_df = sub_df.drop(columns=['Text', 'Phase_Bezeichnung'])
        sub_df['Relevanz'] = "*Ausfüllen*"
        sub_df['Grund'] = "*Ausfüllen*"
        max_new_tokens = tokens_per_row*len(sub_df)
                
        # Try limit_try times, if Gemma cant follow instructions
        n_try = 0
        while n_try < limit_try:
            try:
                print(f'\tStep 2: {int(i+1)}/{num_sub_dfs} max new tokens: {max_new_tokens}')
                
                prompt = """\n* Anweisung: Du erhältst einen Datensatz mit einer Reihe von Sätzen im Abschnitt <Daten>. Die Daten enthalten einen Index, die Startzeit der Rede, den gesprochenen Satz und eine Bezeichnung für die Operationsphase. 
* Aufgabe: Deine Aufgabe ist die einzeln gegebenen Sätze in der Spalte „Text“ in mehreren Schritten analizieren und bestimmen, ob die Sätze für die Erkennung der in der Spalte 'Phase_Bezeichnung' angegebenen chirurgischen Phasen wichtig sind. In diesem Schritt wirst du speziell mit der ausgewählten Teile der Daten arbeiten, die im Abschnitt <Antwort 2> aufgeführt sind. Markiere 'C' für Sätze, die für die Erkennung chirurgischer Phasen relevant sind, und 'T' für Sätze, die im Kontext einer täglichen Unterhaltung stehen. 
* Still: Verwende die gesamte Daten im <Daten> Abschnitt um die Kontext der Gescprähe zu erfahren. gib deine Antwort in der Splate 'Relevanz' im Abschnitt <Antwort 2>. Verwende die Spalte 'Grund' in der Vorlage im <Antwort 2> Bereich, um mit maximal zwei Wörtern zu erklären, warum du dich für diese Sätze entschieden hast. Antworte unbedingt im CSV-Format als in der Vorlage <Antwort 2>."""
                prompt += f"\n<Daten>\n{df_to_print.to_csv(index=True, sep=';', index_label='Index')}</Daten>\n"
                prompt += f"\n<Antwort 2>\n{sub_df.to_csv(index=True, sep=';', index_label='Index')}</Antwort 2>\n"
                
                # Generate Answer
                messages = [{"role": "user", "content": role + prompt}]
                messages, answer = get_answer(language_model, tokenizer, messages, max_new_tokens=max_new_tokens)
                '''
                with open(f'step_2_{i+1}.txt', 'r', encoding='utf-8') as file:
                    answer = file.readlines()
                answer = "\n".join(answer) + '\n'
                '''
                
                # Extract tagged block
                block_answer = get_tagged_block(answer, '<Antwort 2>', '</Antwort 2>')

                # Check line shapes/errors
                block_answer = line_errors(block_answer, len(sub_df))

                # Check format
                correct_format = check_format(block_answer, ['Index', 'Start_Zeit', 'Relevanz', 'Grund'])
                if not correct_format:
                    print('\t\tColumns or rows do not match')
                    raise ValueError("Columns or rows do not match")
                
                # Concat
                dfs_to_concat.append(block_to_df(block_answer))
                
                # For debugging
                if PRINT_MODE:
                    with open(f'step_2_{i+1}.txt', 'w') as f:
                        for m in messages:
                            f.write('*'*50+' <'+m['role']+'> '+'*'*50+'\n')
                            f.write(m['content'])
                            chat_container.append({"role": m['role'], "content": m['content']})
                        
                # Exit loop
                n_try = limit_try
            except:
                print(f'\t\tConnot genreate step_2_{i+1}.txt, will try again')
                n_try += 1
                if n_try == limit_try: limit_try = -1
    
    df_to_rewrite = pd.concat(dfs_to_concat)
    df_to_rewrite = pd.concat([df_filled, df_to_rewrite], axis=1)
    
    ############################## Step 3 ##############################
    #Variables
    limit_try = 3
    dfs_to_concat = []
    tokens_per_row = 50
    max_token_generation = 850
    num_sub_dfs = tokens_per_row*len(df_to_rewrite) // max_token_generation + 1
    
    # Sliding Window
    df_to_print = df_to_rewrite.copy()
    df_to_print.drop(columns=['Start_Zeit'], inplace=True)
    
    for i, sub_df in enumerate(df_splitter(df_daa, num_sub_dfs)):
        # Work data
        sub_df['Text'] = '*Ausfüllen*'
        sub_df = sub_df[['Start_Zeit', 'Phase_Bezeichnung', 'Text']]
        max_new_tokens = tokens_per_row*len(sub_df)
        
        # Try limit_try times, if Gemma cant follow instructions
        n_try = 0
        while n_try < limit_try:
            try:
                print(f'\tStep 3: {int(i+1)}/{num_sub_dfs} max new tokens: {max_new_tokens}')
                
                prompt = """\n* Anweisung: Du erhältst einen Datensatz mit einer Reihe von Sätzen im Abschnitt <Daten>. Die Daten enthalten einen Index, die Startzeit der Rede, den gesprochenen Satz, eine Bezeichnung für die Operationsphase. Vielmehr zeigt die Relevanzssplate an, ob der Satz zu den täglichen (mit 'T' markiert) oder chirurgischen (mit 'C' markiert) Gesprächen gehört, während Grund erklärt, warum die Relevanzspalte mit 'T' oder 'C' bezeichnet ist. 
* Aufgabe: Deine Aufgabe ist es, die Sätze in der Spalte "Text" der bereitgestellten Daten in mehreren Schritten umzuformulieren. In diesem Schritt wirst du speziell die ausgewählten Teile der Daten umformulieren, die im Abschnitt <Antwort 3> aufgeführt sind. Gehe jeden Satz in der Spalte 'Text' einzeln durch. Formuliere die Sätze wie folgt um: Wenn die Spalte "Relevanz" mit 'C' markiert ist, gib die Ereignisse aus dem Text wieder, drücke sie aber in deinen eigenen Worten aus. Wenn die Spalte "Relevanz" mit 'T' markiert ist, gib Sie die Ereignisse im Text nicht wieder, sondern behandel ein anderes Tagesthema, das du dich ausgedacht hat.
* Still: Verwende die gesamte Daten im <Daten> Abschnitt um die Kontext der Gescprähe zu erfahren. Generiere keine Passivsätze und verwende die Perspektive der dritten Person nicht. Gib deine Antwort in der Splate 'Text', die mit '*Ausfüllen*' markiert sind. Antworte unbedingt im CSV-Format als in der Vorlage <Antwort 3>.*"""

                prompt += f"\n<Daten>\n{df_to_print.to_csv(index=True, sep=';', index_label='Index')}</Daten>\n"
                prompt += f"\n<Antwort 3>\n{sub_df.to_csv(index=True, sep=';', index_label='Index')}</Antwort 3>\n"
                
                # Generate Answer
                messages = [{"role": "user", "content": role + prompt}]
                messages, answer = get_answer(language_model, tokenizer, messages, max_new_tokens=max_new_tokens)
                
                # Extract tagged block
                block_answer = get_tagged_block(answer, '<Antwort 3>', '</Antwort 3>')

                # Check line shapes/errors
                block_answer = line_errors(block_answer, len(sub_df))
                
                # Check format
                correct_format = check_format(block_answer, ['Index', 'Start_Zeit', 'Phase_Bezeichnung', 'Text'])
                df_answer = block_to_df(block_answer)
                df_answer = df_answer[['Start_Zeit', 'Text', 'Phase_Bezeichnung']]
                df_answer['Phase_Bezeichnung'] = df_answer['Phase_Bezeichnung'].apply(pd.to_numeric)
                if not correct_format:
                    print('\t\tColumns or rows do not match')
                    raise ValueError(f"Columns or rows do not match")
    
                # Check language
                #TODO
                
                # Concat
                dfs_to_concat.append(df_answer)
                
                # For debugging
                if PRINT_MODE:
                    with open(f'step_3_{i+1}.txt', 'w') as f:
                        for m in messages:
                            f.write('*'*50+' <'+m['role']+'> '+'*'*50+'\n')
                            f.write(m['content'])
                            chat_container.append({"role": m['role'], "content": m['content']})
                
                # Exit loop
                n_try = limit_try
            except:
                print(f'\t\tConnot genreate step_3_{i+1}.txt, will try again')
                n_try += 1
                if n_try == limit_try: raise ValueError(f"Can't generate data:(")

    # Merge dataframes and process
    result_df = pd.concat(dfs_to_concat)
    result_df = result_df.rename(columns={
        'Start_Zeit': 'Start_Time',
        'Phase_Bezeichnung': 'Phase_Label'
    })
    result_df['Text'] = result_df['Text'].str.strip()
    result_df.to_csv('backup.csv')
    
    # Remove log files
    for file in listdir('./', ending='.txt'):
        os.remove(file) 
    
    return result_df, chat_container
        

############################## Main ##############################
if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generation for SPR")
    
    parser.add_argument('-t', '--target_path', type=str,
                        default='SynPoCaP/',
                        help='path to save generated data')
    
    parser.add_argument('-n', '--num_target', 
                        type=int, default=1,
                        help='number of data to generate')
    
    parser.add_argument('-p', '--prefix_index', 
                        type=int, default=0,
                        help='prefix start index of new data')
    
    args = parser.parse_args()

    # Target folder
    if not os.path.exists(args.target_path):
        os.mkdir(args.target_path)

    # Variables
    error_count = 0
    error_patience = 5
    prefix_idx = args.prefix_index
    model_id = 'google/gemma-2-27b-it'
    end_idx = prefix_idx + args.num_target - 1
    
    # Login huggingface environment
    load_dotenv()
    huggingface_hub.login(os.getenv('HF_TOKEN'), add_to_git_credential=False)
    
    # Model
    gemma2_tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=os.getenv('HF_CACHE_DIR'))
    gemma2 = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
        cache_dir=os.getenv('HF_CACHE_DIR')
    )

    # Read reference data
    data_path = 'Transcripts/'
    seedset = ['OP_009.csv', 'OP_012.csv', 'OP_033.csv', 'OP_034.csv', 
                'OP_026.csv', 'OP_029.csv', 'OP_025.csv', 'OP_036.csv', 
                'OP_013.csv', 'OP_030.csv', 'OP_003.csv',
                'OP_005.csv', 'OP_040.csv', 'OP_031.csv',
                'OP_002.csv', 'OP_016.csv', 'OP_014.csv'] #006 -> 500+, 17, 24 --> 250+
    trainset = [os.path.join(data_path, s) for s in seedset]

    validset = ['OP_019.csv', 'OP_032.csv', 'OP_039.csv', 'OP_010.csv', 
                'OP_022.csv', 'OP_008.csv', 'OP_001.csv', 'OP_038.csv']
    validset = sorted([os.path.join(data_path, v) for v in validset])
    
    # Compute quantiles    
    qtile_low, qtile_high = get_stats(trainset)

    # Generate Data
    while prefix_idx <= end_idx and error_count < error_patience:
        # set save name
        save_name = args.target_path + prefix(prefix_idx+1, 'SynOP_') + ".csv"
        
        # select sample data
        random.shuffle(trainset)
        sample_data = trainset[0]
        
        print(f"Target: {save_name}\tSource: {sample_data}")
        
        # generate data, save and log
        try:
            generation_start = time.time()
            df, chat_container = gen_data(
                language_model=gemma2, 
                tokenizer=gemma2_tokenizer, 
                sample_data=sample_data, 
                qtile_low=qtile_low, 
                qtile_high=qtile_high
            )

            df.to_csv(save_name)
            
            # update trainset
            for g in os.listdir(args.target_path):
                if g.endswith('.csv'):
                    trainset.append(os.path.join(args.target_path, g))
            
            # save log
            with open(save_name[:-4] + ".txt", "w") as f:
                f.write('Refence Data:'+sample_data+'\n')
                for c in chat_container:
                    f.write('\n'+'*'*50+' <'+c['role']+'> '+'*'*50+'\n')
                    f.write(c['content'])
                f.write(f'\nElapsed time:\t{time.time() - generation_start}(s)')
                error_count = 0
            
            # increment
            prefix_idx += 1

        # uppps
        except:
            print(f"{save_name} could not generated\nTrying again!")
            error_count += 1