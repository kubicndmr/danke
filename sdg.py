import os
import sys
import copy
import time
import torch
import argparse
import matplotlib
import numpy as np
import pandas as pd
import huggingface_hub
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


LOG_MODE = True
DEBUG_MODE = False
PRINT_MODE = False

if DEBUG_MODE:
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

####################################### Functions, Classes #######################################
class PoCaPCorpus():
    def __init__(self, seedset, target_path):
        self.target_path = target_path
        self.dataset = seedset
        self.count_corpus()
        self.compute_quantiles()
    
    def count_corpus(self):
            self.phase_count = np.zeros((len(self.dataset), 8))
            self.percentage_count = np.zeros((len(self.dataset), 8))
            
            for i, d in enumerate(self.dataset):
                array_count = np.zeros(8, dtype=int)
                df = pd.read_csv(d)
                phases = df['Phase_Label'].value_counts().drop(8, errors='ignore')
                array_count[phases.index] = phases.values
                self.phase_count[i, :] = array_count
                self.percentage_count[i, :] = array_count / np.sum(array_count)
    
    def get_sample_data(self):
        return np.random.choice(self.dataset)
    
    def update_dataset(self):
        new_files = [os.path.join(self.target_path, gen) 
                    for gen in os.listdir(self.target_path) 
                    if gen.endswith('.csv') and os.path.join(self.target_path, gen) not in self.dataset]
        if len(new_files) != 0:
            self.dataset.extend(new_files)
        
    def compute_quantiles(self):
        self.lower_quantile_phase = np.quantile(self.phase_count, 0.25, axis=0)
        self.upper_quantile_phase = np.quantile(self.phase_count, 0.75, axis=0)
        self.lower_quantile_percent = np.quantile(self.percentage_count, 0.25, axis=0)
        self.upper_quantile_percent = np.quantile(self.percentage_count, 0.75, axis=0)
    
    def sample_d_a(self, phase, percent_count, phase_count):
        if (percent_count <= self.lower_quantile_percent[phase] or 
            phase_count <= self.lower_quantile_phase[phase]):
            d = np.random.uniform(0.1, 0.2)
            a = np.random.uniform(0.2, 0.3) #106.25
        elif (percent_count >= self.upper_quantile_percent[phase] or
              phase_count >= self.upper_quantile_phase[phase]):
            d = np.random.uniform(0.2, 0.3)
            a = np.random.uniform(0.2, 0.3) #93.75
        else:
            d = np.random.uniform(0.15, 0.25)
            a = np.random.uniform(0.2, 0.3) #100
        return d, a
            
    def plot_violin(self):
        # update counts
        phase_count = np.zeros((len(self.dataset), 8), dtype=int)
        percentage_count = np.zeros((len(self.dataset), 8))
        
        for i, d in enumerate(self.dataset):
            array_count = np.zeros(8, dtype=int)
            df = pd.read_csv(d)
            phases = df['Phase_Label'].value_counts().drop(8, errors='ignore')
            array_count[phases.index] = phases.values
            phase_count[i, :] = array_count
            percentage_count[i, :] = array_count / np.sum(array_count)
        
        # plot
        #plt.rcParams["font.family"] = "Times New Roman"
        def_cmap = matplotlib.colormaps.get_cmap('tab10')
        color_list = def_cmap(np.linspace(0, 1, 9))
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 9), dpi=75)
        
        '''
        for i in range(0,8):
            data = self.phase_count[:self.index, i]
            axs[0].scatter([i]*len(data), data, color=color_list[i], edgecolor='black', alpha=0.7)
        '''
        violins = axs[0].violinplot(
            phase_count,
            showmeans=False, 
            showmedians=True, 
            showextrema=False
        )
        for j, pc in enumerate(violins['bodies']):
            pc.set_facecolor(color_list[j])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
                 
        #axs[0].set_ylim(-1, 150)
        axs[0].set_ylabel('#Sentences', fontsize=16)
        axs[0].set_xlabel('Surgical Phases', fontsize=16)
        axs[0].tick_params(axis='x', labelsize=14)
        axs[0].tick_params(axis='y', labelsize=14)
        
        violins = axs[1].violinplot(
            percentage_count, 
            showmeans=False, 
            showmedians=True, 
            showextrema=False
        )
        for j, pc in enumerate(violins['bodies']):
            pc.set_facecolor(color_list[j])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        #axs[1].set_ylim(-0.01, 0.6)
        axs[1].set_ylabel('Percentage', fontsize=16)
        axs[1].set_xlabel('Surgical Phases', fontsize=16)
        axs[1].tick_params(axis='x', labelsize=14)
        axs[1].tick_params(axis='y', labelsize=14)
        fig.savefig(os.path.join(self.target_path, 'clas_dist.png'), bbox_inches='tight')
        
        
def prefix(id, name='', buffer=5):
    return name + str(id).zfill(buffer)


def get_answer(tokenizer, language_model, messages, max_new_tokens, 
               do_sample=True, top_p=0.95, temperature=1, repetition_penalty=1.1):
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
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs['input_ids'] = inputs['input_ids'].to(language_model.device)
    
    # Output
    if DEBUG_MODE:
        with open('prompt.txt', 'w') as f:
            f.write(prompt)
            f.write(f"\n\nPrompt has {len(inputs[0])} tokens")
    
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
        
    # Decode the model's output and update the chat history
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Output
    if DEBUG_MODE:
        with open('response.txt', 'w') as f:
            f.write(response)
    
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


def drop_and_add(df, pocap):
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
    fill_tag = '*fehlende Daten*'
    
    # Remove unrelated parts
    df = df.drop(columns=['File_Name', 'End_Time'], errors='ignore')
    df = df[df['Text'] != '<nicht verstanden>']
    df = df[~df['Phase_Bezeichnung'].isin([8, '8'])]
    
    # Get phase distribution of given data
    count = np.zeros(8, dtype=int)
    phases = df['Phase_Bezeichnung'].value_counts()
    count[phases.index] = phases.values
    percentage = count / np.sum(count)
    
    # Find number of lines to drop and add
    to_add = np.zeros(len(count), dtype=int)
    to_drop = np.zeros(len(count), dtype=int)
    for i, (p, c) in enumerate(zip(percentage, count)):
        if c == 0:
            pass
        if c > 0 and c < 3:
            to_drop[i] = 0
            to_add[i] = 1
        else:
            d, a = pocap.sample_d_a(i, p, c)
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
    
    return df, df_to_fill


def merge_small_groups(groups, min_size=6):
        merged_groups = []
        current_group = groups[0]

        for next_group in groups[1:]:
            if len(current_group) < min_size:
                current_group = pd.concat([current_group, next_group])
            else:
                merged_groups.append(current_group)
                current_group = next_group

        # Add the last group
        if len(current_group) < min_size and merged_groups:
            merged_groups[-1] = pd.concat([merged_groups[-1], current_group])
        else:
            merged_groups.append(current_group)
        
        return merged_groups


def df_splitter(df, max_df_length=20):
    split_dfs = []

    for _, split_df in df.groupby('Phase_Bezeichnung'):
        if len(split_df) <= max_df_length:
            split_dfs.append(split_df)
        else:
            num_split = int(np.ceil(len(split_df) / max_df_length))
            sub_dfs = np.array_split(split_df, num_split)
            split_dfs.extend(sub_dfs)
    
    return merge_small_groups(split_dfs)


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
        

def gen_data(tokenizer, language_model, pocap):
    '''
    Generates synthetic data
                        
    language_model  : transformers.AutoModelForCausalLM
                        Gemma-2 model
                    
    tokenizer       : transformers.AutoTokenizer
                        Gemma-2 tokenizer
                        
    sample_data     : DataFrame
                        Example surgical operation transcript
    '''
    # Get sample data
    sample_data = pocap.get_sample_data()
    chat_container = [{'role':'sample', 'content':sample_data}]
    
    label_dic = {
        0: 'Vorbereitung', 1: 'Punktion', 2: 'Führungsdraht',
        3: 'Pouchvorbereitung-und-Katheterplatzierung', 4: 'Katheterpositionierung', 5: 'Katheteranpassung',
        6: 'Katheterkontrolle', 7: 'Abschluss'
    }
    
    ####################################### Step 1 #######################################
    # Variables
    limit_try = 3
    tokens_per_row = 100
    
    # System Prompt
    role = """Du bist ein hilfsbereites Assistent, das die Chirurgen mit der Hilfe der Beispieldaten nachahmt. Die Aufgabe ist es, künstliche Gespräche eines Chirurgen mit dem medizinischen Assistenten und dem Patienten während einer Port-Katheter-Platzierung Operation zu generieren. Die neu generierten Daten werden für das Training eines textbasierten Deep-Learning-Modells verwendet, das entwickelt wurde, um die chirurgischen Phasen der Port-Katheter-Placement-Operation zu erkennen."""
    
    # Prepare Data
    df_sample = pd.read_csv(sample_data, index_col=0)
    df_sample = df_sample.rename(columns={'Start_Time': 'Start_Zeit', 'Phase_Label': 'Phase_Bezeichnung'})
    df_sample, df_to_fill = drop_and_add(df_sample, pocap)
    df_to_print = df_sample.copy()
    
    df_to_print.drop(columns=['Start_Zeit'], inplace=True)
    df_to_print['Phase_Bezeichnung'] = df_to_print['Phase_Bezeichnung'].map(label_dic)
    
    # Sliding window
    dfs_to_concat = []
    df_splits = df_splitter(df_to_fill)
    steps_complete = np.zeros(len(df_splits))
    
    for i, sub_df in enumerate(df_splits):
        max_new_tokens = tokens_per_row*len(sub_df)
        sub_df['Phase_Bezeichnung'] = sub_df['Phase_Bezeichnung'].map(label_dic)
        
        # Try limit_try times, if LM cant follow instructions
        n_try = 0
        while n_try < limit_try:
            try:
                if PRINT_MODE:
                    print(f'\tStep 1: {int(i+1)}/{len(df_splits)} max new tokens: {max_new_tokens}')
                ####################################### Step 1.1 #######################################
                which_phase = sub_df['Phase_Bezeichnung'].unique()
                sub_print_df = df_to_print[df_to_print['Phase_Bezeichnung'].isin(which_phase)]
                
                prompt = """Du wirst die chirurgischen Phasen und Schritte einer Operation festlegen.
* Operation: Chirurgische Phasen und chirurgische Schritte darstellen eine typische Operation. Die Phasen beziehen sich auf die großen Abschnitte des Verfahrens, in denen die wichtigsten Schritte beschrieben werden. Chirurgische Schritte sind die spezifischen Aufgaben, die innerhalb jeder Phase ausgeführt werden sollen. Operationen folgen im Allgemeinen dieser Reihenfolge der Ereignisse, mit Ausnahmen. Die Phasen und Schritte der Port-Katheter-Platzierung Operation sind folgendes:
- Phase 0: Vorbereitung. Schritte: 0.1) Positionierung des Patienten auf dem Tisch 0.2) Tisch fährt hoch 0.3) Radiologe sterilisiert sich 0.4) Vorbereitung des sterilen Materials 0.5) Patient steril abgedeckt
- Phase 1: Punktion. Schritte: 1.1) Lokale Anästhesie, 1.2 Ultraschallgeführte Punktion
- Phase 2: Führungsdraht. Schritte: 2.1) Röntgenmaschine fährt ein, 2.2) Durchleuchtung im Bereich der Subklavia, 2.3) Durchleuchtung im Bereich der Vena cava inferior (VCI), 2.4) Röntgenmaschine fährt heraus
- Phase 3: Pouchvorbereitung-und-Katheterplatzierung. Schritte: 3.1) Lokale Anästhesie, 3.2) Inzision, 3.3) Pouch-Vorbereitung 3.4) Hülleplatzierung
- Phase 4: Katheterpositionierung. Schritte: 4.1) Röntgenmaschine fährt ein, 4.2) Durchleuchtung des VCI-Bereichs, 4.3) Positionierung des Katheters
- Phase 5: Katheteranpassung. Schritte: 5.1) Kürzen des Katheters, 5.2) Röntgenmaschine fährt aus, 5.3) Anschluss des Katheters an die Portkapsel, 5.4) Positionierung der Portkapsel im Pouch, 5.5) Chirurgische Naht, 5.6) Punktion der Portkapsel
- Phase 6: Katheterkontrolle. Schritte: 6.1) Röntgenmaschine fährt ein, 6.2) Digitale Subtraktionsangiographie des Brust 6.3) Röntgenmaschine fährt in Parkposition aus
- Phase 7: Abschluss. Schritte: 7.1) Steriles Pflaster auflegen, 7.2) Tisch fährt nach unten
* Daten: Du erhältst einen Datensatz mit fehlenden Unterhaltungen im Abschnitt <Daten 1>. Die Daten enthalten einen Index, die Startzeit der Rede, den gesprochenen Satz eines Chirurgen und eine Bezeichnung für die Operationsphase.
* Aufgabe: Verwende die Vorlage in Abschnitt <Antwort 1> und beantwort die Fragen. Noch keine Sätze generieren, um fehlende Daten zu ergänzen."""
                prompt += f"\n<Daten 1>\n{sub_print_df.to_csv(index=True, sep=';', index_label='Index')}</Daten 1>\n"
                prompt += """
<Antwort 1> 
1.Welche chirurgische Phase enthalten die gegebenen Daten?
2.Welche chirurgischen Schritte enthält diese Phase?
3.Welche chirurgischen Schritte wurden abgeschlossen?
4.Falls vorhanden, welche chirurgischen Schritte sind in dieser Phase noch nicht abgeschlossen?
</Antwort 1>
"""             
                # Generate Answer 1.1
                messages = [{"role": "system", "content": role}]
                messages.append({"role": "user", "content":prompt})
                answer = get_answer(tokenizer, language_model, messages, max_new_tokens=1500)
                
                # Add to chat
                block_answer = get_tagged_block(answer, '<Antwort 1>', '</Antwort 1>')
                messages.append({"role": "assistant", "content": '\n'.join(block_answer)+'\n'})
                
                ####################################### Step 1.2 #######################################
                prompt = """\nDu wirst die fehlenden Konversationen im Abschnitt <Daten 2> ergänzen, indem du chirurgische Phasen und Schritte berücksichtigst.
* Daten: Du erhältst eine Analyse der chirurgischen Phasen und Schritte in deiner vorherigen Antwort und einen Datensatz mit fehlenden Unterhaltungen im Abschnitt <Daten 2>. Der Datensatz enthalt einen Index, die Startzeit der Rede, den gesprochenen Satz eines Chirurgen und eine Bezeichnung für die Operationsphase.
* Aufgabe: Deine Aufgabe ist es, die Zeilen in der Spalte 'Text' und 'Schritte_Bezeichnung', die mit '"Ausfüllen"' markiert sind, zu ergänzen. Die Spalte Schritt_Bezeichnung zeigt an, welcher chirurgische Schritt in dieser Datenzeile läuft, und die Spalte Text zeigt das Gespräch des Chirurgen mit dem Arzthelfer oder dem Patienten im Operationssaal.
* Strategie: Verwenden Sie die Analyse aus Ihrer vorherigen Antwort und gib zunächst in der Spalte 'Schritt_Bezeichnung' den laufenden Operationsschritt an. Berücksichtig, welche chirurgischen Phasen oder Schritte wurden abgeschlossen, oder durchgeführt werden. Dann erzeuge entsprechende Sätze für diesen Schritt in der Spalte 'Text'.
* Stil: Erstelle die neue Sätze in einem konsistenten Stil mit den unten angegebenen Daten. Stell dich sicher, dass die Nachbarsätzen anknüpfen, wobei der Kontext erhalten bleibt. Wenn du als Chirurg eine Hilfe bei der Durchführung dieser Schritte benötigst, z. B. um das Röntgengerät an den richtigen Ort zu fahren, frage an die Assistentin. Wenn notwendige verfahrensbezogene Gespräche bereits abgeschlossen sind, aber noch Textzeilen auszufüllen sind, führ ein tägliches Gespräch mit dem Patienten, um ihn zu beruhigen.
* Format: Gib deine anwort nur auf Deutsch und im Abschnitt < Antwort 2>. Antwort im CSV-Format wie in der Vorlage <Antwort 2> und verwende immer die Tags <Antwort 2> und </Antwort 2> am Anfang und Ende deiner Antwort."""
                prompt += f"\n<Daten 2>\n{sub_print_df.to_csv(index=True, sep=';', index_label='Index')}</Daten 2>\n"
                sub_df['Schritt_Bezeichnung'] = "*Ausfüllen*"
                sub_df['Text'] = sub_df['Text'].replace("*fehlende Daten*", "*Ausfüllen*")
                sub_df = sub_df[['Start_Zeit', 'Phase_Bezeichnung', 'Schritt_Bezeichnung', 'Text']]
                prompt += f"\n<Antwort 2>\n{sub_df.to_csv(index=True, sep=';', index_label='Index')}</Antwort 2>\n"

                # Generate Answer 1.2
                messages.append({"role": "user", "content": prompt})
                answer = get_answer(tokenizer, language_model, messages, max_new_tokens=max_new_tokens)
                
                # Extract tagged block
                block_answer = get_tagged_block(answer, '<Antwort 2>', '</Antwort 2>')
                
                # Add to chat history
                messages.append({"role": "assistant", "content": '\n'.join(block_answer) + '\n'})

                # Check line shapes/errors
                block_answer = line_errors(block_answer, len(sub_df))

                # Check format
                correct_format = check_format(block_answer, ['Index','Start_Zeit', 'Phase_Bezeichnung', 'Schritt_Bezeichnung', 'Text'])
                if not correct_format: raise ValueError(f"Columns or rows do not match")
                
                # Concat
                dfs_to_concat.append(block_to_df(block_answer))
                
                # Output
                if DEBUG_MODE:
                    with open(f'step_1_{i+1}.txt', 'w') as f:
                        for m in messages:
                            f.write('*'*50+' <'+m['role']+'> '+'*'*50+'\n')
                            f.write(m['content'])
                
                # Add to chat
                for m in messages:
                    chat_container.append({"role": m['role'], "content": m['content']})
                
                # Exit loop
                n_try = limit_try
                steps_complete[i] = 1
                
            except:
                n_try += 1
                if PRINT_MODE:
                    print(f'\t\tConnot genreate step_1_{i+1}.txt, will try again')

    df_empty_rows = pd.concat(dfs_to_concat)
    df_empty_rows['Text'] = df_empty_rows['Text'].str.replace('"""', '')
    df_sample.loc[df_empty_rows.index, 'Text'] = df_empty_rows['Text']
    assert np.prod(steps_complete) == 1, "Not all steps completed"

    ####################################### Step 2 #######################################
    #Variables
    limit_try = 3
    dfs_to_concat = []
    tokens_per_row = 60
    
    # Sliding Window
    df_splits = df_splitter(df_sample)
    steps_complete = np.zeros(len(df_splits))
    
    for i, sub_df in enumerate(df_splits):
        # Work data
        sub_df = sub_df[['Start_Zeit', 'Phase_Bezeichnung', 'Text']]
        max_new_tokens = tokens_per_row*len(sub_df)
        
        # Try limit_try times, if LM cant follow instructions
        n_try = 0
        while n_try < limit_try:
            try:
                if PRINT_MODE:
                    print(f'\tStep 2: {int(i+1)}/{len(df_splits)} max new tokens: {max_new_tokens}')
                
                prompt = """In diesem Schritt wirst du die Konversationen umformulieren.
* Daten: Du erhältst einen Datensatz im Abschnitt <Daten>, um ihn umzuformulieren. Die Daten enthalten einen Index, die Startzeit der Rede, den gesprochenen Satz eines Chirurgen und eine Bezeichnung für die Operationsphase. 
* Aufgabe: Deine Aufgabe ist es, die Sätze in der Spalte "Text" im Abschnitt <Daten> umzuformulieren. Schreibe die Sätze so um, dass, wenn das Gespräch mit einer chirurgischen Tätigkeit zusammenhängt, du diesen Kontext beim Umschreiben beibehaltest. Wenn die Konversation keinen Bezug zu einer chirurgischen Tätigkeit hat, führe neue Konversationen.
* Format: Gib deine anwort nur auf Deutsch und im Abschnitt < Antwort 2>. Antwort im CSV-Format wie in der Vorlage <Antwort 2> und verwende immer die Tags <Antwort 2> und </Antwort 2> am Anfang und Ende deiner Antwort."""
                prompt += f"\n<Daten>\n{sub_df.to_csv(index=True, sep=';', index_label='Index')}</Daten>\n"
                sub_df.loc[:, 'Text'] = '*Ausfüllen*'
                prompt += f"\n<Antwort 2>\n{sub_df.to_csv(index=True, sep=';', index_label='Index')}</Antwort 2>\n"

                # Generate Answer 2
                messages = [{"role": "system", "content": role}]
                messages.append({"role": "user", "content":prompt})
                answer = get_answer(tokenizer, language_model, messages, max_new_tokens=max_new_tokens)
                
                # Extract tagged block
                block_answer = get_tagged_block(answer, '<Antwort 2>', '</Antwort 2>')
                
                # Add to chat history
                messages.append({"role": "assistant", "content": '\n'.join(block_answer) + '\n'})

                # Check line shapes/errors
                block_answer = line_errors(block_answer, len(sub_df))
                
                # Check format
                correct_format = check_format(block_answer, ['Index', 'Start_Zeit', 'Phase_Bezeichnung', 'Text'])
                if not correct_format: raise ValueError(f"Columns or rows do not match")
                
                # Order columns                
                df_answer = block_to_df(block_answer)
                df_answer = df_answer[['Start_Zeit', 'Text', 'Phase_Bezeichnung']]
                
                # Check phase-column-not-numeric error
                df_answer['Phase_Bezeichnung'] = df_answer['Phase_Bezeichnung'].apply(pd.to_numeric)
    
                # Check language
                # maybe TODO
                
                # Concat
                dfs_to_concat.append(df_answer)
                
                # Output
                if DEBUG_MODE:
                    with open(f'step_2_{i+1}.txt', 'w') as f:
                        for m in messages:
                            f.write('*'*50+' <'+m['role']+'> '+'*'*50+'\n')
                            f.write(m['content'])
                if LOG_MODE:
                    for m in messages:
                        chat_container.append({"role": m['role'], "content": m['content']})
                
                # Exit loop
                n_try = limit_try
                steps_complete[i] = 1
                
            except:
                n_try += 1
                if n_try == limit_try: 
                    raise ValueError(f"Can't generate data:(")
                
                if PRINT_MODE:
                    print(f'\t\tConnot genreate step_3_{i+1}.txt, will try again')

    # Check if all steps completed
    assert np.prod(steps_complete) == 1, "Not all steps completed"
    
    # Merge dataframes and process
    result_df = pd.concat(dfs_to_concat)
    result_df = result_df.rename(columns={
        'Start_Zeit': 'Start_Time',
        'Phase_Bezeichnung': 'Phase_Label'
    })
    result_df['Text'] = result_df['Text'].str.strip()
    
    # Remove log files
    if DEBUG_MODE:
        for file in listdir('./', ending='.txt'):
            os.remove(file) 
            
    return result_df, chat_container
        

####################################### Main #######################################
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
    model_id = 'mistralai/Mistral-Large-Instruct-2407'
    end_idx = prefix_idx + args.num_target - 1
    
    # Login huggingface environment
    load_dotenv()
    huggingface_hub.login(os.getenv('HF_TOKEN'), add_to_git_credential=False)
    
    # Model
    auto_tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=os.getenv('HF_CACHE_DIR'))
    auto_language_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        cache_dir=os.getenv('HF_CACHE_DIR')
    )

    # Read reference data
    data_path = 'Transcripts/'
    seedset = ['OP_005.csv', 'OP_023.csv', 'OP_027.csv', 'OP_040.csv', 
               'OP_035.csv', 'OP_038.csv', 'OP_013.csv', 'OP_009.csv',
               'OP_011.csv', 'OP_007.csv', 'OP_019.csv', 'OP_002.csv',
               'OP_039.csv', 'OP_026.csv', 'OP_016.csv'] #006 -> 500+, 17, 22, 24, 32 --> 230+
    seedset = sorted([os.path.join('Transcripts/', s) for s in seedset])
    
    # PoCaP 
    pocap = PoCaPCorpus(seedset=seedset, target_path=args.target_path)
    print(f"The seedset: {pocap.dataset}")
    
    # If exists, add synthetic data to pool
    pocap.update_dataset()
    print(f"Length of dataset: {len(pocap.dataset)}")

    # Generate Data
    while prefix_idx <= end_idx and error_count < error_patience:
        try:
            # set save name
            save_name = os.path.join(args.target_path, prefix(prefix_idx+1, 'SynOP_')+".csv")
            print(f"Generating: {save_name}")
        
            # generate data
            generation_start = time.time()
            df, chat_container = gen_data(
                tokenizer=auto_tokenizer,
                language_model=auto_language_model, 
                pocap=pocap 
            )

            # save & update
            df.to_csv(save_name)
            pocap.update_dataset()
            
            # save log
            with open(save_name[:-4] + ".txt", "w") as f:
                for c in chat_container:
                    f.write('\n'+'*'*50+' <'+c['role']+'> '+'*'*50+'\n')
                    f.write(c['content'])
                f.write(f'\nElapsed time:\t{time.time() - generation_start}(s)')
            
            # increment
            prefix_idx += 1
            error_count = 0

        # uppps
        except:
            print(f"\t{save_name} could not generated. Trying again!")
            error_count += 1

    pocap.plot_violin()