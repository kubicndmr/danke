import os
import sys
import time
import json
import random
import openai
import numpy as np
import pandas as pd

from dotenv import load_dotenv


## Functions
def time2sec(time_str, return_ms=False):
    '''
    Converts hh:mm:ss or hh:mm:ss,ms
    to seconds or miliseconds

    sec             : String
                        Time
    return_ms       : Bool
                        Wheter return miliseconds
    '''
    if len(time_str.split(",")) == 2:
        hms = time_str.split(",")[0]
        hours = int(hms.split(":")[0])
        minutes = int(hms.split(":")[1])
        seconds = int(hms.split(":")[2])
        miliseconds = int(time_str.split(",")[1])
    else:
        hours = int(time_str.split(":")[0])
        minutes = int(time_str.split(":")[1])
        seconds = int(time_str.split(":")[2])
        miliseconds = 0
    
    total_seconds = hours * 3600 + minutes * 60 + seconds

    if return_ms:
        return total_seconds * 1000 + miliseconds
    else:
        return total_seconds
    
    
def prefix(id, name='', buffer=3):
    return name + str(id).zfill(buffer)


def get_answer(client, gpt_model, messages):
    answer = client.chat.completions.create(
        model=gpt_model,
        messages=messages
    )
    print(f"\tUsed Tokens: {answer.usage.total_tokens}")
    messages.append({"role": "assistant", "content": answer.choices[0].message.content})
    return messages, answer.choices[0].message.content


def get_tagged_block(answer, start_tag, end_tag):
    block = []
    tag_flag = False

    for line in answer.splitlines():
        if line == start_tag:
            tag_flag = True
            continue
        if line == end_tag:
            tag_flag = False
        if tag_flag:
            block.append(line)
            
    return block
 

def block_to_df(block):
    rows = []
    columns = block[0].split(',')[1:] #ignore index
    text_end_idx = 2 - len(columns)
    for line in block[1:]:
        line = line.split(',')[1:]
        row = [line[0]]
        text = ''.join(line[1:text_end_idx])
        row.append(text.strip('"'))
        for item in line[text_end_idx:]:
            row.append(item)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def insert_empty_rows(input_df, min_phase=10, beta=0.5):
    '''
    Adds empty rows to given DataFrame considering the
    number of sentences in each phase and distribution
    of durations 

    input_df        : DataFrame
                        DataFrame with surgery releated sentences
    min_phase       : Int
                        Minimum number of sentences to exist in each phase
    beta            : Float
                        Percentage of the total length of the DataFrame 
                        to be added as empty rows. Can be exceed due to 
                        min_phase value. 
    '''
    subt_length = int(len(input_df)*beta)
    
    to_insert = {label: max(0, min_phase-count) 
                 for label, count in input_df['Phase_Label'].value_counts().items()
                 if label!='8'}
    if np.sum(list(to_insert.values())) < subt_length:
        diff = (subt_length - np.sum(list(to_insert.values())))//8
        to_insert = {key: value+diff for key, value in to_insert.items()}
    
    dfs = []
    for label, pdf in input_df.groupby('Phase_Label'):
        if label!='8':
            N = to_insert[str(label)]
            random_positions = np.random.choice(np.arange(1,len(pdf)+N+1), size=N, replace=False)
            for pos in random_positions:
                empty_row = pd.DataFrame({'Start_Time': [np.nan], 'Text': 'NaN', 'Phase_Label': label})
                pdf = pd.concat([pdf.iloc[:pos], empty_row, pdf.iloc[pos:]]).reset_index(drop=True)
            
            dfs.append(pdf)

    df = pd.concat(dfs, ignore_index=True)
    
    # Fill Start_Time column
    df['Start_Time'] = df['Start_Time'].astype(float)
    nan_indices = df[df['Start_Time'].isna()].index
    
    for idx in nan_indices:
        upper_value = df['Start_Time'].loc[:idx].ffill().iloc[-1]
        lower_value = df['Start_Time'].loc[idx:].bfill().iloc[0]
        random_value = np.random.uniform(upper_value, lower_value)
        df.at[idx, 'Start_Time'] = random_value
    
    df['Start_Time'] = df['Start_Time'].apply(pd.to_numeric)
    df['Start_Time'] = df['Start_Time'].apply(add_random_value)
    df['Start_Time'] = df['Start_Time'].round(2)
    
    return df


def add_random_value(x):
    return x + np.random.rand()*0.1


def listdir(path, ending=None):
    '''Returns dir with full path'''
    if ending == None:
        return sorted([os.path.join(path, f) for f in os.listdir(path)])
    else:
        return sorted([os.path.join(path, f) for f in os.listdir(path) 
                       if f.endswith(ending)])
        

def get_phase_dist(annot_file, trainset):
    phase_dur = np.zeros((len(trainset),9))
    op_idx = 0
    
    with open(annot_file) as f:
        annots = json.load(f)
        
        for annot in annots:
            if annot[0]+'.csv' in trainset:
                time_stamps = annot[3]
                phases = annot[4]
                
                for p, t in zip(phases, time_stamps):
                    t = t.split('-')
                    t_start = time2sec(t[0])
                    t_end = time2sec(t[1]) + 1
                    if p == 'Preparation':
                        phase_dur[op_idx, 0] += t_end - t_start
                    if p == 'Puncture':
                        phase_dur[op_idx, 1] += t_end - t_start
                    if p == 'GuideWire':
                        phase_dur[op_idx, 2] += t_end - t_start
                    if p == 'CathPlacement':
                        phase_dur[op_idx, 3] += t_end - t_start
                    if p == 'CathPositioning':
                        phase_dur[op_idx, 4] += t_end - t_start
                    if p == 'CathAdjustment':
                        phase_dur[op_idx, 5] += t_end - t_start
                    if p == 'CathControl':
                        phase_dur[op_idx, 6] += t_end - t_start
                    if p == 'Closing':
                        phase_dur[op_idx, 7] += t_end - t_start
                    if p == 'Transition':
                        phase_dur[op_idx, 8] += t_end - t_start
                    
                op_idx += 1
                        
    mean_values = np.mean(phase_dur, axis=0)
    std_values = np.std(phase_dur, axis=0)
    
    mean_dict = {key: value for key, value in zip(range(8), mean_values)}
    std_dict = {key: value for key, value in zip(range(8), std_values)}
    
    return [mean_dict, std_dict]
    
      
def gen_data():
    # Step 1
    role = """Ich bin ein Chirurg in der Radilogie, der die Port-Katheter Platzierung Operationen durchführt. Ich spreche mit dem Assitent und dem Patient während der Operation."""
        
    prompt = """Deine Aufgabe ist es, Gespräche eines Chirurgen mit dem medizinischen Assistenten und dem Patienten während einer Port-Katheter-Platzierung zu generieren. Du wirst deine Daten in einem konsistenten Stil mit dem unten angegebenen Beispiel generieren. Die generierten Daten werden für das Training eines textbasierten Deep-Learning-Modells verwendet, das für die Schätzung der chirurgischen Phasen der Port-Katheter-Placement-Operation entwickelt wurde.
    
Führe die folgenden Aktionen durch, um eine neue Daten zu erstellen:

1. Das Legen eines Portkatheters ist eine minimalinvasive Operation in der Radiologie und besteht aus acht Phasen. Erläutere diese Phasen mit je einem Satz im <Antwort 1> Bereich.
<Antwort 1>
0. 
1. 
2. 
3. 
4. 
5. 
6.  
7. 
<\Antwort 1>

2. Analysiere jeden Satz in der Spalte „Text“ der unten angegebenen Beispieldaten <Antwort 2> und bestimme, ob die Sätze für die Erkennung der in der Spalte „Phase_Label“ angegebenen chirurgischen Phasen wichtig sind. Verwende die <Antwort 2> als Vorlage und fülle die leere Spalte 'Relevanz' aus, indem Sie 'P' für Sätze, die für die Erkennung chirurgischer Phasen relevant sind, und 'D' für Sätze, die im Kontext einer täglichen Unterhaltung stehen, eintragen. Gebe deine Antwort in den Bereich <Antwort 2> ein. Antworte nur im CSV-Format, getrennt durch Komma.
"""
    random.shuffle(transcripts)
    df = pd.read_csv(transcripts[0], index_col=0)
    df["Relevance"] = ""
    df = df.drop(columns=['File_Name', 'End_Time'])
    df = df[df['Text'] != '<nicht verstanden>']
    df = df[df['Phase_Label'] != '8']
    prompt = prompt + f"\n<Antwort 2>\n{df.to_csv()}<\Antwort 2>"
    
    messages=[{"role": "system", "content": role}, {"role": "user", "content":prompt}]
    messages, answer = get_answer(client, gpt_model, messages)
    
    # Step 2
    answer_2 = get_tagged_block(answer, '<Antwort 2>', '<\Antwort 2>')
    answer_2 = block_to_df(answer_2)
    answer_2 = answer_2[answer_2['Relevance'] != 'D']
    answer_2 = answer_2.drop(columns=['Relevance'])
        
    prompt = f"""3. Jetzt hast du die Daten ohne tägliche Gespräche, wie sie im Bereich <Daten> dargestellt sind. 
<Daten>
{answer_2.to_csv()}
<\Daten>

Verwende diese Daten als Vorlage. Erstelle neue, eigene Daten, indem du den Gesprächen in diesen Daten folgst. Reproduziere die Ereignisse, die in den Daten vorkommen, aber drücke dich mit deinen eigenen Sätzen aus. Fülle die Spalten Start_Time und Phase_Label entsprechend aus. Gebe deine Antwort im Abschnitt <Antwort 3> an und füge so viele Datenzeilen wie nötig hinzu. Antworte nur im CSV-Format, getrennt durch Komma.
<Antwort 3>
Index,Start_Time,Text,Phase_Label
0,
1,
2,
3,
4,
...
<\Antwort 3>
"""
    messages.append({"role": "user", "content":prompt})
    messages, answer = get_answer(client, gpt_model, messages)
            
    # step 4
    answer_3 = get_tagged_block(answer, '<Antwort 3>', '<\Antwort 3>')
    answer_3 = block_to_df(answer_3)
    answer_3 = insert_empty_rows(answer_3)
        
    prompt = f"""4. Eine erweiterte Version der Daten ist unten im Abschnitt <Daten> mit fehlenden Gesprächen angegeben.
<Daten>
{answer_3.to_csv(na_rep='NaN')}
<\Daten>

Fülle die Textfelder, die NaN-Werten haben, mit tatsächlichen Gesprächen. Ahmen Sie den Sprachstil in den Daten nach. Jedes Gespräch muss auf Deutsch sein. Geben Sie Ihre Antwort im Abschnitt <Antwort 4>. Antworte nur im CSV-Format, getrennt durch Komma.

<Antwort 4>
Index,Start_Time,Text,Phase_Label
0,
1,
2,
3,
4,
...
<\Antwort 4>
"""
    messages.append({"role": "user", "content":prompt})
    messages, answer = get_answer(client, gpt_model, messages)
        
    # Get answer  
    result = get_tagged_block(answer, '<Antwort 4>', '<\Antwort 4>')    
    df = block_to_df(result)

    return df, messages
        
            
## Code
if __name__ == "__main__":
    # Target folder
    target_path = "SynPoCaP/"
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # Variables
    num_target = 50
    idx = len(listdir(target_path ,ending='.pkl'))

    # Load environment
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Get openai client
    client = openai.OpenAI()
    gpt_model = "gpt-4o" #"gpt-3.5-turbo-0125"

    # Read data
    data_path = 'Transcripts/'
    trainset = ['OP_009.csv', 'OP_012.csv', 'OP_033.csv', 'OP_034.csv', 
                'OP_026.csv', 'OP_029.csv', 'OP_025.csv', 'OP_036.csv', 
                'OP_017.csv', 'OP_013.csv', 'OP_016.csv', 'OP_030.csv', 
                'OP_005.csv', 'OP_040.csv', 'OP_031.csv']
    transcripts = [os.path.join(data_path, f) for f in trainset]

    # Generate data
    while idx <= num_target:
        print(target_path + prefix(idx+1, 'SynOP_') + ".csv")
        
        try:
            # generate data
            first_call = time.time()
            df, messages = gen_data()
        
            # Save answer
            df.to_csv(target_path + prefix(idx+1, 'SynOP_') + ".csv", index=False)
            
            # Save messages
            with open(target_path + prefix(idx+1, 'SynOP_') + ".txt", "w") as text_file:
                text_file.write('Refence Data:'+transcripts[0]+'\n')
                for m in messages:
                    text_file.write('\n'+'*'*50+' <'+m['role']+'> '+'*'*50+'\n')
                    text_file.write(m['content'])
            
            #index increment
            idx += 1
        except:
            print(target_path + prefix(idx+1, 'SynOP_') + ".csv could not generated")
            print('Trying again!')
            
        # wait for TPM limit‚
        print("waiting...\n")
        time.sleep(max(60 - time.time() + first_call, 0))