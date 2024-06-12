import os
import sys
import time 
import random
import openai
import numpy as np
import pandas as pd

from dotenv import load_dotenv


## Functions
def prefix(id, name = '', buffer = 3):
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


def insert_empty_rows(df, beta = 0.8):
    N = int(len(df)*beta)
    
    random_positions = np.random.choice(np.arange(1,len(df)+N-1), size=N, replace=False)
    for pos in sorted(random_positions):
        empty_row = pd.DataFrame({col: [np.nan] for col in df.columns})
        df = pd.concat([df.iloc[:pos], empty_row, df.iloc[pos:]]).reset_index(drop=True)
    
    # Fill Phase_Label column
    df['Phase_Label'] = df['Phase_Label'].fillna(method='ffill')
    
    # Fill Start_Time column
    df['Start_Time'] = df['Start_Time'].astype(float)
    nan_indices = df[df['Start_Time'].isna()].index
    
    for idx in nan_indices:
        upper_value = df['Start_Time'].loc[:idx].ffill().iloc[-1]
        lower_value = df['Start_Time'].loc[idx:].bfill().iloc[0]
        random_value = np.random.uniform(upper_value, lower_value)
        df.at[idx, 'Start_Time'] = random_value
    
    return df


def add_random_value(x):
    return x + np.random.rand()*0.1


## Code
if __name__ == "__main__":
    # Variables
    num_target = 10

    # Target folder
    target_path = "SynPoCaP/"
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # Load environment
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Get openai client
    client = openai.OpenAI()
    gpt_model = "gpt-4o" #"gpt-3.5-turbo-0125"

    # Read data
    data_path = 'Transcripts/'
    trainset = ['OP_009.csv', 'OP_003.csv', 'OP_012.csv', 'OP_033.csv', 'OP_034.csv', 
                'OP_024.csv', 'OP_026.csv', 'OP_029.csv', 'OP_025.csv', 'OP_036.csv', 
                'OP_002.csv', 'OP_017.csv', 'OP_006.csv', 'OP_013.csv', 'OP_016.csv', 
                'OP_005.csv', 'OP_040.csv', 'OP_031.csv', 'OP_030.csv', 'OP_014.csv']
    transcripts = [os.path.join(data_path, f) for f in trainset]

    # Generate data
    for n in range(num_target):
        print(target_path + prefix(n+1, 'SynOP_') + ".csv")
        
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

2. Analysiere jeden Satz in der Spalte „Text“ der unten angegebenen Beispieldaten <Antwort 2> und bestimme, ob die Sätze für die Erkennung der in der Spalte „Phase_Label“ angegebenen chirurgischen Phasen wichtig sind. Verwende die <Antwort 2> als Vorlage und fülle die leere Spalte 'Relevanz' aus, indem Sie 'P' für Sätze, die für die Erkennung chirurgischer Phasen relevant sind, und 'D' für Sätze, die im Kontext einer täglichen Unterhaltung stehen, eintragen. Gebe deine Antwort in den Bereich <Antwort 2> ein.
"""
        random.shuffle(transcripts)
        df = pd.read_csv(transcripts[0], index_col=0)
        df["Relevance"] = ""
        df = df.drop(columns=['File_Name', 'End_Time'])
        df = df[df['Text'] != '<nicht verstanden>']
        df = df[df['Phase_Label'] != '8']
        prompt = prompt + f"\n<Antwort 2>\n{df.to_csv()}<\Antwort 2>"
        
        first_call = time.time()
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

Verwende diese Daten als Vorlage. Erstelle neue, eigene Daten, indem du den Gesprächen in diesen Daten folgst. Reproduziere die Ereignisse, die in den Daten vorkommen, aber drücke dich mit deinen eigenen Sätzen aus. Fülle die Spalten Start_Time und Phase_Label entsprechend aus. Gebe deine Antwort im Abschnitt <Antwort 3> an und füge so viele Datenzeilen wie nötig hinzu. 
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

Fülle die Textfelder, die NaN-Werten haben, mit tatsächlichen Gesprächen. Ahmen Sie den Sprachstil in den Daten nach. Jedes Gespräch muss auf Deutsch sein. Geben Sie Ihre Antwort im Abschnitt <Antwort 4>.

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
        
        # Add random noise to Start_Time
        df['Start_Time'] = df['Start_Time'].apply(add_random_value)
        df['Start_Time'] = df['Start_Time'].round(2)
        
        # Save answer
        df.to_csv(target_path + prefix(n+1, 'SynOP_') + ".csv", index=False)
        
        # Save messages
        with open(target_path + prefix(n+1, 'SynOP_') + ".txt", "w") as text_file:
            text_file.write('Refence Data:'+transcripts[0]+'\n')
            for m in messages:
                text_file.write('\n'+'*'*50+' <'+m['role']+'> '+'*'*50+'\n')
                text_file.write(m['content'])
                
        # wait for TPM limit‚
        print("waiting...\n")
        time.sleep(max(60 - time.time() + first_call, 0))