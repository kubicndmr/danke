import os
import time
import random
import openai
import numpy as np
import pandas as pd

from dotenv import load_dotenv


## Functions    
def prefix(id, name='', buffer=3):
    return name + str(id).zfill(buffer)


def get_answer(client, gpt_model, messages, temperature,
               max_tokens=4000, presence_penalty=0.1):
    '''
    Calls the API and returns the asnwer
    
    For parameters, see: https://platform.openai.com/docs/api-reference/chat/create
    '''
    answer = client.chat.completions.create(
        model=gpt_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty
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
    
    columns = block[0].split(',')  
    text_end_idx = 3 - len(columns)
    
    for line in block[1:]:
        line = line.split(',')
        row = [line[0], line[1]]
        text = ''.join(line[2:text_end_idx])
        row.append(text.strip('"'))
        for item in line[text_end_idx:]:
            row.append(item)
        rows.append(row)
    
    # Create the DataFrame
    df = pd.DataFrame(rows, columns=columns)
    
    # Set the first column as the index
    df['Index'] = df['Index'].apply(pd.to_numeric)
    df.set_index('Index', inplace=True)
    
    return df


def insert_empty_rows(input_df, min_phase=10, beta=0.4):
    '''
    Adds empty rows to given DataFrame considering the
    number of sentences in each phase and total length

    input_df        : DataFrame
                        DataFrame with surgery releated sentences
    min_phase       : Int
                        Minimum number of sentences to exist in each phase
    beta            : Float
                        Percentage of the total length of the DataFrame 
                        to be added as empty rows. Can be exceed due to 
                        min_phase value.
    '''
    tag = "<Füll aus>"
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
                empty_row = pd.DataFrame({'Start_Time': [np.nan], 'Text': tag, 'Phase_Label': label})
                pdf = pd.concat([pdf.iloc[:pos], empty_row, pdf.iloc[pos:]]).reset_index(drop=True)
            
            dfs.append(pdf)

    df = pd.concat(dfs, ignore_index=True)
    
    # Fill Start_Time column -> TODO could better,i.e., sample random value between previous and next value
    df['Start_Time'] = df['Start_Time'].apply(pd.to_numeric)
    df['Start_Time'] = df['Start_Time'].ffill()
    df['Start_Time'] = df['Start_Time'].apply(lambda x: x + np.random.uniform())
    df['Start_Time'] = df['Start_Time'].round(2)
    
    df.reset_index(drop=True, inplace=True)
    
    empty_rows = df[df['Text'] == tag]
    return df, empty_rows


def listdir(path, ending=None):
    '''Returns dir with full path'''
    if ending == None:
        return sorted([os.path.join(path, f) for f in os.listdir(path)])
    else:
        return sorted([os.path.join(path, f) for f in os.listdir(path) 
                       if f.endswith(ending)])
        

def drop_sentences(df, beta=0.2):
    '''
    This functions removes randomly selected rows from given dataframe.
    df              : DataFrame
                        DataFrame with surgery releated sentences
    beta            : Float
                        Percentage of the total length of the DataFrame
                        to keep
    '''
    df = df.drop(columns=['File_Name', 'End_Time'], errors='ignore')
    df = df[df['Text'] != '<nicht verstanden>']
    df = df[df['Phase_Label'] != '8']
    return df.drop(df.sample(n=int(len(df)*beta)).index)   

      
def gen_data(client, gpt_model, sample_data):
    '''
    Generates synthetic data
    
    sample_data    : String
                        Path to example data
    '''    
    # Step 1
    print('\tStep 1:', end='')
    role = """Ich bin ein Chirurg in der Radilogie, der die Port-Katheter Platzierung Operationen durchführt."""
    prompt = """Deine Aufgabe ist es, Gespräche eines Chirurgen mit dem medizinischen Assistenten und dem Patienten während einer Port-Katheter-Platzierung zu generieren. Du wirst deine Daten in einem konsistenten Stil mit dem unten angegebenen Beispiel generieren. Die generierten Daten werden für das Training eines textbasierten Deep-Learning-Modells verwendet, das für die Schätzung der chirurgischen Phasen der Port-Katheter-Placement-Operation entwickelt wurde. alle Gespräche müssen nur auf Deutsch geführt werden.
    
Führe die folgenden drei Aktionen durch, um eine neue Daten zu erstellen:

*1. Das Legen eines Portkatheters ist eine minimalinvasive Operation in der Radiologie und besteht aus acht Phasen. Erläutere diese Phasen mit je einem Satz im <Antwort 1> Bereich.
<Antwort 1>
Phase_Label, Beschreibung
0, 
1, 
2, 
3, 
4, 
5, 
6,  
7, 
<\Antwort 1>
"""
    messages=[{"role": "system", "content": role}, {"role": "user", "content":prompt}]
    messages, answer_1 = get_answer(client, gpt_model, messages, temperature=0.1)

    with open('prompt_1.txt', 'w') as f:
        print(prompt, file=f)
    with open('answer_1.txt', 'w') as f:
        print(answer_1, file=f)

    # Step 2
    print('\tStep 2:', end='')
    prompt = """*2. Du erhaltest einen Datensatz mit einer Reihe von Sätzen im Abschnitt <Daten>. Die Daten enthalten einen Index, die Startzeit der Rede, den gesprochenen Satz und eine Bezeichnung für die Operationsphase. Als ein Chirurg, schreibe diese Sätze in der Spalte "Text" um, um die Ereignisse im Text wiederzugeben, aber drücke dich mit deinen eigenen Sätzen aus. Generiere deinen eigenen Sätzen (nicht mit Passivsätzen in der dritten Person). Verwende für deine Antwort die Vorlage mit Spaltennamen, die Sie im Abschnitt <Antwort 2> finden. Antworte nur im CSV-Format mit <Antwort 2> Tags, getrennt durch Komma."""
    
    data_1 = pd.read_csv(sample_data, index_col=0)
    data_1 = drop_sentences(data_1, beta=np.random.uniform(0.4, 0.6))
    data_1.reset_index(drop=True, inplace=True)
    prompt = prompt + f"\n<Daten>\n{data_1.to_csv(index=True, index_label='Index')}<\Daten>"
    prompt = prompt + """\n\nDeine Antwort:
<Antwort 2>
Index,Start_Time,Text,Phase_Label
0,
1,
2,
3,
...
<\Antwort 2>
""" 
    messages.append({"role": "user", "content":prompt})
    messages, answer_2 = get_answer(client, gpt_model, messages, temperature=0.25)

    with open('prompt_2.txt', 'w') as f:
        print(prompt, file=f)
    with open('answer_2.txt', 'w') as f:
        print(answer_2, file=f)
        
    # Step 2
    print('\tStep 3:', end='')
    data_2 = get_tagged_block(answer_2, '<Antwort 2>', '<\Antwort 2>')
    data_2 = block_to_df(data_2)
    data_2, empty_rows = insert_empty_rows(data_2, beta=np.random.uniform(0.5, 0.8))
    
    prompt = """*3. Der Datensatz aus dem vorherigen Schritt hat sich um neue Zeilen erweitert. Deine Aufgabe ist es, die Spalte "Text" dieser Zeilen zu füllen. Fülle diese Zeilen in der Spalte 'Text' als der Chirurg mit deinen eigenen Sätzen (nicht mit Passivsätzen in der dritten Person) unter Berücksichtigung der Phasenbezeichnungen und des Kontexts. Verwende für deine Antwort die Vorlage der leeren Zeilen, die Sie im Abschnitt <Antwort 3> finden. Antworte nur im CSV-Format mit <Antwort 3> Tags, getrennt durch Komma."""
    prompt = prompt + f"\n\nDeine Antwort:\n<Antwort 3>\n{empty_rows.to_csv(index=True, index_label='Index')}<\Antwort 3>"

    messages.append({"role": "user", "content":prompt})
    messages, answer_3 = get_answer(client, gpt_model, messages, temperature=0.75)
    
    with open('prompt_3.txt', 'w') as f:
        print(prompt, file=f)
    with open('answer_3.txt', 'w') as f:
        print(answer_3, file=f)
        
    # Get answer  
    data_3 = get_tagged_block(answer_3, '<Antwort 3>', '<\Antwort 3>')
    data_3 = block_to_df(data_3)
    
    # Result
    result = pd.concat([data_2, data_3])
    result = result[result['Text'] != '<Füll aus>']
    result = result.sort_index()

    os.remove('prompt_1.txt')
    os.remove('answer_1.txt')
    os.remove('prompt_2.txt')
    os.remove('answer_2.txt')
    os.remove('prompt_3.txt')
    os.remove('answer_3.txt')

    return result, messages
        

## Code
if __name__ == "__main__":
    # Target folder
    target_path = "SynPoCaP/"
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # Variables
    num_target = 200
    idx = len(listdir(target_path ,ending='.csv'))
    error_patience = 3
    error_count = 0
    
    # Load environment
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Get openai client
    client = openai.OpenAI()
    gpt_model = "gpt-3.5-turbo-0125" 

    # Read data
    data_path = 'Transcripts/'
    trainset = ['OP_009.csv', 'OP_012.csv', 'OP_033.csv', 'OP_034.csv', 
                'OP_026.csv', 'OP_029.csv', 'OP_025.csv', 'OP_036.csv', 
                'OP_017.csv', 'OP_013.csv', 'OP_030.csv', 
                'OP_005.csv', 'OP_040.csv', 'OP_031.csv'] # some bad ops removed
    trainset = [os.path.join(data_path, f) for f in trainset]
    
    while idx <= num_target and error_count < error_patience:
        print(target_path + prefix(idx+1, 'SynOP_') + ".csv")
        
        # select sample data
        random.shuffle(trainset)
        sample_data = trainset[0]
        
        # generate data, save and log
        try:
            first_call = time.time()
            df, messages = gen_data(client, gpt_model, sample_data)

            df.to_csv(target_path + prefix(idx+1, 'SynOP_') + ".csv")
            trainset.append(target_path + prefix(idx+1, 'SynOP_') + ".csv")
            
            with open(target_path + prefix(idx+1, 'SynOP_') + ".txt", "w") as text_file:
                text_file.write('Refence Data:'+sample_data+'\n')
                for m in messages:
                    text_file.write('\n'+'*'*50+' <'+m['role']+'> '+'*'*50+'\n')
                    text_file.write(m['content'])
                error_count = 0
            
            idx += 1
        
        # uppps
        except:
            print(target_path + prefix(idx+1, 'SynOP_') + ".csv could not generated")
            print('Trying again!')
            error_count += 1
            
        # wait for TPM limit‚
        print("waiting...\n")
        #time.sleep(max(60 - time.time() + first_call, 0))