import os
import time
import random
import openai
import pandas as pd

from dotenv import load_dotenv


## Functions
def prefix(id, name = '', buffer = 3):
    return name + str(id).zfill(buffer)

## Variables
num_orginal = 3
num_target = 1000

## Target folder
target_path = "SynPoCaP/"
if not os.path.exists(target_path):
    os.mkdir(target_path)

## Load environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

## Get openai client
client = openai.OpenAI()
gpt_model = "gpt-4o" #"gpt-3.5-turbo-0125"

## Read transcriptions data
data_path = 'Transcripts/'
transcripts = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
transcripts.remove(os.path.join(data_path, 'OP_006.csv')) # this op is too long, wasting tokens

## Iter and add data 
for n in range(num_target):
    # prompt
    question = """Ich habe Transkriptionen von chirurgischen Eingriffen. Die Gespräche des Chirurgen während der Portkatheterplatzierung werden aufgezeichnet. Transkriptionen haben eine Startzeit der Ausführung, entsprechenden Text und eine Phasenbezeichnung. Die Phasenbezeichnung gibt an, welche chirurgische Phase stattfindet. Die Port-Cather-Unterbringung besteht aus 8 Phasen und einer Übergangsphase. Dies sind: 0 ist „Preperation“, 1 ist „Puncture“, 2 ist „GuideWire“, 3 ist „CathPlacement“, 
    4 ist „CathPositioning“, 5 ist „CathAdjustment“, 6 ist „CathControl“, 7 ist „Closing“, 8 ist „Transition“.

    Ich habe einen Datensatz mit Transkriptionen und möchte mithilfe dieser Transkriptionen synthetische Daten generieren, um mein Modell zur Erkennung der chirurgischen Phase zu verfeinern. Ich möchte, dass die Eingabe die gleiche Struktur hat, mit der Behandlungszeit, dem Text und der richtigen Phasenbezeichnung. Achte darauf, dass die Sätze des Chirurgen realistisch, entsprechend lang und manchmal beinhaltet gelegentliche tägliche Gespräche sind, die Reihenfolge der Phasen korrekt ist und die Länge des Dokuments den Beispielen entspricht (generiere mindestens 100 Sätze). Antwort nur im CSV-Format durch Semikolon getrennt, keinen weiteren Erklärungtext am Ende hinzufügen.

    Lese die unten aufgeführten Beispiele und generiere neue ähnliche Daten.


    """
    # shuffle transcripts
    random.shuffle(transcripts)
    print(target_path + prefix(n+1, 'SynOP_') + ".csv")
    
    # read example data
    for i, t in enumerate(transcripts[:num_orginal]):
        print("\t"+t)
        df = pd.read_csv(t, index_col=0)
        df = df.drop(columns=['File_Name', 'End_Time'])
        df = df[~df.apply(lambda row: row.astype(str).str.contains('<nicht verstanden>')).any(axis=1)]

        question = question + f"Beispiel {i+1} ist:\n{df.to_string(index=False)}"
            
    # generate response
    call_time = time.time()
    response = client.chat.completions.create(
    model=gpt_model,
    messages=[
        {"role": "system", "content": "Du bist ein Assistent für Synthetischer Datengeneration."},
        {"role": "user", "content": question}
    ]
    )
    res = response.choices[0].message.content

    print(f"\tUsed Tokens: {response.usage.total_tokens}")

    # backup 
    with open("backup.txt", "w") as text_file:
        text_file.write(res)

    # read
    start_time, text, phase_label = [], [], []
    for line in res.split('\n')[2:-1]:
        s_t = line.split(';')[0]
        t = line.split(';')[1]
        p_l = line.split(';')[2]
        
        start_time.append(s_t)
        text.append(t)
        phase_label.append(p_l)
        
    # save
    data = {
        'Start_Time': start_time,
        'Text': text,
        'Phase_Label':phase_label
    }
    
    df = pd.DataFrame(data)
    df.to_csv(target_path + prefix(n+1, 'SynOP_') + ".csv", sep=";")
    
    # wait for TPM limit
    print("waiting...\n")
    time.sleep(max(30 - time.time() + call_time, 0))