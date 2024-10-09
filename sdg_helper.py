import os
import copy
import random
import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde

# Variables
transcript_path = 'Transcripts/'
transcript_set = ['OP_002.csv', 'OP_003.csv', 'OP_004.csv',
                  'OP_007.csv', 'OP_009.csv', 'OP_011.csv', 'OP_013.csv',
                  'OP_016.csv', 'OP_017.csv', 'OP_019.csv',
                  'OP_024.csv', 'OP_026.csv', 'OP_029.csv', 'OP_030.csv',
                  'OP_032.csv', 'OP_038.csv', 'OP_039.csv', 'OP_040.csv']  # 006 -> 500+, 22 --> 230+
transcripts = [os.path.join('Transcripts/', s)
               for s in transcript_set]

topics = pd.read_csv('Utils/topics.csv', index_col=0)
personae = pd.read_csv('Utils/personae.csv', index_col=0)
examples = pd.read_csv('Utils/examples.csv')
problems = pd.read_csv('Utils/problems.csv', index_col=0)

surgical_phases = {
    0: 'Vorbereitung', 1: 'Punktion', 2: 'Führungsdraht',
    3: 'Pouchvorbereitung-und-Katheterplatzierung', 4: 'Katheterpositionierung', 5: 'Katheteranpassung',
    6: 'Katheterkontrolle', 7: 'Abschluss'
}
reversed_surgical_phases = {
    value: key for key, value in surgical_phases.items()}

surgical_steps = {
    0: ['Positionierung des Patienten auf dem Tisch', 'Tisch fährt hoch', 'Radiologe sterilisiert sich', 'Vorbereitung des sterilen Materials', 'Patient steril abgedeckt'],
    1: ['Lokale Anästhesie', 'Ultraschallgeführte Punktion'],
    2: ['Röntgenmaschine fährt ein', 'Durchleuchtung im Bereich der Subklavia', 'Durchleuchtung im Bereich der Vena cava inferior (VCI)', 'Röntgenmaschine fährt heraus'],
    3: ['Lokale Anästhesie', 'Inzision', 'Pouch-Vorbereitung', 'Hülleplatzierung'],
    4: ['Röntgenmaschine fährt ein', 'Durchleuchtung des VCI-Bereichs', 'Positionierung des Katheters'],
    5: ['Kürzen des Katheters', 'Röntgenmaschine fährt aus', 'Anschluss des Katheters an die Portkapsel', 'Positionierung der Portkapsel im Pouch', 'Chirurgische Naht', 'Punktion der Portkapsel'],
    6: ['Röntgenmaschine fährt ein', 'Digitale Subtraktionsangiographie des Brust', 'Röntgenmaschine fährt in Parkposition aus'],
    7: ['Steriles Pflaster auflegen', 'Tisch fährt nach unten']
}

# Functions


def prefix(id, name='', buffer=5):
    return name + str(id).zfill(buffer)


def listdir(path, ending=None):
    '''Returns dir with full path'''
    if ending == None:
        return sorted([os.path.join(path, f) for f in os.listdir(path)])
    else:
        return sorted([os.path.join(path, f) for f in os.listdir(path)
                       if f.endswith(ending)])


def extract_model_response(answer, instruction_tag='[/INST]'):
    model_response = []
    response_flag = False

    for line in answer.splitlines():
        if instruction_tag in line:
            response_flag = True
            if line.startswith(instruction_tag):
                line = line[len(instruction_tag):]
                line = line.strip()
        if response_flag:
            model_response.append(line)

    return '\n'.join(model_response)


def get_tagged_block(answer, start_tag, end_tag):
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

    for line in answer.splitlines():
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

    if len(block) == 0:
        search_open = True
        for line in reversed(answer.split('\n')):
            if line != '':
                if line[0].isdigit() and search_open:
                    block.append(line)
                if line.startswith('Index'):
                    block.append(line)
                    search_open = False
        block = [b for b in reversed(block)]

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


def line_errors(block, true_len, delimeter=";"):
    if true_len == len(block) - 1:
        # delimeter ending error
        tested_block = [block[0]]
        for line in block[1:]:
            if line.endswith(delimeter):
                tested_block.append(line[:-1])
            else:
                tested_block.append(line)
        return tested_block
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


def df_splitter(df, max_df_length=5):
    split_dfs = []
    temp_df = pd.DataFrame()

    prev_person = None

    for i, row in df.iterrows():
        current_person = row['Person']

        if current_person != prev_person or len(temp_df) >= max_df_length:
            if not temp_df.empty:
                split_dfs.append(temp_df)
            temp_df = pd.DataFrame()

        temp_df = pd.concat([temp_df, pd.DataFrame([row])], ignore_index=False)
        prev_person = current_person

    if not temp_df.empty:
        split_dfs.append(temp_df)

    return split_dfs


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


def phase_count_limits(dataset):
    tolerence_percentage = 0.4
    min_upper_limit = 10
    min_phase_length = 2
    phase_count = np.zeros((len(dataset), 8), dtype=int)

    for i, d in enumerate(dataset):
        array_count = np.zeros(8, dtype=int)
        df = pd.read_csv(d)
        phases = df['Phase_Label'].value_counts().drop(8, errors='ignore')
        array_count[phases.index] = phases.values
        phase_count[i, :] = array_count

    # compute quantiles
    lower_limit = np.min(phase_count, axis=0)
    upper_limit = np.max(phase_count, axis=0)

    lower_limit -= (lower_limit * tolerence_percentage).astype(int)
    upper_limit += (upper_limit * tolerence_percentage).astype(int)

    lower_limit[lower_limit == 0] = min_phase_length
    upper_limit[upper_limit < min_upper_limit] = min_upper_limit

    return lower_limit, upper_limit, phase_count


def sample_phase_lengths():
    lower_limit, upper_limit, phase_count = phase_count_limits(
        dataset=transcripts)
    phase_lengths = np.zeros(8, dtype=int)

    for phase in range(8):
        current_phase_data = phase_count[:, phase]

        kde = gaussian_kde(current_phase_data[current_phase_data > 0])

        sampled_length = kde.resample(1)[0][0]

        phase_lengths[phase] = int(
            np.clip(sampled_length, lower_limit[phase], upper_limit[phase]))

    return phase_lengths, 1 - phase_lengths/upper_limit


def time_count_limits(dataset):
    phase_start = np.zeros((len(dataset), 8))

    # Read dataset
    for i, d in enumerate(dataset):
        df = pd.read_csv(d)

        df = df[df['Phase_Label'] != 8]

        for phase in range(8):
            phase_df = df[df['Phase_Label'] == phase]

            if not phase_df.empty:
                start_time = phase_df['Start_Time'].iloc[0]
                phase_start[i, phase] = start_time

    phase_start_low = np.min(phase_start, axis=0)
    phase_start_high = np.max(phase_start, axis=0)

    # Add randomness to time values
    random_addition = np.random.uniform(-np.min(phase_start_low)*0.9, 1000)
    phase_start_low += random_addition
    phase_start_high += random_addition

    return phase_start_low, phase_start_high


def sample_time_stamps(phase_lengths):
    idx = 0
    time_stamps = np.zeros(np.sum(phase_lengths))
    lower_limit, upper_limit = time_count_limits(dataset=transcripts)

    for phase in range(8):
        time_stamps[idx:idx+phase_lengths[phase]] = np.random.uniform(
            lower_limit[phase], upper_limit[phase], size=phase_lengths[phase])
        idx += phase_lengths[phase]

    return np.sort(time_stamps)


def sample_daily_percentage():  # values precomputed
    return 0.28 + np.random.uniform(-0.07, 0.07)


def sample_daily_topic():
    return topics.loc[np.random.randint(1, len(topics)), 'Thema']


def sample_radiologe():
    return personae.loc[np.random.randint(1, len(personae)), 'Radiologe']


def sample_assistent():
    return personae.loc[np.random.randint(1, len(personae)), 'Assistent']


def sample_patient():
    return personae.loc[np.random.randint(1, len(personae)), 'Patient']


def sample_pocap_example(phase, n_examples):
    df_phase = examples.loc[examples['Phase_Label']
                            == reversed_surgical_phases[phase]].copy()
    df_sample = df_phase.sample(n=n_examples, replace=False)
    df_sample = df_sample.drop(columns=['Phase_Label'])
    df_sample['Text'] = df_sample['Text'].apply(lambda x: f'Satz: <{x}>')
    df_sample['Explanation'] = df_sample['Explanation'].apply(
        lambda x: f', Erklärung: <{x}>')

    return df_sample.to_csv(index=False, sep='\t', header=False, lineterminator="\n\t")


def sample_problem(complication_probability=0.3):
    random_index = np.random.randint(low=0, high=int(
        len(problems)/complication_probability))
    if random_index <= len(problems):
        return problems.loc[random_index, 'complication']
    else:
        return None


def draft_OP():
    phase_dfs = []
    patient_percentage = random.choice([0.25, 0.3, 0.35, 0.4])
    assistant_percentage = random.choice([0.25, 0.3, 0.35, 0.4])
    phase_lengths, phase_length_scale = sample_phase_lengths()
    daily_percentage = sample_daily_percentage()
    time_stamps = sample_time_stamps(phase_lengths)

    # Create phasewise dataframes
    for phase in range(8):
        phase_df = pd.DataFrame(index=range(phase_lengths[phase]))
        steps = surgical_steps[phase]

        # In preperation and closing phases, step order can change
        if phase in [0, 7]:
            random.shuffle(steps)

        # Randomly remove some steps, proportional to relative length of the phase
        steps_omit_percentage = min(random.choice(
            [0.5, 0.6, 0.7]), phase_length_scale[phase])
        steps_to_remove = random.sample(steps, min(
            len(steps) - 1, int(len(steps)*steps_omit_percentage)))
        steps = [step for step in steps if step not in steps_to_remove]

        # Fill
        if len(phase_df) > len(steps):
            phase_df['Schritt'] = np.repeat(steps, int(
                np.ceil(len(phase_df) / len(steps))))[:len(phase_df)]
        else:
            phase_df['Schritt'] = steps[:len(phase_df)]

        phase_df['Phase'] = phase

        phase_dfs.append(phase_df)

    # Combine dataframe
    df = pd.concat(phase_dfs)
    df = df.reset_index(drop=True)
    df['Text'] = '*Ausfüllen*'
    df['Startzeit'] = time_stamps
    df['Startzeit'] = df['Startzeit'].astype(float).round(3)

    # Add daily conversations
    random_indices = np.random.choice(df.index,
                                      size=int(len(df)*daily_percentage), replace=False)
    df.loc[random_indices, 'Schritt'] = 'Alltäglich'

    # Add talking person
    df['Person'] = 'Radiologe'

    num_patient_rows = int(len(df) * patient_percentage)
    num_assistant_rows = int(len(df) * assistant_percentage)

    # Patient
    random_indices = np.random.choice(
        df.index, size=num_patient_rows, replace=False)

    for index in random_indices:
        new_row = pd.DataFrame({
            'Startzeit': [np.nan],
            'Schritt': [np.nan],
            'Phase': [np.nan],
            'Person': ['Patient'],
            'Text': ['*Ausfüllen*']
        })

        df = pd.concat([df.iloc[:index], new_row,
                       df.iloc[index:]]).reset_index(drop=True)

    # Assistant
    random_indices = np.random.choice(
        df.index, size=num_assistant_rows, replace=False)

    for index in random_indices:
        new_row = pd.DataFrame({
            'Startzeit': [np.nan],
            'Schritt': [np.nan],
            'Phase': [np.nan],
            'Person': ['Assistent'],
            'Text': ['*Ausfüllen*']
        })

        df = pd.concat([df.iloc[:index], new_row,
                       df.iloc[index:]]).reset_index(drop=True)

    df['Startzeit'] = df['Startzeit'].apply(pd.to_numeric).ffill().bfill()
    df['Phase'] = df['Phase'].ffill().bfill().astype(int)
    df['Schritt'] = df['Schritt'].ffill().bfill()

    # Adjust order of columns
    df = df[['Startzeit', 'Schritt', 'Phase', 'Person', 'Text']]

    return df
