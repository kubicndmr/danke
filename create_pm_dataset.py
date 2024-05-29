import os
import nltk
import random
import pandas as pd

from datasets import load_dataset


# Download the necessary NLTK data
nltk.download('punkt')

# Create target data folder
os.mkdir('Data')

# Parameters
n_split = [100000, 120000, 140000] # number of abstracts in train, valid, test sets
min_abs_len = 1 # min number of sentences in abstract
max_abs_len = 20 # max number of sentences in abstract

# Load the dataset
pubmed = load_dataset('pubmed', streaming=True, trust_remote_code=True)

# Initialize a list to store the entries
entries = []

# Function to shuffle sentences in a text
def shuffle_sentences(text):
    sentences = nltk.sent_tokenize(text)
    zipped_sentences = list(zip(sentences, range(len(sentences))))
    random.shuffle(zipped_sentences)
    
    shuffled_sentences, shuffled_indices = zip(*zipped_sentences)
    
    shuffled_sentences = list(shuffled_sentences)
    shuffled_indices = list(shuffled_indices)
    return shuffled_sentences, shuffled_indices

# Iterate over the entries in the dataset
size_counter = 0
for idx, entry in enumerate(pubmed['train']):
    print(f'{size_counter}/{idx}', end='\r')
    
    # Extract the desired fields
    pmid = entry['MedlineCitation']['PMID']
    abstract_text = entry['MedlineCitation']['Article']['Abstract']['AbstractText']
    
    if abstract_text != '':
        # Shuffle the sentences in the abstract text
        shuffled_sentences, shuffled_indices = shuffle_sentences(abstract_text)
        
        if (len(shuffled_sentences) > 1 and 
            len(shuffled_sentences) < 20): 
            # Append the entry to the list
            entries.append({
                'PMID': pmid,
                'AbstractText': abstract_text,
                'ShuffledSentences': shuffled_sentences,
                'ShuffledIndices': shuffled_indices
            })
            
            # Check if the number of entries reached the limit
            if size_counter == n_split[0]:
                # Create a DataFrame from the list of entries
                df = pd.DataFrame(entries)
                
                # Save the DataFrame
                df.to_pickle('Data/train.pkl')
                
                # Clear the entries list
                entries.clear()
                
            if size_counter == n_split[1]:
                # Create a DataFrame from the list of entries
                df = pd.DataFrame(entries)
                
                # Save the DataFrame
                df.to_pickle('Data/valid.pkl')
                
                # Clear the entries list
                entries.clear()
                
            if size_counter == n_split[2]:
                # Create a DataFrame from the list of entries
                df = pd.DataFrame(entries)
                
                # Save the DataFrame
                df.to_pickle('Data/test.pkl')
                
                # Clear the entries list
                entries.clear()
            
                break
        
            size_counter += 1    