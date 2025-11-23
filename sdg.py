import os
os.environ['HF_HOME'] = "path"

import time
import torch
import argparse
import numpy as np
import pandas as pd
import huggingface_hub
import SynDataGen.synHelper as synHelper
import SynDataGen.synPrompts as synPrompts

from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText


####################################### Generation Functions #######################################
torch.set_float32_matmul_precision('high')
torch._dynamo.config.cache_size_limit = 32

def get_answer(processor, language_model, messages, max_new_tokens,
               do_sample=False, repetition_penalty=1.1):
    """
    Generates a response from the model based on the provided chat history.

    Parameters:
    language_model: The language model to generate the response.
    processor: The tokenizer to process the text and feature extractor.
    messages: A list of dictionaries containing the chat history.
    max_new_tokens: The maximum number of new tokens to generate.
    do_sample: Whether to sample the output (default: False).
    repettition_penalty: Penalize repettitionscle

    Returns:
    response: The generated response from the model.
    """
    # Encode the prompt to input tensor
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(language_model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]

    # Generate the model's response
    with torch.inference_mode():
        output = language_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            use_cache=True,
        )
        output = output[0][input_len:]

    # Decode the model's output and update the chat history
    response = processor.decode(output, skip_special_tokens=True)

    return response


def gen_data(processor, language_model, prompter):
    '''
    Generates synthetic data using a language model and a prompter.

    Parameters:
    processor: The tokenizer to process the text and feature extractor.
    language_model: The language model to generate the responses.
    prompter: An instance of SDGPrompts to generate prompts for the language model.

    Returns:
    result_df: A DataFrame containing the generated synthetic data.
    messages_log: A list of dictionaries containing the chat history with the language model.
    '''
    # Variables
    limit_try = 5
    tokens_per_row = 75
    template_tokens = 50
    summary_tokens = 360
    prompter.init_OR()

    # Get op draft
    df = synHelper.draft_OP()

    # Sliding Window
    dfs_to_concat = []
    df_splits = synHelper.df_splitter(df)
    steps_complete = np.zeros(len(df_splits))

    for i, answer_df in enumerate(df_splits):
        answer_df['Phase'] = answer_df['Phase'].map(
            synHelper.surgical_phases)

        person = answer_df['Person'].iloc[0]
        if (answer_df['Schritt'] != 'Alltäglich').any():
            step_label = answer_df.loc[answer_df['Schritt'] != 'Alltäglich', 'Schritt'].iloc[0]
        else:
            step_label = 'Alltäglich'
        step_count = df[(df['Schritt'] == step_label) &
                        (df['Person'] == person)].shape[0]

        if person == 'Radiologe':
            max_new_tokens = tokens_per_row*len(answer_df) + summary_tokens
        elif person == 'Assistent':
            max_new_tokens = tokens_per_row*len(answer_df) + template_tokens
        elif person == 'Patient':
            max_new_tokens = tokens_per_row*len(answer_df) + template_tokens
        else:
            print('The person has a problem')

        # Manage chat
        if len(dfs_to_concat) == 0:
            prompt = prompter.get_prompt(
                person, False, None, answer_df, step_label, step_count)

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            messages_log = messages.copy()

        else:
            step_df = pd.concat(dfs_to_concat)
            step_df['Text'] = step_df['Text'].str.strip()
            step_df['Text'] = step_df['Text'].str.strip('*')
            step_df['Text'] = step_df['Text'].str.strip('"""')

            prompt = prompter.get_prompt(
                person, True, step_df, answer_df, step_label, step_count)

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            messages_log.extend(messages)

        # Try limit_try times, if LM cant follow instructions
        n_try = 0
        while n_try < limit_try:
            #if True:
            try:
                print(
                    f'\tStep: {int(i+1)}/{len(df_splits)}\t|\tMax new tokens: {max_new_tokens}')

                # Generate Answer
                answer = get_answer(processor, language_model,
                                    messages, max_new_tokens=max_new_tokens)

                # Extract tagged block
                block_answer = synHelper.get_tagged_block(
                    answer, '<Antwort>', '</Antwort>')

                # Check line shapes/errors
                block_answer = synHelper.line_errors(
                    block_answer, len(answer_df))

                # Check format
                correct_format = synHelper.check_format(block_answer, [
                    'Index', 'Startzeit', 'Schritt', 'Phase', 'Person', 'Text'])
                if not correct_format:
                    raise ValueError(f"Columns or rows do not match")

                # Concat
                dfs_to_concat.append(synHelper.block_to_df(block_answer))

                # Add to chat
                messages_log.append(
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]})

                # Exit loop
                n_try = limit_try
                steps_complete[i] = 1

            #if False:
            except:
                n_try += 1
                max_new_tokens += 5
                print(f'\t\tConnot genreate Step {i+1}, will try again')

        if steps_complete[i] == 0:
            limit_try = -1

    result_df = pd.concat(dfs_to_concat)
    result_df = result_df[['Startzeit', 'Person', 'Text', 'Schritt', 'Phase']]
    result_df['Phase_Label'] = result_df['Phase'].map(
        synHelper.reversed_surgical_phases)
    result_df['Text'] = result_df['Text'].str.strip()
    result_df['Text'] = result_df['Text'].str.strip('*')
    result_df['Text'] = result_df['Text'].str.strip('"""')
    assert np.prod(steps_complete) == 1, "Not all steps completed"

    return result_df, messages_log


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

    # Parameters
    error_count = 0
    error_patience = 3
    prefix_idx = args.prefix_index
    end_idx = prefix_idx + args.num_target - 1
    model_id = 'google/gemma-3-27b-it' #'google/gemma-2-2b-it'

    # Login huggingface environment
    huggingface_hub.login("hf_token", add_to_git_credential=False)

    # Model
    auto_processor = AutoProcessor.from_pretrained(
        model_id
    )

    auto_language_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )

    # Prompts
    prompter = synPrompts.SDGPrompts()

    # Generate Data
    while prefix_idx <= end_idx and error_count < error_patience:
        #if True:
        try:
            # set save name
            save_name = os.path.join(
                args.target_path, synHelper.prefix(prefix_idx+1, 'SynOP_')+".csv")
            print(f"Generating: {save_name}")

            # generate data
            generation_start = time.time()
            df, log_container = gen_data(
                processor=auto_processor,
                language_model=auto_language_model,
                prompter=prompter
            )

            # save & update
            df.to_csv(save_name)

            # save log
            with open(save_name[:-4] + ".txt", "w") as f:
                for l in log_container:
                    f.write('\n'+'*'*50+' <'+l['role']+'> '+'*'*50+'\n')
                    f.write(l['content'][0]['text'])
                f.write(
                    f'\nElapsed time:\t{time.time() - generation_start}(s)')

            # increment
            prefix_idx += 1
            error_count = 0

        #if False:
        except:
            print(f"\t{save_name} could not generated. Trying again!")
            error_count += 1
