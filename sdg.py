import os
import sys
import time
import torch
import argparse
import sdg_helper
import sdg_prompts
import numpy as np
import pandas as pd
import huggingface_hub

from dotenv import load_dotenv
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


DEBUG_MODE = False
if DEBUG_MODE:
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

####################################### Generation Functions #######################################


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
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

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


def gen_data(tokenizer, language_model):
    '''
    Generates synthetic data
    '''
    # Variables
    limit_try = 3
    tokens_per_row = 75
    summary_tokens = 250

    # Get op draft
    df = sdg_helper.draft_OP()

    # Sliding Window
    dfs_to_concat = []
    df_splits = sdg_helper.df_splitter(df)
    steps_complete = np.zeros(len(df_splits))

    for i, sub_df in enumerate(df_splits):
        max_new_tokens = tokens_per_row*len(sub_df) + summary_tokens
        sub_df['Phase'] = sub_df['Phase'].map(
            sdg_helper.surgical_phases)

        # Manage chat
        if i == 0:
            # Add answer template
            prompt = sdg_prompts.initial_prompt
            prompt += f"\n<Antwort>\n{sub_df.to_csv(index=True, sep=';', index_label='Index')}</Antwort>\n"

            messages = [{"role": "system", "content": sdg_prompts.system_role},
                        {"role": "user", "content": prompt}]
        else:
            prompt = sdg_prompts.iteration_prompt
            prompt += f"\n<Antwort>\n{sub_df.to_csv(index=True, sep=';', index_label='Index')}</Antwort>\n"
            
            messages.append(
                    {"role": "user", "content": prompt})

        # Try limit_try times, if LM cant follow instructions
        n_try = 0
        while n_try < limit_try:
            try:
                print(
                    f'\tStep 1: {int(i+1)}/{len(df_splits)} max new tokens: {max_new_tokens}')

                # Generate Answer
                answer = get_answer(tokenizer, language_model,
                                    messages, max_new_tokens=max_new_tokens)

                # Extract tagged block
                block_answer = sdg_helper.get_tagged_block(
                    answer, '<Antwort>', '</Antwort>')

                # Check line shapes/errors
                block_answer = sdg_helper.line_errors(
                    block_answer, len(sub_df))

                # Check format
                correct_format = sdg_helper.check_format(block_answer, [
                    'Index', 'Startzeit', 'Schritt', 'Phase', 'Text'])
                if not correct_format:
                    raise ValueError(f"Columns or rows do not match")
                
                # Concat
                dfs_to_concat.append(sdg_helper.block_to_df(block_answer))
                
                # Add to chat
                messages.append(
                    {"role": "assistant", "content": '\n'.join(block_answer) + '\n'})

                # Exit loop
                n_try = limit_try
                steps_complete[i] = 1
                
                with open(f'step_1_{i+1}.txt', 'w') as f:
                        for m in messages:
                            f.write('\n'+'*'*50+' <'+m['role']+'> '+'*'*50+'\n')
                            f.write(m['content'])

            except:
                n_try += 1
                print(f'\t\tConnot genreate Step {i+1}, will try again')

    result_df = pd.concat(dfs_to_concat)
    result_df = result_df[['Index', 'Startzeit', 'Text', 'Step_Label', 'Phase_Label']]
    result_df['Text'] = result_df['Text'].str.strip()
    result_df['Text'] = result_df['Text'].str.strip('*')
    result_df['Text'] = result_df['Text'].str.strip('"""')
    assert np.prod(steps_complete) == 1, "Not all steps completed"

    return result_df, messages


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
    error_patience = 1
    prefix_idx = args.prefix_index
    end_idx = prefix_idx + args.num_target - 1
    model_id = 'mistralai/Mistral-Large-Instruct-2407'

    # Login huggingface environment
    load_dotenv()
    huggingface_hub.login(os.getenv('HF_TOKEN'), add_to_git_credential=False)

    # Model
    auto_tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=os.getenv('HF_CACHE_DIR'))
    auto_language_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        cache_dir=os.getenv('HF_CACHE_DIR')
    )
    '''
    auto_tokenizer = None
    auto_language_model = None
    '''
    # Generate Data
    while prefix_idx <= end_idx and error_count < error_patience:
        try:
            # set save name
            save_name = os.path.join(
                args.target_path, sdg_helper.prefix(prefix_idx+1, 'SynOP_')+".csv")
            print(f"Generating: {save_name}")

            # generate data
            generation_start = time.time()
            df, log_container = gen_data(
                tokenizer=auto_tokenizer,
                language_model=auto_language_model
            )

            # save & update
            df.to_csv(save_name)

            # save log
            with open(save_name[:-4] + ".txt", "w") as f:
                for l in log_container:
                    f.write('\n'+'*'*50+' <'+l['role']+'> '+'*'*50+'\n')
                    f.write(l['content'])
                f.write(
                    f'\nElapsed time:\t{time.time() - generation_start}(s)')

            # increment
            prefix_idx += 1
            error_count = 0

        # uppps
        except:
            print(f"\t{save_name} could not generated. Trying again!")
            error_count += 1
