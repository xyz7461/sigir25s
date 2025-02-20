import fire
import json
import pandas as pd
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import islice
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import random
import ast


def batch_iterator(iterator, batch_size):
    iterator = iter(iterator)
    while True:
        batch = tuple(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def read_system_prompt(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        return ""


def read_completed_ids(output_csv_path):
    if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
        df = pd.read_csv(output_csv_path)
        return set(df['id'])
    else:
        return set()


def parse_result(result):
    match = re.search(r'\{[\s\S]*\}', result)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None


def generate_prompt(row, system_prompt):
    sample_string = ''
    for key, value in row['samples'].items():
        sample_string += f'{key}: {value}'

    random_key = random.choice(list(row['samples'].keys()))

    user_prompt = f"""
                You are a movie recommender.
                Given the movies/rating pairs from the user below, predict the rating of the movie {row['target'][0]}:
                Return the Prediction in Json Format. The Json should consist of the following fields
                MovieName : The input movie name.
                RatingPrediction : The predicted rating. (Possible Values: 1,2,3,4,5)
                A correct output will look like:
                {{"MovieName" : "{random_key}",
                "RatingPrediction" : "{row['samples'][random_key]}"}}
                There should be no other text, analysis, or code of any kind.
                You will be penalized for every extra token before or after the rating prediction JSON
                <USER INPUT>
                Input:
                Movie Ratings: {sample_string}
                Movie to rate: {row['target'][0]}
                </USER INPUT>
                """
    return system_prompt + '\n' + user_prompt


class EndCurlyBraceStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.end_token_id = tokenizer.convert_tokens_to_ids("}")

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last token generated is the end curly brace
        return input_ids[0, -1] == self.end_token_id



def main(
        model_name='../final/llama/Meta-Llama-3.1-70B-Instruct-FP8',
        input_csv_path='orig_data/fin_tst_all.csv',
        temperature=.5,
        top_p=0.3,
        max_new_tokens=75,
        batch_size=10,
        language_in='en',
        language_out='en'
):
    # current_working_directory = os.getcwd()
    # print(current_working_directory)

    # paths = [('es-tr', 'jp'), ('jp-tr', 'es')]
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

    output_csv_path = f"good/llama70b_rp_all.csv"
    bad_csv_path = f"bad/Bad_llama70b_rp_all.csv"
    system_prompt = read_system_prompt('none')
    input_df = pd.read_csv(input_csv_path)
    completed_ids = read_completed_ids(output_csv_path)
    bad_ids = read_completed_ids(bad_csv_path)

    input_df['samples'] = input_df['samples'].apply(ast.literal_eval)
    input_df['target'] = input_df['target'].apply(ast.literal_eval)

    input_df = input_df[~input_df['id'].isin(completed_ids)]
    input_df = input_df[~input_df['id'].isin(bad_ids)]
    print('out: ', output_csv_path)
    print(f"Removed {len(completed_ids)} rows from input dataframe")
    print("Remaining rows: ", len(input_df))

    print('Going into the main loop now!')
    end_curly_brace_criteria = EndCurlyBraceStoppingCriteria(tokenizer)

    j = 0
    tot = len(input_df) // batch_size
    print(f"{tot}")
    for batch in batch_iterator(input_df.iterrows(), batch_size):
        j += 1
        if j % 5 == 0:
            print(f"{j} : {tot}")
        batch_prompts = []
        batch_metadata = []
        results = []

        for index, row in batch:
            full_prompt = generate_prompt(row, system_prompt)
            batch_prompts.append(full_prompt)
            this_metadata = {'id': row['id'], 'userId': row['userId'], 'prompt': row['target'],
                             'samples': row['samples']}
            batch_metadata.append(this_metadata)


        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate outputs in batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([end_curly_brace_criteria])
            )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i, generated_text in enumerate(generated_texts):
            prompt_length = len(batch_prompts[i])
            result = generated_text[prompt_length:]
            results.append(result)

        new_rows = []
        bad_rows = []
        for metadata, result in zip(batch_metadata, results):
            if result is not None:
                # print(result)
                parsed_json = parse_result(result)
                if parsed_json is not None:
                    metadata['json'] = parsed_json
                    new_rows.append(metadata)
                else:
                    # print("couldn't parse this: ", metadata[language_in])
                    metadata['result'] = result
                    bad_rows.append(metadata)

        if new_rows:
            print(f"Writing {len(new_rows)} rows to csv")
            print(f"New ids completed: {[row['id'] for row in new_rows]}")
            new_df = pd.DataFrame(new_rows)
            if os.path.exists(output_csv_path):
                new_df.to_csv(output_csv_path, mode='a', header=False, index=False)
            else:
                new_df.to_csv(output_csv_path, index=False)

        if bad_rows:
            bad_df = pd.DataFrame(bad_rows)
            if os.path.exists(bad_csv_path):
                bad_df.to_csv(bad_csv_path, mode='a', header=False, index=False)
            else:
                bad_df.to_csv(bad_csv_path, index=False)

        # if i > 10:
        #     break

if __name__ == "__main__":
    fire.Fire(main)

