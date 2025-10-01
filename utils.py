import os
import json
import time
import datetime
import random
import numpy as np

import torch
import openai


def show_time():
    time_stamp = '\033[1;31;40m[' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ']\033[0m'

    return time_stamp


def text_wrap(text):
    return '\033[1;31;40m' + str(text) + '\033[0m'


def get_device(index=0):
    return torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu")


def print_metrics(metrics):
    for k, v in metrics.items():
        ff = "{} " + k + " ("
        metric = metrics[k]
        for sub_k in metric.keys():
            ff += sub_k + "/"
        ff = ff[:-1] + "): "
        for sub_v in metric.values():
            ff += format(sub_v, ".4f") + "/"
        ff = ff[:-1]
        print(ff.format(show_time()))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def write_to_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)


def update_answers_json(title, answer, gt, method):
    # Check if answers.json exists
    if os.path.exists('answers.json'):
        with open('answers.json', 'r') as file:
            answers_dict = json.load(file)
    else:
        answers_dict = {}

    if title not in answers_dict:
        answers_dict[title] = {}
    answers_dict[title][method] = answer
    answers_dict[title]['gt'] = gt
    with open('answers.json', 'w') as file:
        json.dump(answers_dict, file, indent=4)


def save_ground_truth_to_json(group, ground_truth):
    # Load existing data if file exists
    try:
        with open('ground_truth.json', 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    # Update the data with new ground truth
    data[group] = ground_truth
    # Write back to file
    with open('ground_truth.json', 'w') as file:
        json.dump(data, file, indent=4)


def get_llm_response_via_api(prompt,
                             API_BASE="https://api.together.xyz",
                             API_KEY="[YOUR API KEY]",
                             LLM_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1",
                             TAU=1.0,
                             TOP_P=1.0,
                             N=1,
                             SEED=42,
                             MAX_TRIALS=2,
                             TIME_GAP=2):
    '''
    res = get_llm_response_via_api(prompt='hello')  # Default: TAU Sampling (TAU=1.0)
    res = get_llm_response_via_api(prompt='hello', TAU=0)  # Greedy Decoding
    res = get_llm_response_via_api(prompt='hello', TAU=0.5, N=2, SEED=None)  # Return Multiple Responses w/ TAU Sampling
    '''
    openai.api_base = API_BASE
    openai.api_key = API_KEY
    completion = None
    while MAX_TRIALS:
        MAX_TRIALS -= 1
        try:
            completion = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                n=N,
                temperature=TAU,
                top_p=TOP_P,
                seed=SEED,
            )
            break
        except Exception as e:
            print(e)
            print("Retrying...")
            time.sleep(TIME_GAP)

    if completion is None:
        raise Exception("Reach MAX_TRIALS={}".format(MAX_TRIALS))
    contents = completion.choices
    if len(contents) == 1:
        return contents[0].message["content"]
    else:
        return [c.message["content"] for c in contents]



if __name__ == '__main__':
    res = get_llm_response_via_api(prompt='hello', TAU=0, TOP_P=0)
    print(res)


