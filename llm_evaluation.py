import argparse

from utils import *


def llm_eval(generate_response_1, generate_response_2, ground_truth):
    prompt = "### Generated Response 1:\n{gen1}\n\n### Generated Response 2:\n{gen2}\n\n### Reference Response:\n{ref}\n\n" \
             "We would like to request your feedback on the generation quality of the Generated Response 1 and 2 in response to the user instruction and input displayed above.\n" \
             "Please refer to the Reference Response to rate the semantic similarity, relevance and information completeness of the Generated Response 1 and 2. Please output an overall score on a scale of 1 to 100, where a higher score indicates better " \
             "overall performance.\nPlease first output a single line containing only two values indicating the scores for the Generated Response 1 and 2, respectively." \
             "\nIn the subsequent line, please provide a comprehensive explanation of your evaluation," \
             "avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."

    input = {}
    input['gen1'] = generate_response_1
    input['gen2'] = generate_response_2
    input['ref'] = ground_truth
    prompt = prompt.format_map(input)
    content = get_llm_response_via_api(prompt=prompt,
                                       API_BASE=API_BASE,
                                       API_KEY=API_KEY,
                                       LLM_MODEL=LLM_MODEL,
                                       TAU=TAU,
                                       SEED=SEED)
    print(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_base", type=str, default='https://api.together.xyz')
    parser.add_argument("--api_key", type=str, default="[YOUR API KEY]",
                        choices=["your api_key"])
    parser.add_argument("--llm_model", type=str, default='garage-bAInd/Platypus2-70B-instruct',
                        choices=["gpt-3.5-turbo",
                                 "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                 "Open-Orca/Mistral-7B-OpenOrca",
                                 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
                                 'garage-bAInd/Platypus2-70B-instruct'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0)
    opt = parser.parse_args()

    API_BASE = opt.api_base
    API_KEY = opt.api_key
    LLM_MODEL = opt.llm_model
    SEED = opt.seed
    TAU = opt.tau

    set_seed(int(SEED))

    answer_path = "answers.json"
    with open(answer_path, 'r', encoding='utf-8') as file:
        answer_history = json.load(file)

    ROOT = './ACA'
    # ROOT = './WCEP'
    for data_file in os.listdir(ROOT):
        if not data_file.endswith('.json'):
            continue
        data_path = os.path.join(ROOT, data_file)
        with open(data_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)

        gt = dataset[0]["gt"]
        if "title" in dataset[0]:
            sample_answer = answer_history[dataset[0]['title']]
        else:
            sample_answer = answer_history[dataset[0]['id']]
        # llm_eval(sample_answer['main'], sample_answer['dense_contriever'], sample_answer['gt'])
        llm_eval(sample_answer['dense_contriever'], sample_answer['dense_contrieverreduce'], sample_answer['gt'])
        # llm_eval(sample_answer['main'], sample_answer['main_ip'], sample_answer['gt'])
        # llm_eval(sample_answer['main'], sample_answer['main_t5'], sample_answer['gt'])
        # llm_eval(sample_answer['main'], sample_answer['sparse_bm25'], sample_answer['gt'])
        # llm_eval(sample_answer['dense_contriever'], sample_answer['sparse_bm25'], sample_answer['gt'])
        # llm_eval(sample_answer['sparse_bm25'], sample_answer['main'], sample_answer['gt'])
