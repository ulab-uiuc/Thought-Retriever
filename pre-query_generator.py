import os
import argparse
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

from utils import *


def docT5query(input_ids, tokenizer, model, max_length=128, top_k=10, tau=1.0, num_return_questions=2):
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        temperature=tau,
        num_return_sequences=num_return_questions)

    ret = []
    for i in range(len(outputs)):
        q = tokenizer.decode(outputs[i], skip_special_tokens=True)
        ret.append(q)
        print(f'Generated pre-question {i + 1}: {q}')

    return ret


def llm2query(prompt, api_base, api_key):
    content = get_llm_response_via_api(prompt=prompt,
                                       API_BASE=api_base,
                                       API_KEY=api_key,
                                       LLM_MODEL="Qwen/Qwen1.5-72B-Chat",
                                       TAU=0.5,
                                       SEED=42)
    content = content.split("\n")
    for ind, c in enumerate(content):
        for start_ind in range(len(c)):
            if str(c[start_ind]).isalpha():
                break
        content[ind] = c[start_ind:]

    return content


def generate_pre_query_via_docT5query_deprecated(dataset, device, max_length=128, top_k=10, tau=1.0,
                                                 num_return_questions=2, feat_cross=True):
    print("{} Generating Pre-queries...".format(show_time()))
    tokenizer = T5Tokenizer.from_pretrained('./t5-base-docT5query')
    config = T5Config.from_pretrained('./t5-base-docT5query')
    model = T5ForConditionalGeneration.from_pretrained(
        './t5-base-docT5query/model.ckpt-1004000', from_tf=True, config=config)
    # tokenizer = T5Tokenizer.from_pretrained('./t5-large-docT5query')
    # config = T5Config.from_pretrained('./t5-large-docT5query')
    # model = T5ForConditionalGeneration.from_pretrained(
    #     './t5-large-docT5query/model.ckpt-1004700', from_tf=True, config=config)
    model.to(device)

    doc_text_list = []
    for data in dataset:
        if "label" in data and data["label"] == "related work":
            doc_text = data["title"] + "\n" + data["abstract"]
            doc_text_list.append(doc_text)

    pre_queries = []
    if feat_cross:
        cross_type_list = [2, 3, 4]
        for _ in range(num_return_questions):
            cross_type = np.random.choice(cross_type_list)
            candidates = np.random.choice(doc_text_list, size=cross_type, replace=False)
            doc_text = ""
            for can in candidates:
                doc_text += can
            input_ids = tokenizer.encode(doc_text, return_tensors='pt').to(device)
            sub_pre_queries = docT5query(input_ids,
                                         tokenizer,
                                         model,
                                         max_length=max_length,
                                         top_k=top_k,
                                         tau=tau,
                                         num_return_questions=1)
            pre_queries.extend(sub_pre_queries)
    else:
        for doc_text in doc_text_list:
            input_ids = tokenizer.encode(doc_text, return_tensors='pt').to(device)
            sub_pre_queries = docT5query(input_ids,
                                         tokenizer,
                                         model,
                                         max_length=max_length,
                                         top_k=top_k,
                                         tau=tau,
                                         num_return_questions=num_return_questions)
            pre_queries.extend(sub_pre_queries)

    print("{} {} Pre-queries Generated".format(show_time(), len(pre_queries)))

    return pre_queries


def generate_pre_query_using_title_or_abs(dataset, device, dataset_type='related_multi', tool='llm', llm_paras=None,
                                          t5_paras=None, api_base=None, api_key=None):
    if dataset_type not in ['related_multi', 'abs_single', 'abs_multi']:
        raise Exception("Error")
    pre_queries = []
    if dataset_type == 'related_multi':
        title = dataset[0]["title"]
        abstract = dataset[0]["abstract"]
    elif dataset_type == 'abs_single':
        title = dataset["title"]
        abstract = dataset["abstract"]
    else:
        return pre_queries
    if tool == "llm":
        num_return_questions = llm_paras
        if dataset_type == "related_multi":
            prompt = "### Title:\n{text1}\n\n### Abstract:\n{text2}\n\n" \
                     "Please generate {text3} questions for the Title and Abstract provided above." \
                     "The generated questions should try to simulate the tone of human questions as much as possible, " \
                     "and the diversity of questions should be maintained and should not be limited to the same type of questions." \
                     "Most of the questions generated should revolve around the three words: What, How, and Why and start the question with one of these three words." \
                     "Please ensure that the generated questions are all interrogative sentences and are diverse." \
                     "Please directly output the generated questions, one line per question, do not output irrelevant text."
            input = {}
            input['text1'] = title
            input['text2'] = abstract
            input['text3'] = num_return_questions
        else:
            prompt = "### Title:\n{text1}\n\n" \
                     "Please generate {text2} questions for the Title provided above." \
                     "The generated questions should try to simulate the tone of human questions as much as possible, " \
                     "and the diversity of questions should be maintained and should not be limited to the same type of questions." \
                     "Most of the questions generated should revolve around the three words: What, How, and Why and start the question with one of these three words." \
                     "Please ensure that the generated questions are all interrogative sentences and are diverse." \
                     "Please directly output the generated questions, one line per question, do not output irrelevant text."
            input = {}
            input['text1'] = title
            input['text2'] = num_return_questions
        prompt = prompt.format_map(input)
        pre_queries.extend(llm2query(prompt, api_base, api_key))
    elif tool == "doct5query":
        tokenizer, model, max_length, top_k, tau, num_return_questions = t5_paras
        if dataset_type == "related_multi":
            doc_text = title + "\n" + abstract
        else:
            doc_text = title
        input_ids = tokenizer.encode(doc_text, return_tensors='pt').to(device)
        sub_pre_queries = docT5query(input_ids,
                                     tokenizer,
                                     model,
                                     max_length=max_length,
                                     top_k=top_k,
                                     tau=tau,
                                     num_return_questions=num_return_questions)
        pre_queries.extend(sub_pre_queries)
    else:
        raise Exception("Error")

    return pre_queries


def generate_pre_query_using_chunks(dataset, device, dataset_type='related_multi', chunk_size=500, feat_cross=True,
                                    tool='llm', llm_paras=None, t5_paras=None, api_base=None, api_key=None):
    if dataset_type not in ['related_multi', 'abs_single', 'abs_multi']:
        raise Exception("Error")
    pre_queries = []
    if dataset_type == "related_multi":
        title = dataset[0]["title"]
        abstract = dataset[0]["abstract"]
        main_content_chunks = []
        for c in dataset[1:]:
            main_content_chunks.append(c["title"] + "\n" + c["abstract"])
        main_content_sentences = " ".join(main_content_chunks).split(". ")
    elif dataset_type == "abs_single":
        title = dataset["title"]
        abstract = ""
        main_content = dataset["main_content"].replace("\n", " ")
        main_content_sentences = main_content.split(". ")
        main_content_chunks = main_content.split(" ")
        main_content_chunks = [" ".join(main_content_chunks[i * chunk_size: (i + 1) * chunk_size]) for i in
                               range(len(main_content_chunks) // chunk_size)]
    else:
        title = ""
        abstract = ""
        main_content_chunks = []
        for c in dataset[1:]:
            main_content_chunks.append(c["title"] + "\n" + c["abstract"])
        main_content_sentences = " ".join(main_content_chunks).split(". ")

    if tool == "llm":
        num_return_questions = llm_paras
        paragraph = np.random.choice(main_content_chunks)
        if feat_cross:
            prompt = "### Title:\n{text1}\n\n### Abstract:\n{text2}\n\n### Paragraph:\n{text3}\n\n" \
                     "Please generate {text4} questions for the Title, Abstract, and Paragraph provided above." \
                     "Please try your best to generate questions based on the relationship between Title, Abstract and Paragraph." \
                     "The generated questions should try to simulate the tone of human questions as much as possible, " \
                     "and the diversity of questions should be maintained and should not be limited to the same type of questions." \
                     "Most of the questions generated should revolve around the three words: What, How, and Why and start the question with one of these three words." \
                     "Please ensure that the generated questions are all interrogative sentences and are diverse." \
                     "Please directly output the generated questions, one line per question, do not output irrelevant text."
            input = {}
            input['text1'] = title
            input['text2'] = abstract
            input['text3'] = paragraph
            input['text4'] = num_return_questions
            prompt = prompt.format_map(input)
            pre_queries.extend(llm2query(prompt, api_base, api_key))
        else:
            prompt = "### Paragraph:\n{text1}\n\n" \
                     "Please generate {text2} questions for the Paragraph provided above." \
                     "The generated questions should try to simulate the tone of human questions as much as possible, " \
                     "and the diversity of questions should be maintained and should not be limited to the same type of questions." \
                     "Most of the questions generated should revolve around the three words: What, How, and Why and start the question with one of these three words." \
                     "Please ensure that the generated questions are all interrogative sentences and are diverse." \
                     "Please directly output the generated questions, one line per question, do not output irrelevant text."
            input = {}
            input['text1'] = paragraph
            input['text2'] = num_return_questions
            prompt = prompt.format_map(input)
            pre_queries.extend(llm2query(prompt, api_base, api_key))
    elif tool == "doct5query":
        tokenizer, model, max_length, top_k, tau, num_return_questions = t5_paras
        num_return_questions_1 = num_return_questions // 2
        num_return_questions_2 = num_return_questions - num_return_questions_1
        # Get Queries using Sentences
        cross_type_list = [4, 6, 8]
        for _ in range(num_return_questions_1):
            cross_type = np.random.choice(cross_type_list)
            candidates = np.random.choice(main_content_sentences, size=cross_type, replace=False)
            doc_text = ""
            for can in candidates:
                doc_text += can
            if feat_cross:
                doc_text = title + abstract + doc_text
            input_ids = tokenizer.encode(doc_text, return_tensors='pt').to(device)
            sub_pre_queries = docT5query(input_ids,
                                         tokenizer,
                                         model,
                                         max_length=max_length,
                                         top_k=top_k,
                                         tau=tau,
                                         num_return_questions=1)
            pre_queries.extend(sub_pre_queries)
        # Get Queries using A Single Chunk
        for _ in range(num_return_questions_2):
            doc_text = np.random.choice(main_content_chunks)
            if feat_cross:
                doc_text = title + abstract + doc_text
            input_ids = tokenizer.encode(doc_text, return_tensors='pt').to(device)
            sub_pre_queries = docT5query(input_ids,
                                         tokenizer,
                                         model,
                                         max_length=max_length,
                                         top_k=top_k,
                                         tau=tau,
                                         num_return_questions=1)
            pre_queries.extend(sub_pre_queries)
    else:
        raise Exception("Error")

    return pre_queries


def generate_pre_query_AcademicEval(dataset, device, dataset_type='related_multi', chunk_size=500, feat_cross=True,
                                    withDocT5query=False, t5_paras=(128, 10, 1.0, 5), llm_paras=(5), api_base=None,
                                    api_key=None):
    print("{} Generating Pre-queries...".format(show_time()))
    pre_queries = []
    pre_queries.extend(generate_pre_query_using_title_or_abs(dataset, device, dataset_type=dataset_type, tool='llm',
                                                             llm_paras=llm_paras, api_base=api_base, api_key=api_key))
    pre_queries.extend(generate_pre_query_using_chunks(dataset, device, dataset_type=dataset_type,
                                                       chunk_size=chunk_size, feat_cross=feat_cross, tool='llm',
                                                       llm_paras=llm_paras, api_base=api_base, api_key=api_key))
    if withDocT5query:
        tokenizer = T5Tokenizer.from_pretrained('./t5-base-docT5query')
        config = T5Config.from_pretrained('./t5-base-docT5query')
        model = T5ForConditionalGeneration.from_pretrained(
            './t5-base-docT5query/model.ckpt-1004000', from_tf=True, config=config)
        model.to(device)
        max_length, top_k, tau, num_return_questions = t5_paras
        t5_paras = (tokenizer, model, max_length, top_k, tau, num_return_questions)
        pre_queries.extend(generate_pre_query_using_title_or_abs(dataset, device, dataset_type=dataset_type,
                                                                 tool='doct5query', t5_paras=t5_paras))
        pre_queries.extend(generate_pre_query_using_chunks(dataset, device, dataset_type=dataset_type,
                                                           chunk_size=chunk_size, feat_cross=feat_cross,
                                                           tool='doct5query', t5_paras=t5_paras))

    print("{} {} Pre-queries Generated".format(show_time(), len(pre_queries)))

    return pre_queries


def generate_pre_query_others(dataset="wcep"):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate pre-queries for academic datasets')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory path for dataset (e.g., /data/AcademicEval/related_multi)')
    parser.add_argument('--api_base', type=str, default='https://api.together.xyz',
                        help='API base URL')
    parser.add_argument('--api_key', type=str, required=True,
                        help='API key for LLM service')
    parser.add_argument('--dataset_type', type=str, default='related_multi',
                        choices=['related_multi', 'abs_single', 'abs_multi'],
                        help='Type of dataset')
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='Chunk size for processing')
    parser.add_argument('--feat_cross', action='store_true', default=True,
                        help='Enable feature crossing')
    parser.add_argument('--with_doct5query', action='store_true', default=False,
                        help='Use DocT5Query model')
    parser.add_argument('--device', type=int, default=3,
                        help='GPU device number')

    args = parser.parse_args()

    ROOT = args.root

    for filename in os.listdir(ROOT):
        with open(os.path.join(ROOT, filename), 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        device = get_device(args.device)
        pre_queries = generate_pre_query_AcademicEval(dataset, device,
                                                      dataset_type=args.dataset_type,
                                                      chunk_size=args.chunk_size,
                                                      feat_cross=args.feat_cross,
                                                      withDocT5query=args.with_doct5query,
                                                      t5_paras=(128, 10, 1.0, 5),
                                                      llm_paras=(5),
                                                      api_base=args.api_base,
                                                      api_key=args.api_key)
        for q in pre_queries:
            print(q)
        # dataset["pre_questions"] = pre_queries
        # dataset[0]["pre_questions"] = pre_queries
        # write_to_json(dataset, os.path.join(ROOT, filename))
        break
