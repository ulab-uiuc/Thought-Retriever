import copy
import argparse

from tqdm import tqdm
from bert_score import score
from rouge_score import rouge_scorer

import thought_saver
from retrieval import *
from utils import *


def bert_score_eval(generate_response, ground_truth):
    P, R, F = score([generate_response], [ground_truth], lang="en")
    P = P.numpy()[0]
    R = R.numpy()[0]
    F = F.numpy()[0]

    return P, R, F


def rouge_eval(generate_response, ground_truth, type='rougeL'):
    scorer = rouge_scorer.RougeScorer([type], use_stemmer=True)
    scores = scorer.score(prediction=generate_response, target=ground_truth)
    P = scores[type].precision
    R = scores[type].recall
    F = scores[type].fmeasure

    return P, R, F


def rag_eval(id, paper_abs, context, ground_truth, method, prompt_choice=0):
    answer = rag(paper_abs, context, prompt_choice)
    # print(answer)
    # print(ground_truth)
    update_answers_json(id, answer, gt, method)
    metrics = dict()
    bert_P, bert_R, bert_F = bert_score_eval(answer, ground_truth)
    rouge_L_P, rouge_L_R, rouge_L_F = rouge_eval(answer, ground_truth, type='rougeL')
    rouge_1_P, rouge_1_R, rouge_1_F = rouge_eval(answer, ground_truth, type='rouge1')
    rouge_2_P, rouge_2_R, rouge_2_F = rouge_eval(answer, ground_truth, type='rouge2')
    metrics["BERT SCORE"] = {"P": bert_P, "R": bert_R, "F": bert_F}
    metrics["ROUGE-L"] = {"P": rouge_L_P, "R": rouge_L_R, "F": rouge_L_F}
    metrics["ROUGE-1"] = {"P": rouge_1_P, "R": rouge_1_R, "F": rouge_1_F}
    metrics["ROUGE-2"] = {"P": rouge_2_P, "R": rouge_2_R, "F": rouge_2_F}
    print_metrics(metrics)

    return metrics


def eval_flat_chunk(num_gt, index_last):
    neib_text_all_label = np.arange(num_gt)
    set_ground = set(neib_text_all_label)
    set_retrieved = set(index_last)
    intersection = set_ground.intersection(set_retrieved)
    precision = len(intersection) / len(set_retrieved) if len(set_retrieved) > 0 else 0
    recall = len(intersection) / len(set_ground) if len(set_retrieved) > 0 else 0
    print("{} Retrieval Precision: {:.4f}".format(show_time(), precision))
    print("{} Retrieval Recall: {:.4f}".format(show_time(), recall))


def get_response_via_llm(question, context, prompt_qa):
    content_l = []
    for inter1 in range(len(question)):
        question_i = question[inter1]
        context_i = context[inter1]
        input = {}
        input['question'] = question_i
        input['context'] = context_i
        prompt = prompt_qa.format_map(input)
        content = get_llm_response_via_api(prompt=prompt,
                                           API_BASE=API_BASE,
                                           API_KEY=API_KEY,
                                           LLM_MODEL=LLM_MODEL,
                                           TAU=TAU,
                                           SEED=SEED)
        content_l.append(content)

    return content_l


def qa_via_LLM(question, context):
    prompt_qa = "Given text: {context}, based on this text, answer the question: {question}"

    return get_response_via_llm(question, context, prompt_qa)


def summary_via_llm(question, context, verify=False):
    if verify:
        prompt_qa = (
            "Input: Given question:{question}, given answer:{context}. Based on the provided question and its corresponding answer, perform the following steps:"
            "Step 1: Determine if the answer is an actual answer or if it merely indicates that the question cannot be answered due to insufficient information. If the latter is true, just output 'idk' without any extra words "
            "Step 2: If it is a valid answer, succinctly summarize both the question and answer into a coherent knowledge point, forming a fluent passage."
        )
    else:
        prompt_qa = (
            "Given question:{question},given answer:{context},based on the given question and corresponding answer, "
            "summarize them into a knowledge point like a fluent passage.")

    return get_response_via_llm(question, context, prompt_qa)


def chunk_summary_via_llm(chunk, prompt_choice=0):
    prompt_qa = [(
        "Please summarize the content described in the following paragraph as concisely as possible, "
        "and ensure that the length of the summary text is significantly shorter than the original text."
        "Please try your best to highlight the key points or key information of the following paragraph in the summary, "
        "and while ensuring the completeness and accuracy of the summary relative to the original text, "
        "you can appropriately ignore some redundant information."
        "### Paragraph:{c}\n\n"),
        (
            "Please summarize the contents described in the following several paragraphs as concisely as possible, "
            "and ensure that the length of the summary text is significantly shorter than the original text."
            "Please try your best to highlight the key points or key information of the following paragraphs in the summary, "
            "and while ensuring the completeness and accuracy of the summary relative to the original text, "
            "you can appropriately ignore some redundant information."
            "### Paragraph:{c}\n\n")
    ]
    prompt_qa = prompt_qa[prompt_choice]
    input = {}
    if prompt_choice == 0:
        input['c'] = chunk
    else:
        input['c'] = "\n".join(chunk)

    prompt = prompt_qa.format_map(input)
    content = get_llm_response_via_api(prompt=prompt,
                                       API_BASE=API_BASE,
                                       API_KEY=API_KEY,
                                       LLM_MODEL=LLM_MODEL,
                                       TAU=TAU,
                                       SEED=SEED)

    return content


def rag(abstract, context, prompt_choice=0):
    prompt_qa = [(
        "Given the abstract and related work of a research article, along with a sample material, "
        "write a paragraph about its related work. Use the following as guidance:\n\n"
        "### Abstract: This research paper investigates the impact of climate change on global agricultural productivity. "
        "The study employs a comprehensive dataset of temperature and precipitation changes over the past century, "
        "combined with historical crop yield data. Through advanced statistical modeling and machine learning techniques, "
        "the research identifies significant correlations between temperature and precipitation fluctuations and variations in crop yields. "
        "Furthermore, it predicts future scenarios of agricultural productivity under different climate change scenarios, "
        "providing valuable insights for policymakers and stakeholders in the agricultural sector to develop adaptive strategies.\n\n"
        "### Related Work: Previous studies in the field have explored the relationship between climate change and agriculture but have primarily "
        "focused on specific regions or crops. Smith et al. (2017) conducted a comprehensive analysis of the impact of temperature on wheat yields "
        "in North America, highlighting the vulnerability of wheat crops to warming temperatures. Additionally, Johnson et al. (2019) investigated "
        "the effects of changing precipitation patterns on rice production in Southeast Asia, emphasizing the importance of water management in mitigating "
        "climate-related risks to agriculture. While these studies contribute valuable insights, our research extends their scope by considering a global "
        "perspective and employing advanced modeling techniques to provide more accurate predictions of future agricultural productivity under climate change scenarios.\n\n"

        "Based on the abstract of this article and related materials, write a paragraph about its related work:\n"
        "### Abstract: {abstract}\n\n ### Related Materials: {context}\n\n"
    ),
        (
            "Given the category and related materials of an event, "
            "write a short sentence summarizing this event.\n\n"
            "### Category: {abstract}\n\n ### Related Materials: {context}\n\n"
        )
    ]
    prompt_qa = prompt_qa[prompt_choice]
    input = {}
    input['abstract'] = abstract
    input['context'] = context
    prompt = prompt_qa.format_map(input)
    content = get_llm_response_via_api(prompt=prompt,
                                       API_BASE=API_BASE,
                                       API_KEY=API_KEY,
                                       LLM_MODEL=LLM_MODEL,
                                       TAU=TAU,
                                       SEED=SEED)
    # print(content)

    return content


def chunk_construction(dataset, thoughts=None, thoughts_limit=20):
    text_chunk_l = []
    if "title" in dataset[0]:
        paper = dataset[0]
        paper_abs = " "
        paper_abs += paper['title']
        paper_abs += paper['abstract']
        print("{} Evaluate Paper: {}\n".format(show_time(), paper['title']))
        num_gt = 0
        for i in dataset[1:]:
            abstract_title = " "
            abstract_title += i['title']
            abstract_title += i['abstract']
            text_chunk_l.append(abstract_title)
            if "label" in i and i["label"] == "related work":
                num_gt += 1
    else:
        meta_data = dataset[0]
        id = meta_data['id']
        category = meta_data['category']
        print("{} Evaluate Document: {}\n".format(show_time(), id))
        for i in dataset[1:]:
            text_chunk_l.append(i['article'])

    print("{} Num of Raw Chunks: {}".format(show_time(), len(text_chunk_l)))
    num_raw_chunks = len(text_chunk_l)

    identifier = paper['title'] if "title" in dataset[0] else id
    if thoughts is not None:
        thoughts_num = 0
        for item in thoughts[identifier]:
            if thoughts_num < thoughts_limit:
                text_chunk_l.append(item[0])
                thoughts_num = thoughts_num + 1

    if "title" in dataset[0]:
        return paper, paper_abs, text_chunk_l, num_gt, num_raw_chunks
    else:
        return id, category, text_chunk_l, num_raw_chunks


def thought_generation(pre_query, ch_text_chunk, ch_text_chunk_embed, id, retriever, query_tokenizer, ctx_tokenizer,
                              query_encoder, ctx_encoder):
    print("{} Thought Generation...".format(show_time()))
    num_chunks = len(ch_text_chunk)
    thoughts_container = thought_saver.load_thoughts()
    thoughts_container[id] = []
    pre_query_embedding = get_dense_embedding(pre_query, retriever=retriever, tokenizer=query_tokenizer,
                                              model=query_encoder)
    new_knowledge_sum = []
    new_knowledge_sum_embed = []

    for inter2 in tqdm(range(len(pre_query_embedding)), desc="Processing: "):
        print("{} Pre-question: {}".format(show_time(), pre_query[inter2]))
        neib_pre_node_idx = dense_neiborhood_search(ch_text_chunk_embed, [pre_query_embedding[inter2]])
        print("{} Retrieved Chunks: {}".format(show_time(), neib_pre_node_idx))
        retrieve_text = ''
        for inter3 in neib_pre_node_idx:
            retrieve_text += ch_text_chunk[inter3]
        answer_pre_idx = qa_via_LLM([pre_query[inter2]], [retrieve_text])
        new_knowledge_idx = summary_via_llm([pre_query[inter2]], answer_pre_idx, verify=False)
        new_knowledge_idx_test = summary_via_llm([pre_query[inter2]], answer_pre_idx, verify=True)
        if ('idk' not in new_knowledge_idx_test[0]) and ('Step 1' not in new_knowledge_idx_test[0]) and ('Step 1:' not in new_knowledge_idx_test[0]):
            print("{} Add A Thought...".format(show_time()))
            new_knowledge_embed_idx = get_dense_embedding(new_knowledge_idx, retriever=retriever,
                                                          tokenizer=ctx_tokenizer, model=ctx_encoder)
            new_knowledge_sum += new_knowledge_idx
            new_knowledge_sum_embed += new_knowledge_embed_idx
            ch_text_chunk += new_knowledge_idx
            ch_text_chunk_embed += new_knowledge_embed_idx
            thoughts_container = thought_saver.add_or_update_thought(id, thoughts_container, new_knowledge_idx)

        print("{} Number of Chunks & Thoughts: {} & {}".format(show_time(), num_chunks, len(ch_text_chunk) - num_chunks))

    thought_saver.save_thoughts(thoughts_container)

    print("{} Thought Generation Finished".format(show_time()))

    return ch_text_chunk, ch_text_chunk_embed


def run_sparse_retrieval(query, ch_text_chunk_embed, text_chunk_l):
    print("{} Sparse Retrieval...".format(show_time()))
    neib_ini = sparse_neiborhood_search(ch_text_chunk_embed, query, text_chunk_l)
    neib = np.array(neib_ini)
    retrieve_text = ''
    for inter3 in neib:
        retrieve_text += text_chunk_l[inter3]

    print("{} Retrieved Chunks:".format(show_time()), neib)

    return list(neib), retrieve_text


def run_dense_retrieval(query_embedding, ch_text_chunk_embed, ch_text_chunk, retriever, query_tokenizer, ctx_tokenizer,
                        query_encoder, ctx_encoder, chunk_num=8, recall_coe=5, sim_thre=0.85):
    print("{} Dense Retrieval...[SIM_THRE={}] [RECALL_COE={}]".format(show_time(), sim_thre, recall_coe))
    neib_ini = dense_neiborhood_search(ch_text_chunk_embed, query_embedding, num=chunk_num * recall_coe)
    neib_ini = list(neib_ini)
    context_last = []
    index_last = []
    context_last_embed = []
    index_last.append(neib_ini[0])
    context_last.append(ch_text_chunk[neib_ini[0]])
    context_last_embed += get_dense_embedding(context_last, retriever=retriever, tokenizer=ctx_tokenizer,
                                              model=ctx_encoder)
    for inter1 in range(1, len(neib_ini)):
        add_signal = True
        if len(index_last) < chunk_num:
            retrieve_index = neib_ini[inter1]
            retrieve_text = ch_text_chunk[retrieve_index]
            text_embed = get_dense_embedding([retrieve_text], retriever=retriever, tokenizer=ctx_tokenizer,
                                             model=ctx_encoder)
            similarity_list = calculate_similarity(context_last_embed, text_embed[0])
            for value in similarity_list:
                if value > sim_thre:
                    add_signal = False
                    break
            if add_signal:
                index_last.append(retrieve_index)
                context_last.append(retrieve_text)
                context_last_embed += text_embed

    print("{} Retrieved Chunks:".format(show_time()), index_last)
    retrieve_text = ''
    for inter3 in context_last:
        retrieve_text += inter3

    return index_last, retrieve_text


def main(dataset, pre_query, groundtruth, thoughts=None, thoughts_limit=15, chunk_num=8, recall_coe=5, sim_thre=0.85, retriever='contriever', exp='main'):
    query_tokenizer, ctx_tokenizer, query_encoder, ctx_encoder = get_dense_retriever(retriever=retriever)
    query_encoder = query_encoder.to(DEVICE)
    ctx_encoder = ctx_encoder.to(DEVICE)
    if "title" in dataset[0]:
        paper, paper_abs, text_chunk_l, num_gt, num_raw_chunks = chunk_construction(dataset, thoughts, thoughts_limit=thoughts_limit)
    else:
        id, category, text_chunk_l, num_raw_chunks = chunk_construction(dataset, thoughts, thoughts_limit=thoughts_limit)
    chunks_embedding_text_all = get_dense_embedding(text_chunk_l, retriever=retriever, tokenizer=ctx_tokenizer,
                                                    model=ctx_encoder)
    ch_text_chunk = copy.copy(text_chunk_l)
    ch_text_chunk_embed = copy.copy(chunks_embedding_text_all)
    if "title" in dataset[0]:
        query = ["Could you please write a related work for introducing this paper? Its abstract is:" + paper_abs]
    else:
        query = ["Could you please write a summary for introducing an event according to the articles provided? The topic of the event is:" + category]
    query_embedding = get_dense_embedding(query, retriever=retriever, tokenizer=query_tokenizer, model=query_encoder)
    if thoughts is None:
        ch_text_chunk, ch_text_chunk_embed = thought_generation(pre_query, ch_text_chunk,
                                                                ch_text_chunk_embed,
                                                                paper['title'] if "title" in dataset[0] else id,
                                                                retriever=retriever,
                                                                query_tokenizer=query_tokenizer,
                                                                ctx_tokenizer=ctx_tokenizer,
                                                                query_encoder=query_encoder,
                                                                ctx_encoder=ctx_encoder)
    print("{} Num of Chunks & Thoughts: {}".format(show_time(), len(ch_text_chunk)))
    index_last, context = run_dense_retrieval(query_embedding, ch_text_chunk_embed, ch_text_chunk, retriever=retriever,
                                              query_tokenizer=query_tokenizer, ctx_tokenizer=ctx_tokenizer,
                                              query_encoder=query_encoder, ctx_encoder=ctx_encoder, chunk_num=chunk_num,
                                              recall_coe=recall_coe, sim_thre=sim_thre)
    thought_utilization = len(set(range(num_raw_chunks, len(ch_text_chunk))).intersection(set(index_last))) / chunk_num
    print("{} Utilization of Thoughts: {}%".format(show_time(), thought_utilization * 100))
    if "title" in dataset[0]:
        metrics = rag_eval(paper['title'], paper_abs, context, groundtruth, method=exp, prompt_choice=0)
    else:
        metrics = rag_eval(id, category, context, groundtruth, method=exp, prompt_choice=1)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_base", type=str, default='https://api.together.xyz')
    parser.add_argument("--api_key", type=str, default="[YOUR API KEY]")
    parser.add_argument("--llm_model", type=str, default='mistralai/Mixtral-8x7B-Instruct-v0.1')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--thoughts_limit", type=int, default=999)
    parser.add_argument("--chunk_num", type=int, default=8)
    parser.add_argument("--recall_coe", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0)
    parser.add_argument("--sim_thre", type=float, default=0.85)
    parser.add_argument("--root", type=str, default='./data_sample',
                        help='Root directory containing data files')
    opt = parser.parse_args()

    API_BASE = opt.api_base
    API_KEY = opt.api_key
    LLM_MODEL = opt.llm_model
    SEED = opt.seed
    TAU = opt.tau
    THOUGHTS_LIMIT = opt.thoughts_limit
    CHUNK_NUM = opt.chunk_num
    SIM_THRE = opt.sim_thre
    RECALL_COE = opt.recall_coe

    set_seed(int(SEED))
    DEVICE = get_device(int(opt.cuda))

    # thought_path = "thoughts_abs_tradeoff.json"
    # with open(thought_path, 'r', encoding='utf-8') as file:
    #     thoughts = json.load(file)

    ROOT = opt.root
    metrics_list = []
    for data_file in os.listdir(ROOT):
        if not data_file.endswith('.json'):
            continue
        data_path = os.path.join(ROOT, data_file)
        with open(data_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)

        questions = dataset[0]["pre_questions"]
        gt = dataset[0]["gt"]

        metrics = main(dataset=dataset,
                       pre_query=questions,
                       groundtruth=gt,
                       thoughts=None,
                       # thoughts=thoughts,
                       thoughts_limit=THOUGHTS_LIMIT,
                       chunk_num=CHUNK_NUM,
                       recall_coe=RECALL_COE,
                       sim_thre=SIM_THRE,
                       retriever='contriever',
                       exp="main")

        metrics_list.append(metrics)
        break

    all_metrics = dict()
    for key in metrics_list[0].keys():
        all_metrics[key] = {kk: np.mean([vv[key][kk] for vv in metrics_list]) for kk in metrics_list[0][key].keys()}

    print("\n")
    print(text_wrap("=" * 50 + "Final Evaluation" + "=" * 50))
    print_metrics(all_metrics)
