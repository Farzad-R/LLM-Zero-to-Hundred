import pandas as pd
import yaml
from dotenv import load_dotenv, find_dotenv
import os
import openai
from pyprojroot import here
from tqdm import tqdm
_ = load_dotenv(find_dotenv())
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")

with open(here("configs/config.yml")) as cfg:
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)

questions_df = pd.read_excel(os.path.join(
    here(cfg["eval_questions_dir"]), cfg["eval_file_name"]))

llm_system_role = """You will receive a question, along with its correct answer and the answer from langchain_token_mmr technique. Score the given answer from 0 to 1.
    Write down each answer in a separate line with no extra character.\n\n

    Example response:
    langchain_token_mmr: 0.7
    """
# llm_system_role = """You will receive a question, along with its correct answer. In the following
#     you will recieve the answer from other systems and their names. To each answer give an score from 0 to 1.
#     Write down each answer in a separate line with no extra character.\n\n

#     Example response:
#     langchain_recursive_similarity: 0.7
#     langchain_recursive_mmr: 0.5
#     """
# llm_system_role = """You will receive a question, along with its correct answer. In the following
#     you will recieve the answer from other systems and their names. To each answer give an score from 0 to 1.
#     Write down each answer in a separate line with no extra character.\n\n

#     Example response:
#     llama_index_sentence_retrieval: 0.7
#     llama_index_auto_merging_retrieval: 0.5
#     """

# tqdm(questions_df.iterrows(), total=len(questions_df)):
for idx, row in questions_df.iterrows():
    question = row["question"]
    correct_answer = row["correct_answer"]
    langchain_token_mmr_result = row["langchain_token_mmr_result"]
    langchain_recursive_similarity_result = row["langchain_recursive_similarity_result"]
    langchain_recursive_mmr_result = row["langchain_recursive_mmr_result"]
    llama_index_sentence_retrieval_result = row["llama_index_sentence_retrieval_result"]
    llama_index_auto_merging_retrieval_result = row["llama_index_auto_merging_retrieval_result"]

    # prompt = f"Question: {question}\n\nCorrect answer: {correct_answer},\
    #     \n\n\nlangchain_recursive_similarity: {langchain_recursive_similarity_result}\
    #     \n\n\nlangchain_recursive_mmr: {langchain_recursive_mmr_result}"

    prompt = f"Question: {question}\n\nCorrect answer: {correct_answer},\
        \n\n\nlangchain_token_mmr: {langchain_token_mmr_result}"

    # prompt = f"Question: {question}\n\nCorrect answer: {correct_answer},\
    #     \n\n\nllama_index_sentence_retrieval: {llama_index_sentence_retrieval_result}\
    #     \n\n\nllama_index_auto_merging_retrieval: {llama_index_auto_merging_retrieval_result}"

    messages = [
        {"role": "system", "content": llm_system_role},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        engine=cfg["llm_cfg"]["gpt_model"],
        messages=[
            {"role": "system", "content": llm_system_role},
            {"role": "user", "content": prompt}
        ],
        temperature=cfg["llm_cfg"]["temperature"]
    )
    result = response["choices"][0]["message"]["content"]
    print(f"\nresult:\n {result}\n")
    # Initialize variables to store the results
    lowest_score = float('inf')
    highest_score = float('-inf')
    lowest_keys = []
    highest_keys = []

    # Split the text into lines

    lines = result.split('\n')
    lines = [x for x in lines if x != '']

    print(f"\nlines:\n {lines}\n")
    for line in lines:
        result_dict = {}
        # Split the line into key and value parts
        key, value = line.split(': ')
        value = float(value)
        questions_df.at[idx, f"{key}_score"] = value
        # Check for lowest score
        if value < lowest_score:
            lowest_score = value
            lowest_keys = [key]
        elif value == lowest_score:
            lowest_keys.append(key)
        # Check for highest score
        if value > highest_score:
            highest_score = value
            highest_keys = [key]
        elif value == highest_score:
            highest_keys.append(key)

        result_dict[key] = value
    # After processing all lines, update the dataframe with the lowest and highest keys
    questions_df.at[idx, "lowest_score"] = ", ".join(lowest_keys)
    questions_df.at[idx, "highest_score"] = ", ".join(highest_keys)
    # except:
    #     print(f"index: {idx} had an issue.")
    #     pass
questions_df.to_excel(
    os.path.join(here(cfg["eval_questions_dir"]), cfg["eval_file_name"]), index=False)
print("Scoring is done and excel file was updated!")
