import os
import yaml
import time
from pyprojroot import here
import pandas as pd
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import (load_index_from_storage,
                         set_global_service_context,
                         ServiceContext,
                         StorageContext,
                         )
from llama_utils import get_sentence_window_query_engine, get_automerging_query_engine

from dotenv import load_dotenv, find_dotenv
start_time = time.time()
_ = load_dotenv(find_dotenv())

with open(here("configs/config.yml")) as cfg:
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)

llm = AzureOpenAI(
    model=cfg["llm_cfg"]["gpt_model"],
    engine=cfg["llm_cfg"]["gpt_model"],
    deployment_name=os.getenv("gpt_deployment_name"),
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)
embed_model = AzureOpenAIEmbedding(
    model=cfg["llm_cfg"]["embed_model_name"],
    deployment_name=os.getenv("embed_deployment_name"),
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

set_global_service_context(service_context)

llama_eval_method = cfg["llama_index_cfg"]["llama_eval_method"]

print(
    f"Questions will be processed using llama_index {llama_eval_method} method.")
print("--------------------------------------\n\n")

if llama_eval_method == "sentence_retrieval":

    # Rebuild storage context
    storage_context = StorageContext.from_defaults(
        persist_dir=here(cfg["llama_index_cfg"]["sentence_retrieval"]["index_save_dir"]))
    # Load index from the storage context
    sentence_index = load_index_from_storage(storage_context)
    engine = get_sentence_window_query_engine(sentence_index=sentence_index,
                                              rerank_model=cfg["llm_cfg"]["rerank_model"],
                                              similarity_top_k=cfg["llama_index_cfg"]["sentence_retrieval"]["similarity_top_k"],
                                              rerank_top_n=cfg["llama_index_cfg"]["sentence_retrieval"]["rerank_top_n"])

elif llama_eval_method == "auto_merging_retrieval":
    # Rebuild storage context
    storage_context = StorageContext.from_defaults(
        persist_dir=cfg["llama_index_cfg"]["auto_merging_retrieval"]["index_save_dir"])
    automerging_index = load_index_from_storage(storage_context)
    engine = get_automerging_query_engine(
        automerging_index,
        rerank_model=cfg["llm_cfg"]["rerank_model"],
        similarity_top_k=cfg["llama_index_cfg"]["auto_merging_retrieval"]["similarity_top_k"],
        rerank_top_n=cfg["llama_index_cfg"]["auto_merging_retrieval"]["rerank_top_n"],
    )

# window_response = engine.query(
#     "Explain is the architecture of vision transformer model"
# )
# print(str(window_response))

questions_df = pd.read_excel(os.path.join(
    here(cfg["eval_questions_dir"]), cfg["eval_file_name"]))
print("\n\n")
for idx, row in questions_df.iterrows():
    inference_start_time = time.time()
    question = row["question"]
    result = engine.query(question)
    infernece_time = time.time() - inference_start_time
    answer = result.response
    column_name = f"llama_index_{llama_eval_method}_result"
    questions_df.at[idx, column_name] = answer
    time_column_name = f"llama_index_{llama_eval_method}_inference_time"
    questions_df.at[idx, time_column_name] = round(infernece_time, 2)
    terminal_message = f"Question {idx}:\n{question}\n\nAnswer:\n{answer}"
    print(terminal_message)
    print(f"Inference time: {infernece_time}")
    print("--------------------------------------\n\n")

questions_df.to_excel(
    os.path.join(here(cfg["eval_questions_dir"]), cfg["eval_file_name"]), index=False)
print(
    f"Execution was successfull and {cfg['eval_file_name']} file is saved.\n\n")
end_time = time.time()
# Calculate and print the execution time
execution_time = end_time - start_time
print(f"Execution Time: {round(execution_time, 2)} seconds")
