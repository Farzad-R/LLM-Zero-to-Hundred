import pandas as pd
from pyprojroot import here
from pprint import pprint
import yaml
import os
import jsonlines
from typing import List


def concat_all_json_files(data_dir: str) -> pd.DataFrame:
    """
    Concatenate data from multiple JSON files into a single DataFrame.

    Parameters:
    - data_dir (str): Directory path containing JSON files.

    Returns:
    - pd.DataFrame: Concatenated DataFrame.
    """
    df = pd.DataFrame()
    for json_file in os.listdir(data_dir):
        if not json_file == "product_user_manual.json":
            tmp_df = pd.read_json(os.path.join(data_dir, json_file))
            df = pd.concat([df, tmp_df], ignore_index=True)
    return df


def prepare_interim_qa_dataset(df: pd.DataFrame) -> List:
    """
    Prepare a question-answer dataset for interim processing.

    Parameters:
    - df (pd.DataFrame): DataFrame containing question and answer columns.

    Returns:
    - List[dict]: List of dictionaries with 'question' and 'answer' keys.
    """
    finetuning_dataset = []
    for i in range(len(df["question"])):
        question = f"### Question:\n{df['question'][i]}\n\n\n### Answer:\n"
        answer = df['answer'][i]
        finetuning_dataset.append(
            {"question": question, "answer": answer})

    return finetuning_dataset


if __name__ == "__main__":
    with open(here("configs/config.yml")) as cfg:
        app_config = yaml.load(cfg, Loader=yaml.FullLoader)
    all_q_a_df = concat_all_json_files(here(app_config["json_dir"]["dir"]))
    print("dataframe shape:", all_q_a_df.shape)
    print("Dataframe head:\n", all_q_a_df.head(4))
    dataset = prepare_interim_qa_dataset(all_q_a_df)
    with jsonlines.open(here(app_config["interim_dir"]["cubetriangle_qa"]), 'w') as writer:
        writer.write_all(dataset)


"""
To run the module:
In the parent folder, open a terminal and execute:
```
python src/data_preparation/prepare_qa_interim_ds.py
```
"""
