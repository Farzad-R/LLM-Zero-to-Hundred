
from typing import List
from datasets import load_dataset
from pyprojroot import here
import yaml
from functools import partial

with open(here("configs/config.yml")) as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

cubetriangle_qa_interim_dir = str(
    here(app_config["interim_dir"]["cubetriangle_qa"]))
cubetriangle_instruction_response_interim_dir = str(
    here(app_config["interim_dir"]["cubetriangle_instruction_response"]))
tokenizer_max_length = 2048


def tokenize_the_data(examples,
                      tokenizer,
                      tokenizer_max_length: int = tokenizer_max_length,
                      column_names: List = ["question", "answer"],
                      data_type: str = "cubetriangle"
                      ):
    # if "question" in examples and "answer" in examples:
    if data_type == "cubetriangle":
        text = examples[column_names[0]][0] + examples[column_names[1]][0]
    elif data_type == "guanaco":
        text = examples["text"][0]
    else:
        raise ValueError(
            "Invalid data_type. Supported values are 'cubetriangle' and 'guanaco'.")

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
    )

    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        tokenizer_max_length
    )
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length
    )
    return tokenized_inputs


def prepare_cubetrianlge_qa_dataset(tokenizer,
                                    tokenizer_max_length: int = tokenizer_max_length,
                                    column_names: List = [
                                        "question", "answer"],
                                    data_dir: str = cubetriangle_qa_interim_dir,
                                    data_type: str = "cubetriangle"
                                    ):
    finetuning_dataset = load_dataset(
        'json', data_files=data_dir, split="train")
    print("Raw dataset shape:", finetuning_dataset)
    # Define a partial function with fixed arguments
    partial_tokenize_function = partial(
        tokenize_the_data,
        tokenizer=tokenizer,
        tokenizer_max_length=tokenizer_max_length,
        column_names=column_names,
        data_type=data_type
    )
    # print("Processed data description:\n")
    # print(finetuning_dataset)
    # print("---------------------------")
    tokenized_dataset = finetuning_dataset.map(
        partial_tokenize_function,
        batched=True,
        batch_size=1,
        drop_last_batch=True
    )
    tokenized_dataset = tokenized_dataset.add_column(
        "labels", tokenized_dataset["input_ids"])
    return tokenized_dataset


def prepare_cubetriangle_instruction_response_dataset(tokenizer,
                                                      tokenizer_max_length: int = tokenizer_max_length,
                                                      column_names: List = [
                                                          "instruction", "response"],
                                                      data_dir: str = cubetriangle_instruction_response_interim_dir,
                                                      data_type: str = "cubetriangle"
                                                      ):
    finetuning_dataset = load_dataset(
        'json', data_files=data_dir, split="train")
    # Define a partial function with fixed arguments
    partial_tokenize_function = partial(
        tokenize_the_data,
        tokenizer=tokenizer,
        tokenizer_max_length=tokenizer_max_length,
        column_names=column_names,
        data_type=data_type
    )
    # print("Processed data description:\n")
    # print(finetuning_dataset)
    # print("---------------------------")
    tokenized_dataset = finetuning_dataset.map(
        partial_tokenize_function,
        batched=True,
        batch_size=1,
        drop_last_batch=True
    )
    tokenized_dataset = tokenized_dataset.add_column(
        "labels", tokenized_dataset["input_ids"])
    return tokenized_dataset


def prepare_openassistant_guanaco_dataset(tokenizer,
                                          tokenizer_max_length: int = tokenizer_max_length,
                                          data_type: str = "guanaco"
                                          ):
    guanaco_train = load_dataset(
        path="timdettmers/openassistant-guanaco", split="train")
    guanaco_test = load_dataset(
        path="timdettmers/openassistant-guanaco", split="test")
    # Define a partial function with fixed arguments
    partial_tokenize_function = partial(
        tokenize_the_data,
        tokenizer=tokenizer,
        tokenizer_max_length=tokenizer_max_length,
        data_type=data_type
    )
    print("Processed data description:\n")
    print(guanaco_train)
    print("---------------------------")
    tokenized_guanaco_train_dataset = guanaco_train.map(
        partial_tokenize_function,
        batched=True,
        batch_size=1,
        drop_last_batch=True
    )
    tokenized_guanaco_test_dataset = guanaco_test.map(
        partial_tokenize_function,
        batched=True,
        batch_size=1,
        drop_last_batch=True
    )
    tokenized_guanaco_train_dataset = tokenized_guanaco_train_dataset.add_column(
        "labels", tokenized_guanaco_train_dataset["input_ids"])
    tokenized_guanaco_test_dataset = tokenized_guanaco_test_dataset.add_column(
        "labels", tokenized_guanaco_test_dataset["input_ids"])
    return tokenized_guanaco_train_dataset, tokenized_guanaco_test_dataset
