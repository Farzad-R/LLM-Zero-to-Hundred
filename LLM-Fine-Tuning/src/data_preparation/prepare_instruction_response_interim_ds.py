from pyprojroot import here
import yaml
import pandas as pd
import jsonlines


def prepare_instruction_response_interim_ds(data_dir):
    df = pd.read_json(str(here(data_dir)))
    print("dataframe shape:", df.shape)
    print("Dataframe head:\n", df.head(4))
    finetuning_dataset = []
    for i in range(len(df["instruction"])):
        instruction = f"### Instruction:\n{df['instruction'][i]}\n\n\n### Response:\n"
        response = df['response'][i]
        finetuning_dataset.append(
            {"instruction": instruction, "response": response})
    return finetuning_dataset


if __name__ == "__main__":
    with open(here("configs/config.yml")) as cfg:
        app_config = yaml.load(cfg, Loader=yaml.FullLoader)
    dataset = prepare_instruction_response_interim_ds(
        data_dir=app_config["json_dir"]["product_user_manual_instruction_response"])
    with jsonlines.open(here(app_config["interim_dir"]["cubetriangle_instruction_response"]), 'w') as writer:
        writer.write_all(dataset)


"""
To run the module:
In the parent folder, open a terminal and execute:
```
python src/data_preparation/prepare_instruction_response_interim_ds.py
```
"""
