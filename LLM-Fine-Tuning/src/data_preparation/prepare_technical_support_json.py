import fitz  # install PyMuPDF
import re
import json
from pyprojroot import here


def extract_qa_from_customer_support_pdf(pdf_path: str, output_json_path: str) -> str:
    """
    Extracts question-answer pairs from a PDF document containing customer support information.

    Parameters:
    - pdf_path (str): The file path to the PDF document.
    - output_json_path (str): The file path to save the extracted question-answer pairs in JSON format.

    Returns:
    - str: The file path to the saved JSON file containing the extracted question-answer pairs.
    """
    qa_list = []
    # Open the PDF file
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text = page.get_text()
            # Define a regular expression to capture question-answer pairs
            pattern = re.compile(
                r"Q:\s*(.+?)(?:(?=\nQ:)|(?=\nA:)|$)(.*?)(?:(?=\nQ:)|(?=\n$))", re.DOTALL)
            # Find all matches in the text
            matches = pattern.findall(text)
            # Process each match and create a dictionary
            for match in matches:
                question = match[0].strip().lstrip('Q:').strip()
                answer = match[1].strip().lstrip('A:').strip()
                question = question.replace('\n', '')
                answer = answer.replace('\n', '')
                # Create a dictionary for each question-answer pair
                qa_dict = {"question": question, "answer": answer}
                qa_list.append(qa_dict)
    # Save the collected Q&A pairs to a JSON file
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(qa_list, json_file, ensure_ascii=False, indent=2)

    print(f"Number of extracted Q&As: {len(qa_list)}")

    return output_json_path


if __name__ == "__main__":
    import yaml
    from pyprojroot import here
    with open(here("configs/config.yml")) as cfg:
        app_config = yaml.load(cfg, Loader=yaml.FullLoader)

    technical_support_pdf_dir = here(
        app_config["raw_data_dir"]["technical_support_pdf_dir"])
    json_dir = here(app_config["json_dir"]["technical_support_qa"])
    result_json_path = extract_qa_from_customer_support_pdf(
        technical_support_pdf_dir, output_json_path=json_dir)
    print(f"Q&A pairs extracted and saved to: {result_json_path}")


"""
To run the module:
In the parent folder, open a terminal and execute:
```
python src/data_preparation/prepare_technical_support_json.py
```
"""
