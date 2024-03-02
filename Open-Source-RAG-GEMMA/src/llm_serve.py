import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils.load_config import LoadConfig

APPCFG = LoadConfig()

app = Flask(__name__)

# Load the LLM and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    APPCFG.llm_engine, token=APPCFG.gemma_token, device=APPCFG.device)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="google/gemma-7b-it",
                                             token=APPCFG.gemma_token,
                                             torch_dtype=torch.float16,
                                             device_map=APPCFG.device
                                             )
app_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)


@app.route("/generate_text", methods=["POST"])
def generate_Text():
    data = request.json
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", 1000)
    do_sample = data.get("do_sample", True)
    temperature = data.get("temperature", 0.1)
    top_k = data.get("top_k", 50)
    top_p = data.get("top_p", 0.95)

    tokenized_prompt = app_pipeline.tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True)
    outputs = app_pipeline(
        tokenized_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    return jsonify({"response": outputs[0]["generated_text"][len(tokenized_prompt):]})


if __name__ == "__main__":
    app.run(debug=False, port=8888)
