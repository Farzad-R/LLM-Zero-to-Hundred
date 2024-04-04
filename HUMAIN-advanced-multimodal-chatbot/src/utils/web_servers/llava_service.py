from flask import Flask, request, jsonify
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
from load_web_service_config import LoadWebServicesConfig

WEB_SERVICE_CFG = LoadWebServicesConfig()


def load_llava(quantized=True):
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf")
    if quantized:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("===============================================")
        print("Loading the quantized version of the model:")
        print("===============================================")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quantization_config, device_map="auto")
    else:
        print("Loading the full model:")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda:0")
    return processor, model


processor, model = load_llava(quantized=True)

app = Flask(__name__)


@app.route('/interact_with_llava', methods=['POST'])
def interact_with_llava():
    try:
        data = request.get_json()
        prompt = data['prompt']
        url = data['image_url']
        max_output_token = data['max_output_token']
        print("==================")
        print(url)
        print(prompt)
        print("==================")
        prompt = f"[INST] <image>\n{prompt} [/INST]"
        try:
            image = Image.open(url)
        except:
            image = None
        inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=max_output_token)
        print(output)
        response = processor.decode(output[0], skip_special_tokens=True)
        print(response)
        del output
        del inputs
        del image
        torch.cuda.empty_cache()
        return jsonify({'text': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=WEB_SERVICE_CFG.llava_service_port)
