from flask import Flask, request, jsonify
import time
from pyprojroot import here
import torch
from load_web_service_config import LoadWebServicesConfig

WEB_SERVICE_CFG = LoadWebServicesConfig()


def load_stable_diffusionv_1_5():
    """
    Load the Stable Diffusion v1.5 model.

    Returns:
        StableDiffusionPipeline: The loaded model pipeline.
    """
    from diffusers import StableDiffusionPipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto")
    pipe = pipe.to("cuda")
    return pipe


def load_stable_diffusion_xl_base_1_0():
    """
    Load the Stable Diffusion XL Base 1.0 model.

    Returns:
        StableDiffusionXLPipeline: The loaded model pipeline.
    """
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download

    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    # Use the correct ckpt for your step setting!
    ckpt = "sdxl_lightning_4step_lora.safetensors"

    # Load model.
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.load_lora_weights(hf_hub_download(repo, ckpt))
    pipe.fuse_lora()

    # Ensure sampler uses "trailing" timesteps.
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing")
    return pipe


pipe = load_stable_diffusion_xl_base_1_0()

app = Flask(__name__)


@app.route('/generate_image', methods=['POST'])
def generate_image():
    """
    Endpoint for generating images based on prompts.

    Expects a POST request with JSON data containing:
    - 'prompt': The prompt for generating the image.

    Returns a JSON response containing the path to the generated image.

    Example JSON request:
    {
        "prompt": "A beautiful sunset over the mountains."
    }

    Example JSON response:
    {
        "text": "data/generated_images/1648735800.png"
    }
    """
    try:
        data = request.get_json()
        prompt = data['prompt']
        print(prompt)
        print("===========================")
        current_time_unix = int(time.time())
        image_dir = here(f"data/generated_images/{current_time_unix}.png")
        print(prompt)
        # Ensure using the same inference steps as the loaded model and CFG set to 0.
        pipe(prompt, num_inference_steps=4,
             guidance_scale=0).images[0].save(f"{image_dir}")
        torch.cuda.empty_cache()
        return jsonify({'text': f"data/generated_images/{current_time_unix}.png"}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, port=WEB_SERVICE_CFG.stable_diffusion_service_port)
