import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download

def generate_image(text):
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_lora.safetensors"  # Use the correct ckpt for your step setting!

    # Load model.
    pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.load_lora_weights(hf_hub_download(repo, ckpt))
    pipe.fuse_lora()

    # Ensure sampler uses "trailing" timesteps.
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # Ensure using the same inference steps as the loaded model and CFG set to 0.
    pipe(text, num_inference_steps=4, guidance_scale=0).images[0].save("output.png")

    return "output.png"

demo = gr.Interface(
    fn=generate_image,
    inputs=gr.components.Textbox(label='Input text', lines=5),
    outputs="image"
)

if __name__ == '__main__':
    demo.launch()