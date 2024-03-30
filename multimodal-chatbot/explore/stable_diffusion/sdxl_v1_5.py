# https://huggingface.co/runwayml/stable-diffusion-v1-5

from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
pipe = pipe.to("cuda")

prompt = "a red supercar with blue rims"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")