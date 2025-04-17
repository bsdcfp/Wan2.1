import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from datetime import datetime
import os
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)])

# Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
# model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
model_id = "/model_zoo/Wan2.1-I2V-14B-480P-Diffusers/"
num_inference_steps=30

image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

# image = load_image(
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
# )
image_path = "examples/i2v_input.JPG"
image = load_image(image_path)

max_area = 480 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

# prompt = (
#     "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
#     "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
# )
prompt = ("Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
        "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
        "Blurred beach scenery forms the background featuring crystal-clear waters, "
        "distant green hills, and a blue sky dotted with white clouds. "
        "The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. "
        "A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.")

negative_prompt = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, "
                "paintings, images, static, overall gray, worst quality, low quality, "
                "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
                "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
                "still picture, messy background, three legs, many people in the background, walking backwards")

# warmup
pipe(
    image=image, 
    prompt=prompt, 
    negative_prompt=negative_prompt,
    height=height, 
    width=width, 
    num_frames=81, 
    num_inference_steps=1,
    guidance_scale=5.0
).frames[0]
# 开始总计时  
start_time = time.perf_counter()  
output = pipe(
    image=image, 
    prompt=prompt, 
    negative_prompt=negative_prompt,
    height=height, 
    width=width, 
    num_frames=81, 
    num_inference_steps=num_inference_steps,
    guidance_scale=5.0
).frames[0]
# 结束总计时  
elapsed = time.perf_counter() - start_time  
logging.info(f"Inference execution time: {elapsed:.2f} seconds ({int(elapsed // 60)} minutes and {int(elapsed % 60)} seconds)") 

formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
formatted_prompt = prompt.replace(" ", "_").replace("/","_")[:50]
output_dir="./result"
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
model_name=model_id.strip().strip("/").split("/")[-1]
output_file = (f"{model_name}_steps{num_inference_steps}" 
            f"_{formatted_prompt}_{formatted_time}"
            f".mp4")
output_file = os.path.join(output_dir, output_file)
print(output_file)
export_to_video(output, output_file, fps=16)
