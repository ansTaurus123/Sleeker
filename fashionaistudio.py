import streamlit as st
import mediapy as media
import random
import sys
import torch

from diffusers import AutoPipelineForText2Image

st.title("Fashion Design Generator")

# Load the pipeline
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    )
pipe = pipe.to("cuda")

# Ask the user for information
topwear = st.text_input("Enter the type of upper garment you want (for example, shirt, blouse, jacket): ")
bottomwear = st.text_input("Enter the type of lower garment you want (for example, pants, skirt, shorts): ")
accessories = st.text_input("Do you want to add any accessories? If so, what? ")

# Create the prompt based on the user's preferences
prompt = f"Generate a unique and spectacular fashion design for men. The upper garment must be a {topwear}, the lower garment some {bottomwear}, and include {accessories} as accessories. The design must be photorealistic, with cinematic lighting and hyper-realistic details. Use a cinematic composition, focusing on latest fashion and trends."

# Generate the fashion design
seed = random.randint(0, sys.maxsize)
num_inference_steps = 4
images = pipe(
    prompt=prompt,
    guidance_scale=0.0,
    num_inference_steps=num_inference_steps,
    generator=torch.Generator("cuda").manual_seed(seed),
).images

# Show the generated design
st.write(f"Prompt: {prompt}\nSeed: {seed}")
media.show_images(images)
images[0].save("output.jpg")

st.image("output.jpg", caption="Generated Fashion Design")
