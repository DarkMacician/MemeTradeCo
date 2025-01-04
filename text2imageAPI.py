from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from io import BytesIO
import torch
from diffusers import DiffusionPipeline
import base64
from PIL import Image

# Initialize the app
app = FastAPI()

# Load the Stable Diffusion model
model_path = "stabilityai/stable-diffusion-xl-base-1.0"

print("Loading model...")
pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

pipe.to("cuda")  # Move the model to GPU

# Define request and response schema
class TextPrompt(BaseModel):
    prompt: str

class ImageResponse(BaseModel):
    base64_image: str

# Endpoint for generating an image
@app.post("/generate_image", response_model=ImageResponse)
async def generate_image(prompt: TextPrompt):
    try:
        # Generate the image
        image = pipe(prompt=prompt.prompt).images[0]

        # Convert the image to a base64 string
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"base64_image": image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))