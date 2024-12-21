from fastapi import FastAPI, File, UploadFile
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

app = FastAPI()

@app.post("/predict")
async def generate_caption_from_image(file: UploadFile = File(...)):
    try:

        image_data = await file.read()
        raw_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_length=100)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return {"description": caption}

    except Exception as e:
        return {"error": str(e)}

