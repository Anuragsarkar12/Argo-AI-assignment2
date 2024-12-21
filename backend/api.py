import io
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pydantic import BaseModel

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Create the FastAPI app
app = FastAPI()

# Define a Pydantic model for the response
class DescriptionResponse(BaseModel):
    description: str

# Max file size (e.g., 5 MB)
MAX_FILE_SIZE_MB = 5

# Function to validate image size
def is_file_size_valid(file: UploadFile) -> bool:
    file_size = len(file.file.read()) / (1024 * 1024)  # Size in MB
    file.file.seek(0)  # Reset the file pointer after reading
    return file_size <= MAX_FILE_SIZE_MB

# Route to process the image and generate description
@app.post("/predict", response_model=DescriptionResponse)
async def generate_caption_from_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Invalid file type. Please upload a valid image (PNG, JPG, JPEG)."
            )

        # Validate file size
        if not is_file_size_valid(file):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds {MAX_FILE_SIZE_MB} MB limit. Please upload a smaller file."
            )

        # Read image and process
        image_data = await file.read()
        raw_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Generate caption using BLIP model
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_length=100)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return {"description": caption}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while processing the image.")
