# Import necessary libraries
from fastapi import FastAPI, File, UploadFile  # FastAPI for creating the web API
import io  # To handle byte streams
from PIL import Image  # To handle image processing
from transformers import BlipProcessor, BlipForConditionalGeneration  # For BLIP image captioning model

# Load the BLIP processor and model
# The processor handles image preprocessing, and the model generates captions for images.
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize the FastAPI application
app = FastAPI()

# Define the /predict endpoint
@app.post("/predict")
async def generate_caption_from_image(file: UploadFile = File(...)):
    """
    Endpoint to generate an image caption.

    Parameters:
        file (UploadFile): The image file uploaded by the user.

    Returns:
        dict: A dictionary containing either the generated caption or an error message.
    """
    try:
        # Read the uploaded image file as bytes
        image_data = await file.read()

        # Open the image using PIL and ensure it is in RGB format
        raw_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess the image using the BLIP processor
        inputs = processor(raw_image, return_tensors="pt")

        # Generate a caption for the image using the BLIP model
        out = model.generate(**inputs, max_length=100)

        # Decode the generated output to a human-readable caption
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Return the generated caption as a JSON response
        return {"description": caption}

    except Exception as e:
        # If any error occurs, return an error message
        return {"error": str(e)}
