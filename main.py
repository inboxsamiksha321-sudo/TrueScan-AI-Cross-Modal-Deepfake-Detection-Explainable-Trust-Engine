from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import pipeline
import io

classifier = pipeline("image-classification", model="umm-maybe/AI-image-detector")

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Backend is running 🚀"}


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents))

    # Run prediction
    result = classifier(image)

    return {"filename": file.filename, "prediction": result}
