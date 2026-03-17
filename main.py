from fastapi import FastAPI, File, UploadFile
from PIL import Image
from PIL.ExifTags import TAGS
import io
import numpy as np

app = FastAPI()


# 🔍 Function to extract metadata
def get_metadata(image):
    try:
        exif_data = image._getexif()

        if not exif_data:
            return None

        metadata = {}
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            metadata[decoded] = value

        return metadata

    except:
        return None


def noise_score(image):
    # Convert to grayscale
    gray = np.array(image.convert("L"))

    # Calculate variance (measure of noise)
    variance = np.var(gray)

    return variance


@app.get("/")
def home():
    return {"message": "Backend running 🚀"}


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert to image
    image = Image.open(io.BytesIO(contents))

    # 🔍 Metadata check
    metadata = get_metadata(image)

    if metadata is None:
        metadata_flag = "MISSING ⚠️"
        metadata_score = 1  # suspicious
    else:
        metadata_flag = "PRESENT"
        metadata_score = 0  # normal

    noise = noise_score(image)

    if noise < 500:
        noise_flag = "LOW NOISE (AI-like) ⚠️"
        noise_score_flag = 1
    else:
        noise_flag = "NORMAL NOISE"
        noise_score_flag = 0

    return {
        "filename": file.filename,
        "metadata_flag": metadata_flag,
        "metadata_score": metadata_score,
        "noise_value": float(noise),
        "noise_flag": noise_flag,
        "noise_score": noise_score_flag,
    }
