from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from PIL.ExifTags import TAGS
from transformers import pipeline
import io
import numpy as np
import cv2
import base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
classifier = pipeline("image-classification", model="umm-maybe/AI-image-detector")


# -------------------------------
# METADATA FUNCTION
# -------------------------------
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


# -------------------------------
# NOISE FUNCTION
# -------------------------------
def get_noise_variance(image):
    gray = np.array(image.convert("L"))
    return np.var(gray)


# -------------------------------
# EDGE FUNCTION
# -------------------------------
def get_edge_score(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.mean(edges)


# -------------------------------
# HEATMAP FUNCTION
# -------------------------------
def generate_heatmap(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Resize for frontend performance
    overlay = cv2.resize(overlay, (300, 300))

    return overlay


@app.get("/")
def home():
    return {"message": "AI Image Detection Backend Running "}


@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        return {"error": "Invalid image file"}

    # -------------------------------
    # MODEL PREDICTION
    # -------------------------------
    result = classifier(image)

    if not result:
        return {"error": "Model failed to process image"}

    human_score = 0
    ai_score = 0

    for r in result:
        label = r["label"].lower()

        if any(x in label for x in ["real", "human"]):
            human_score = r["score"]

        elif any(x in label for x in ["fake", "ai", "generated", "artificial"]):
            ai_score = r["score"]

    # -------------------------------
    # METADATA
    # -------------------------------
    metadata = get_metadata(image)

    if metadata is None:
        metadata_flag = "MISSING"
        metadata_score = 1
    else:
        metadata_flag = "PRESENT"
        metadata_score = 0

    # -------------------------------
    # NOISE
    # -------------------------------
    noise_value = get_noise_variance(image)

    if noise_value < 500:
        noise_flag = "LOW NOISE (AI-like)"
        noise_score_flag = 1
    else:
        noise_flag = "NORMAL NOISE"
        noise_score_flag = 0

    # -------------------------------
    # EDGES
    # -------------------------------
    edge_value = get_edge_score(image)

    if edge_value < 5:
        edge_flag = "TOO SMOOTH (AI-like)"
        edge_score_flag = 1
    else:
        edge_flag = "NORMAL EDGES"
        edge_score_flag = 0

    # -------------------------------
    # HEATMAP
    # -------------------------------
    heatmap = generate_heatmap(image)
    _, buffer = cv2.imencode(".jpg", heatmap)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    # -------------------------------
    # DECISION ENGINE
    # -------------------------------
    ai_votes = 0
    reasons = []

    if ai_score > 0.6:
        ai_votes += 2
        reasons.append("Model detects artificial visual patterns")

    if metadata_score == 1:
        ai_votes += 1
        reasons.append("Missing camera metadata (EXIF)")

    if noise_score_flag == 1:
        ai_votes += 1
        reasons.append("Lack of natural sensor noise")

    if edge_score_flag == 1:
        ai_votes += 1
        reasons.append("Unnaturally smooth textures")

    # Final decision
    if ai_votes >= 3:
        prediction = "AI GENERATED (High Confidence)"
        risk = "HIGH"
    elif ai_votes == 2:
        prediction = "UNCERTAIN (Moderate Confidence)"
        risk = "MEDIUM"
    else:
        prediction = "LIKELY REAL (High Confidence)"
        risk = "LOW"

    # -------------------------------
    # SMART SCORING
    # -------------------------------
    ai_likelihood = round(
        (ai_score * 0.6)
        + (metadata_score * 0.1)
        + (noise_score_flag * 0.15)
        + (edge_score_flag * 0.15),
        2,
    )

    ai_probability = round(ai_score * 100, 2)
    real_probability = round(human_score * 100, 2)

    # -------------------------------
    # FINAL RESPONSE
    # -------------------------------
    return {
        "prediction": prediction,
        "risk_level": risk,
        "ai_likelihood": ai_likelihood,
        "ai_probability": ai_probability,
        "real_probability": real_probability,
        "signals": {
            "model_ai_score": round(ai_score, 2),
            "metadata": metadata_flag,
            "noise": noise_flag,
            "edges": edge_flag,
        },
        "explanation": reasons,
        "heatmap": heatmap_base64,
        "raw_model_output": result,
    }
