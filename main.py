from fastapi import FastAPI, File, UploadFile
from PIL import Image
from PIL.ExifTags import TAGS
from transformers import pipeline
import io
import numpy as np
import cv2
import base64

app = FastAPI()

# 🔥 Load model once
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
def noise_score(image):
    gray = np.array(image.convert("L"))
    return np.var(gray)


# -------------------------------
# EDGE FUNCTION
# -------------------------------
def edge_score(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.mean(edges)


def generate_heatmap(image):
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

    return heatmap


@app.get("/")
def home():
    return {"message": "Backend running 🚀"}


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # -------------------------------
    # MODEL PREDICTION
    # -------------------------------
    result = classifier(image)

    human_score = 0
    ai_score = 0

    for r in result:
        if r["label"] == "human":
            human_score = r["score"]
        elif r["label"] == "artificial":
            ai_score = r["score"]

    # -------------------------------
    # METADATA
    # -------------------------------
    metadata = get_metadata(image)

    if metadata is None:
        metadata_flag = "MISSING ⚠️"
        metadata_score = 1
    else:
        metadata_flag = "PRESENT"
        metadata_score = 0

    # -------------------------------
    # NOISE
    # -------------------------------
    noise = noise_score(image)

    if noise < 500:
        noise_flag = "LOW NOISE (AI-like) ⚠️"
        noise_score_flag = 1
    else:
        noise_flag = "NORMAL NOISE"
        noise_score_flag = 0

    # -------------------------------
    # EDGES
    # -------------------------------
    edges = edge_score(image)

    if edges < 5:
        edge_flag = "TOO SMOOTH (AI-like) ⚠️"
        edge_score_flag = 1
    else:
        edge_flag = "NORMAL EDGES"
        edge_score_flag = 0

    heatmap = generate_heatmap(image)

    _, buffer = cv2.imencode(".jpg", heatmap)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    # -------------------------------
    # FINAL DECISION ENGINE 🔥
    # -------------------------------
    ai_votes = 0
    reasons = []

    # Model (strong weight)
    if ai_score > 0.6:
        ai_votes += 2
        reasons.append("Model detects artificial patterns")

    # Metadata
    if metadata_score == 1:
        ai_votes += 1
        reasons.append("Metadata missing")

    # Noise
    if noise_score_flag == 1:
        ai_votes += 1
        reasons.append("Low noise detected (AI-like)")

    # Edges
    if edge_score_flag == 1:
        ai_votes += 1
        reasons.append("Image too smooth (low edge detail)")

    # Final decision
    if ai_votes >= 3:
        final_prediction = "AI GENERATED"
        risk = "HIGH"
    elif ai_votes == 2:
        final_prediction = "UNCERTAIN ⚠️"
        risk = "MEDIUM"
    else:
        final_prediction = "LIKELY REAL"
        risk = "LOW"

    confidence = round(max(human_score, ai_score), 2)

    # -------------------------------
    # FINAL RESPONSE ✅
    # -------------------------------
    return {
        "prediction": final_prediction,
        "confidence": confidence,
        "risk_level": risk,
        "signals": {
            "model_ai_score": round(ai_score, 2),
            "metadata": metadata_flag,
            "noise": noise_flag,
            "edges": edge_flag,
        },
        "explanation": reasons,
        "raw_model_output": result,
        "heatmap": heatmap_base64,
    }