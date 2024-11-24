from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import cv2
import numpy as np
from io import BytesIO
import base64

app = FastAPI()

AI_BACKEND_URL = "http://ai-backend:8001/detect/"

# Load class names from coco.names
with open('coco.names', 'r') as f:
    class_names = f.read().strip().splitlines()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory storage for detection results
detection_data_storage = {}
image_storage = {}

#connect with the html file
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#use the uploaded file make predictions
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_content = await file.read()
    files = {
        'file': (file.filename, file_content, file.content_type)
    }

    try:
        # Post the image to the AI backend for detection
        response = requests.post(AI_BACKEND_URL, files=files)
        response.raise_for_status()
        detection_data = response.json()

        # Process image and add bounding boxes
        image_np = np.frombuffer(file_content, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        for detection in detection_data.get("results", []):
            bbox = detection["xyxy"][0]
            x_min, y_min, x_max, y_max = map(int, bbox)
            confidence = detection["confidence"]
            class_id = detection["class"]

            class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else "Unknown"

            # Draw bounding box and label
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"Class {class_name}  ({confidence:.2f})"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode image with bounding boxes
        _, encoded_image = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

        # Store the results in memory
        detection_data_storage["json"] = detection_data
        image_storage["image"] = encoded_image.tobytes()

        # Return URLs for downloading JSON and image
        return JSONResponse(content={
            "message": "Processing successful.",
            "json_url": "/save-json",
            "image_url": "/save-image",
            "image_base64": image_base64
        })

    except requests.exceptions.RequestException as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

#This is a GET request handler that provides the ability to download the JSON data generated from object detection.
@app.get("/save-json", response_class=JSONResponse)
async def save_json():
    if "json" in detection_data_storage:
        return JSONResponse(content=detection_data_storage["json"])
    return JSONResponse(content={"error": "No JSON data available."}, status_code=404)

# This is a GET request handler that allows the user to download the processed image with bounding boxes.
@app.get("/save-image")
async def save_image():
    if "image" in image_storage:
        return StreamingResponse(
            BytesIO(image_storage["image"]),
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=bbox_image.jpg"}
        )
    return JSONResponse(content={"error": "No image available."}, status_code=404)
