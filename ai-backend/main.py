from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model from ultralytics
model = YOLO('yolo11n.pt', task = 'detect')

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Load the image
    try:
        image = Image.open(io.BytesIO(await file.read()))
    
        # Perform inference
        results = model(image)
    
        # Parse detections into a structured response
        response = []
        
        for box in results[0].boxes:
            detection = {
                "xyxy": box.xyxy.tolist(),
                "confidence": box.conf.item(),
                "class": box.cls.item()
            }
            response.append(detection)
        return {"results": response}
    
    except Exception as e:
        return {"error": "Failed to process image", "message": str(e)}
