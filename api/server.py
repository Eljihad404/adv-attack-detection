from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import tempfile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import InferenceSystem

app = FastAPI(title="Secure Chest X-Ray Analysis API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
inference_system = None

@app.on_event("startup")
async def startup_event():
    global inference_system
    print("Loading inference system...")
    try:
        # Models are in the parent directory (project root)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_path = os.path.join(project_root, "global_model_final.pth")
        detector_path = os.path.join(project_root, "poison_detector.pth")
        
        print(f"Loading models from: {project_root}")
        inference_system = InferenceSystem(model_path=model_path, detector_path=detector_path)
        print("Inference system loaded successfully.")
    except Exception as e:
        print(f"Error loading inference system: {e}")
        # We don't exit here to allow the API to start, but /predict will fail

@app.post("/predict")
async def predict(file: UploadFile = File(...), use_detector: bool = Form(True)):
    global inference_system
    if inference_system is None:
        raise HTTPException(status_code=503, detail="Inference system not initialized")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Create a temporary file to save the uploaded image
    # The InferenceSystem expects a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # Run inference
        result = inference_system.predict_single_image(tmp_path, check_adversarial=use_detector)
        
        # Clean up the file path from the result before returning (optional, but cleaner)
        if 'image_path' in result:
           result['image_path'] = os.path.basename(result['image_path'])

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
