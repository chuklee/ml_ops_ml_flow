from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import mlflow
import uvicorn
from typing import List, Optional

class PredictionInput(BaseModel):
    features: List[List[float]]

class ModelVersion(BaseModel):
    version: str

app = FastAPI()

# Global variables for models
current_model = None
next_model = None
prediction_probability = 0.8  # Probability to use current model (for canary deployment)

def load_model(version: Optional[str] = None):
    """Load model from MLflow"""
    try:
        if version:
            model_uri = f"models:/tracking-quickstart/{version}"
        else:
            model_uri = "models:/tracking-quickstart/latest"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize both models with the latest version on startup"""
    global current_model, next_model
    current_model = load_model()
    next_model = load_model()

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """Make predictions using either current or next model based on probability"""
    if np.random.random() < prediction_probability:
        model = current_model
    else:
        model = next_model
    
    try:
        predictions = model.predict(input_data.features)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/update-model")
async def update_model(model_version: ModelVersion):
    """Update the next model to a specific version"""
    global next_model
    try:
        next_model = load_model(model_version.version)
        return {"message": f"Next model updated to version {model_version.version}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model update failed: {str(e)}")

@app.post("/accept-next-model")
async def accept_next_model():
    """Set the next model as the current model"""
    global current_model, next_model
    current_model = next_model
    return {"message": "Next model promoted to current model"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 