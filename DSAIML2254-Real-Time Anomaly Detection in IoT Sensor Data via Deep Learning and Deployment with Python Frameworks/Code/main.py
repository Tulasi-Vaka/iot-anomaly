# Main FastAPI application for Vercel deployment
import os
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any
import io
import base64

app = FastAPI(title="IoT Anomaly Detection API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
model = None
scaler = None
ref_data = None
SEQUENCE_LENGTH = 10

def load_model_and_data():
    """Load the LSTM model and reference data"""
    global model, scaler, ref_data
    
    try:
        # Load model
        model_path = "lstm_model.keras"
        if not os.path.exists(model_path):
            # Try alternative path
            model_path = "../lstm_model.keras"
        
        model = load_model(model_path)
        
        # Load reference data
        data_path = "dataset_final.csv"
        if not os.path.exists(data_path):
            data_path = "../dataset_final.csv"
            
        ref_data = pd.read_csv(data_path)
        if 'Time' in ref_data.columns:
            ref_data['Time'] = pd.to_datetime(ref_data['Time'], unit='s')
            ref_data.set_index('Time', inplace=True)
        
        # Initialize scaler
        scaler = MinMaxScaler()
        scaler.fit(ref_data)
        
        print("Model and data loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def create_sequences(data, sequence_length=SEQUENCE_LENGTH):
    """Create sequences for LSTM prediction"""
    X = []
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i+sequence_length])
    return np.array(X)

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    success = load_model_and_data()
    if not success:
        print("Warning: Failed to load model and data")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "IoT Anomaly Detection API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": ref_data is not None
    }

@app.post("/predict")
async def detect_anomalies(file: UploadFile = File(...)):
    """
    Detect anomalies in uploaded CSV file
    """
    if model is None or scaler is None or ref_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Process Time column if it exists
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'], unit='s')
            df.set_index('Time', inplace=True)
        
        # Validate columns
        ref_columns = ref_data.columns.tolist()
        if set(ref_columns) != set(df.columns):
            raise HTTPException(
                status_code=400, 
                detail=f"Column mismatch. Expected: {ref_columns}, Got: {df.columns.tolist()}"
            )
        
        # Reorder columns to match reference data
        df = df[ref_columns]
        
        # Normalize data
        scaled_data = scaler.transform(df)
        
        # Create sequences
        sequences = create_sequences(scaled_data, SEQUENCE_LENGTH)
        
        # Make predictions
        predictions = model.predict(sequences, verbose=0)
        reconstruction_error = np.mean(np.square(sequences - predictions), axis=(1,2))
        
        # Calculate threshold and detect anomalies
        threshold = np.percentile(reconstruction_error, 95)
        anomalies = reconstruction_error > threshold
        
        # Prepare results
        results_df = df.copy()
        results_df['Anomaly'] = False
        results_df.iloc[SEQUENCE_LENGTH-1:, -1] = anomalies
        
        # Convert to JSON-serializable format
        results = {
            "status": "success",
            "total_data_points": len(results_df),
            "anomaly_points": int(results_df['Anomaly'].sum()),
            "anomaly_percentage": float((results_df['Anomaly'].sum() / len(results_df)) * 100),
            "threshold": float(threshold),
            "data": results_df.to_dict('records'),
            "reconstruction_errors": reconstruction_error.tolist(),
            "columns": results_df.columns.tolist()
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/predict-json")
async def detect_anomalies_json(data: Dict[str, Any]):
    """
    Detect anomalies in JSON data
    Expected format:
    {
        "data": [
            {"Temperature": 25.5, "Humidity": 60.2, "Air Quality": 150, "Light": 300, "Loudness": 45},
            ...
        ]
    }
    """
    if model is None or scaler is None or ref_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert JSON to DataFrame
        df = pd.DataFrame(data["data"])
        
        # Validate columns
        ref_columns = ref_data.columns.tolist()
        if set(ref_columns) != set(df.columns):
            raise HTTPException(
                status_code=400, 
                detail=f"Column mismatch. Expected: {ref_columns}, Got: {df.columns.tolist()}"
            )
        
        # Reorder columns to match reference data
        df = df[ref_columns]
        
        # Normalize data
        scaled_data = scaler.transform(df)
        
        # Create sequences
        sequences = create_sequences(scaled_data, SEQUENCE_LENGTH)
        
        # Make predictions
        predictions = model.predict(sequences, verbose=0)
        reconstruction_error = np.mean(np.square(sequences - predictions), axis=(1,2))
        
        # Calculate threshold and detect anomalies
        threshold = np.percentile(reconstruction_error, 95)
        anomalies = reconstruction_error > threshold
        
        # Prepare results
        results_df = df.copy()
        results_df['Anomaly'] = False
        results_df.iloc[SEQUENCE_LENGTH-1:, -1] = anomalies
        
        # Convert to JSON-serializable format
        results = {
            "status": "success",
            "total_data_points": len(results_df),
            "anomaly_points": int(results_df['Anomaly'].sum()),
            "anomaly_percentage": float((results_df['Anomaly'].sum() / len(results_df)) * 100),
            "threshold": float(threshold),
            "data": results_df.to_dict('records'),
            "reconstruction_errors": reconstruction_error.tolist(),
            "columns": results_df.columns.tolist()
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_input_shape": model.input_shape,
        "model_output_shape": model.output_shape,
        "sequence_length": SEQUENCE_LENGTH,
        "expected_columns": ref_data.columns.tolist() if ref_data is not None else None,
        "model_summary": str(model.summary())
    }

# Export for Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
