# IoT Anomaly Detection - Vercel Deployment

This project provides a real-time anomaly detection system for IoT sensor data using LSTM neural networks, now optimized for deployment on Vercel.

## Architecture

The deployment consists of:
- **FastAPI Backend** (`api/index.py`) - RESTful API for anomaly detection
- **Web Frontend** (`index.html`) - Interactive web interface
- **Vercel Configuration** (`vercel.json`) - Deployment settings

## Features

- Real-time anomaly detection using LSTM neural networks
- RESTful API endpoints
- Interactive web interface with visualizations
- CSV file upload support
- JSON data support
- Downloadable results
- Responsive design

## API Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check endpoint

### `POST /predict`
Upload CSV file for anomaly detection
- **Body**: `multipart/form-data` with file field
- **Response**: JSON with detection results and metrics

### `POST /predict-json`
Submit JSON data for anomaly detection
- **Body**: 
```json
{
  "data": [
    {"Temperature": 25.5, "Humidity": 60.2, "Air Quality": 150, "Light": 300, "Loudness": 45},
    ...
  ]
}
```
- **Response**: JSON with detection results and metrics

### `GET /model-info`
Get model information and expected data format

## Expected Data Format

CSV files should contain the following columns:
- Temperature (numeric)
- Humidity (numeric)  
- Air Quality (numeric)
- Light (numeric)
- Loudness (numeric)

Optional Time column (Unix timestamp format) is also supported.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the FastAPI server:
```bash
uvicorn api.index:app --reload
```

3. Open `http://localhost:8000` in your browser

## Vercel Deployment

### Prerequisites
- Vercel account
- Vercel CLI installed
- Git repository

### Deployment Steps

1. **Initialize Git Repository**
```bash
git init
git add .
git commit -m "Initial commit"
```

2. **Install Vercel CLI**
```bash
npm i -g vercel
```

3. **Login to Vercel**
```bash
vercel login
```

4. **Deploy to Vercel**
```bash
vercel
```

5. **Follow the prompts**:
   - Set up and deploy project
   - Link to existing Vercel account
   - Confirm project settings
   - Deploy!

### Environment Variables

The deployment uses Python 3.10. The `vercel.json` configuration handles:
- Function timeout (30 seconds)
- Python version specification
- Route configuration

### File Structure for Deployment

```
.
|-- api/
|   |-- index.py          # FastAPI backend
|-- index.html            # Web frontend
|-- vercel.json           # Vercel configuration
|-- requirements.txt      # Python dependencies
|-- lstm_model.keras      # Trained model
|-- dataset_final.csv     # Reference dataset
```

## Model Details

- **Architecture**: LSTM Autoencoder
- **Sequence Length**: 10 timesteps
- **Features**: 5 sensor readings
- **Threshold**: 95th percentile of reconstruction errors
- **Input Shape**: (None, 10, 5)

## Performance Considerations

- **Function Timeout**: Set to 30 seconds for processing larger datasets
- **Memory Usage**: Model loaded once per function instance
- **Cold Starts**: First request may be slower due to model loading

## Monitoring and Debugging

- Check Vercel function logs for deployment issues
- Monitor response times for performance
- Use `/health` endpoint for status checks
- Review `/model-info` for model configuration

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure `lstm_model.keras` is in the root directory
   - Check file permissions and paths

2. **Memory Issues**
   - Reduce input data size for processing
   - Optimize sequence length

3. **Timeout Errors**
   - Increase function timeout in `vercel.json`
   - Process data in batches

4. **CORS Issues**
   - API includes CORS middleware for frontend access

### Error Responses

The API returns detailed error messages:
- `400`: Invalid data format or column mismatch
- `500`: Processing errors
- `503`: Model not loaded

## Usage Examples

### Python Client
```python
import requests

# Upload CSV file
with open('sensor_data.csv', 'rb') as f:
    response = requests.post('https://your-app.vercel.app/predict', files={'file': f})
    result = response.json()
    print(f"Anomalies detected: {result['anomaly_points']}")
```

### JavaScript Client
```javascript
// Upload file
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log('Results:', data));
```

## License

This project is for educational and research purposes. Please ensure compliance with data privacy regulations when processing IoT sensor data.
