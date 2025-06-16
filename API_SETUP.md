
# Flask API Setup Guide

This guide explains how to set up and run the Flask API backend for the CIEEEN AI-LMS system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Setup Instructions

### 1. Navigate to the API directory
```bash
cd api
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the AI models (optional - for real predictions)
```bash
# Go back to project root
cd ..

# Install AI model dependencies
pip install -r requirements.txt

# Run the model training pipeline
python ai_models/run_all_models.py
```

### 5. Start the Flask server
```bash
cd api
python app.py
```

The API server will start on `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/health`
- Returns server status and loaded models information

### Certification Prediction
- **POST** `/api/certification/predict`
- Body:
```json
{
  "trainingHours": "45",
  "assessmentScore": "85",
  "complianceHistory": "90",
  "continuingEducation": "25",
  "yearsExperience": "5",
  "previousCertifications": "2"
}
```

### Risk Analysis
- **POST** `/api/risk/analyze`
- Body:
```json
{
  "learningHours": "15",
  "engagementScore": "75",
  "assessmentAttempts": "2",
  "coursesCompleted": "3",
  "lastActivityDays": "7",
  "complianceRating": "85"
}
```

### Course Recommendations
- **POST** `/api/recommendations/courses`
- Body:
```json
{
  "specialization": "power-engineering",
  "experienceLevel": "intermediate",
  "learningGoals": "Certification preparation"
}
```

## Testing the API

### Using curl
```bash
# Health check
curl http://localhost:5000/health

# Test certification prediction
curl -X POST http://localhost:5000/api/certification/predict \
  -H "Content-Type: application/json" \
  -d '{"trainingHours":"45","assessmentScore":"85","complianceHistory":"90","continuingEducation":"25","yearsExperience":"5","previousCertifications":"2"}'
```

### Using the Frontend
1. Start the Flask API server (as described above)
2. Start the React frontend:
```bash
npm run dev
```
3. Navigate to the frontend URLs and use the forms to test the API

## Model Integration

The API checks for trained models in `../ai_models/trained_models/` directory:
- `certification_predictor.pkl` - Certification eligibility model
- `risk_predictor.pkl` - Performance risk model
- `user_item_matrix.pkl` - Course recommendation model components

If models are not found, the API falls back to rule-based logic for demonstration purposes.

## Troubleshooting

### CORS Issues
The API includes CORS headers for development. In production, configure CORS properly for your domain.

### Port Conflicts
If port 5000 is in use, modify the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port here
```

And update the frontend API URLs accordingly.

### Missing Dependencies
If you get import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```
