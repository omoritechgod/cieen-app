
# CIEEEN AI-Powered Learning Management System

## Project Overview

This project builds an AI-powered Learning Management System (LMS) for the Chartered Institute of Electrical and Electronic Engineering of Nigeria (CIEEEN). The system integrates predictive models and recommendation engines to enhance professional development, automate certification workflows, and monitor compliance.

## Core AI Features

### 1. Certification Eligibility Predictor
- **Purpose**: Determines whether a professional qualifies for certification
- **Input**: Training hours, assessment scores, compliance history, continuing education
- **Output**: Approval probability with confidence scores and improvement recommendations
- **Model**: Random Forest/Gradient Boosting with feature importance analysis

### 2. Performance Risk Analyzer  
- **Purpose**: Identifies members at risk of non-compliance or poor performance
- **Input**: Learning patterns, engagement metrics, assessment data
- **Output**: Risk level (LOW/MEDIUM/HIGH) with intervention recommendations
- **Model**: Binary classifier with risk factor analysis

### 3. Course Recommendation System
- **Purpose**: Suggests relevant courses based on learning history and goals
- **Input**: User interactions, course features, member profiles
- **Output**: Personalized course recommendations with similarity scores
- **Model**: Collaborative filtering + Content-based filtering

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js and npm (for web interface)

### Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd cieeen-ai-lms
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install web dependencies**
```bash
npm install
```

### 🤖 Training the AI Models

#### Option 1: Train All Models at Once
```bash
# Run the master training script
python ai_models/run_all_models.py
```

#### Option 2: Train Individual Models

1. **Generate Datasets**
```bash
python ai_models/datasets/download_data.py
```

2. **Train Certification Predictor**
```bash
python ai_models/certification_predictor.py
```

3. **Train Risk Analyzer**
```bash
python ai_models/risk_analyzer.py
```

4. **Train Course Recommender**
```bash
python ai_models/course_recommender.py
```

### 📊 Generated Outputs

After training, you'll find:

#### **Plots and Visualizations** (`ai_models/plots/`)
- `certification_model_evaluation.png` - Confusion matrix, ROC curves, feature importance
- `certification_chi_square_analysis.png` - Chi-square feature analysis
- `risk_analyzer_evaluation.png` - Risk model performance metrics
- `risk_factor_analysis.png` - Risk factor correlations and patterns
- `course_interaction_analysis.png` - User behavior and course popularity
- `recommendation_evaluation.png` - Recommendation system accuracy

#### **Trained Models** (`ai_models/trained_models/`)
- `certification_predictor.pkl` - Main certification model
- `certification_scaler.pkl` - Feature scaler
- `certification_encoders.pkl` - Label encoders
- `risk_analyzer.pkl` - Risk assessment model
- `risk_scaler.pkl` - Risk feature scaler
- `course_nmf_model.pkl` - Matrix factorization model
- `user_item_matrix.pkl` - User-course interaction matrix

#### **Datasets** (`ai_models/datasets/`)
- `certification_data.csv` - Synthetic certification training data
- `risk_assessment_data.csv` - Risk analysis training data  
- `courses.csv` - Course catalog data
- `course_interactions.csv` - User-course interaction data

### 🧪 Testing Individual Models

#### Test Certification Predictor
```bash
python -c "
import pandas as pd
import joblib
from ai_models.certification_predictor import CertificationPredictor

# Load trained model
predictor = CertificationPredictor()
predictor.model = joblib.load('ai_models/trained_models/certification_predictor.pkl')
predictor.scaler = joblib.load('ai_models/trained_models/certification_scaler.pkl')

# Test prediction
sample_data = {
    'training_hours': 50,
    'assessment_score': 85,
    'compliance_history': 90,
    'continuing_education': 25,
    'years_experience': 5,
    'previous_certifications': 2
}

print('Sample prediction for certification eligibility:')
# Add your prediction logic here
"
```

#### Test Risk Analyzer
```bash
python -c "
import joblib
import numpy as np

# Load risk model
model = joblib.load('ai_models/trained_models/risk_analyzer.pkl')
scaler = joblib.load('ai_models/trained_models/risk_scaler.pkl')

# Test data
test_data = np.array([[15, 75, 2, 3, 5, 85, 5, 2, 0, 0, 0]])  # Sample member data
scaled_data = scaler.transform(test_data)
risk_prob = model.predict_proba(scaled_data)[0][1]

print(f'Risk probability: {risk_prob:.3f}')
print(f'Risk level: {"HIGH" if risk_prob > 0.7 else "MEDIUM" if risk_prob > 0.3 else "LOW"}')
"
```

### 🌐 Running the Web Application

```bash
# Start the development server
npm run dev
```

Navigate to `http://localhost:8080` to access the web interface.

## 📈 Model Performance Metrics

The training scripts automatically generate comprehensive evaluation metrics:

### Certification Predictor
- **Confusion Matrix**: True/False positives and negatives
- **ROC Curve**: Area Under Curve (AUC) analysis  
- **Chi-square Analysis**: Statistical feature importance
- **Classification Report**: Precision, recall, F1-scores
- **Feature Importance**: Most influential factors

### Risk Analyzer  
- **Risk Factor Analysis**: Correlation with performance outcomes
- **Precision-Recall Curves**: Model threshold optimization
- **Risk Score Distribution**: Probability distributions by risk level
- **Intervention Recommendations**: Actionable insights

### Course Recommender
- **RMSE/MAE**: Recommendation accuracy metrics
- **User Interaction Patterns**: Engagement analysis
- **Content Similarity**: Course relationship mapping
- **Collaborative Filtering**: User-based recommendations

## 🔧 Customization

### Adding New Features
1. Extend datasets in `ai_models/datasets/download_data.py`
2. Modify model architectures in respective training scripts
3. Update evaluation metrics as needed

### Using Real Data
Replace the synthetic data generation with your actual datasets:
- Ensure column names match expected features
- Maintain data types and ranges
- Update preprocessing steps if needed

## 📝 API Integration

The trained models can be integrated into the web application:

```python
# Example: Load and use certification predictor
import joblib
import numpy as np

def predict_certification(member_data):
    model = joblib.load('ai_models/trained_models/certification_predictor.pkl')
    scaler = joblib.load('ai_models/trained_models/certification_scaler.pkl')
    
    # Preprocess and predict
    scaled_data = scaler.transform([member_data])
    prediction = model.predict_proba(scaled_data)[0][1]
    
    return {
        'approved': prediction > 0.5,
        'confidence': prediction,
        'recommendation': 'Eligible' if prediction > 0.5 else 'Needs improvement'
    }
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branches for new AI models or improvements
3. Add comprehensive tests for new functionality
4. Update documentation and README
5. Submit pull requests with detailed descriptions

## 📊 Expected Academic Outputs

This system generates all required academic deliverables:

- ✅ **Confusion Matrices** for all classification models
- ✅ **ROC Curves** with AUC calculations
- ✅ **Chi-square Feature Analysis** for statistical significance
- ✅ **Visual Prediction Insights** with comprehensive plots
- ✅ **Model Comparison Studies** across different algorithms
- ✅ **Performance Evaluation Metrics** (Precision, Recall, F1, RMSE, MAE)

## 📚 Technologies Used

**Backend AI/ML:**
- scikit-learn (Machine Learning)
- pandas & numpy (Data Processing)  
- matplotlib & seaborn (Visualizations)
- joblib (Model Persistence)

**Frontend Web Application:**
- React + TypeScript
- Tailwind CSS
- shadcn/ui Components
- Vite Build System

## 🎯 Project Goals Achieved

✅ **Explainable AI Models** with feature importance analysis  
✅ **Testable Prototypes** with comprehensive evaluation metrics  
✅ **Academic Rigor** with statistical tests and visualizations  
✅ **Practical Application** with web interface integration  
✅ **Nigerian Context** tailored for CIEEEN professional development  

## 📞 Support

For questions about the AI models or implementation:
1. Check the generated plots for model insights
2. Review individual training script outputs
3. Examine the confusion matrices and performance metrics
4. Test with the provided sample data

---

**Chartered Institute of Electrical and Electronic Engineering of Nigeria (CIEEEN)**  
*AI-Based Systems for Professional Development and Compliance Monitoring*
