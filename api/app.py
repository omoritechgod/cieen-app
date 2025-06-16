
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store loaded models
certification_model = None
certification_scaler = None
certification_encoders = None
risk_model = None
risk_scaler = None
recommendation_models = {}

def load_models():
    """Load all trained models at startup"""
    global certification_model, certification_scaler, certification_encoders
    global risk_model, risk_scaler, recommendation_models
    
    try:
        # Load certification models
        if os.path.exists('../ai_models/trained_models/certification_predictor.pkl'):
            certification_model = joblib.load('../ai_models/trained_models/certification_predictor.pkl')
            certification_scaler = joblib.load('../ai_models/trained_models/certification_scaler.pkl')
            certification_encoders = joblib.load('../ai_models/trained_models/certification_encoders.pkl')
            logger.info("Certification models loaded successfully")
        
        # Load risk analysis models
        if os.path.exists('../ai_models/trained_models/risk_predictor.pkl'):
            risk_model = joblib.load('../ai_models/trained_models/risk_predictor.pkl')
            risk_scaler = joblib.load('../ai_models/trained_models/risk_scaler.pkl')
            logger.info("Risk analysis models loaded successfully")
        
        # Load recommendation models
        if os.path.exists('../ai_models/trained_models/user_item_matrix.pkl'):
            recommendation_models['user_item_matrix'] = joblib.load('../ai_models/trained_models/user_item_matrix.pkl')
            recommendation_models['item_similarity'] = joblib.load('../ai_models/trained_models/item_similarity.pkl')
            recommendation_models['nmf_model'] = joblib.load('../ai_models/trained_models/course_nmf_model.pkl')
            logger.info("Recommendation models loaded successfully")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'certification': certification_model is not None,
            'risk_analysis': risk_model is not None,
            'recommendations': len(recommendation_models) > 0
        }
    })

@app.route('/api/certification/predict', methods=['POST'])
def predict_certification():
    """Predict certification eligibility"""
    try:
        data = request.json
        
        # Extract features from request
        features = {
            'training_hours': float(data.get('trainingHours', 0)),
            'assessment_score': float(data.get('assessmentScore', 0)),
            'compliance_history': float(data.get('complianceHistory', 0)),
            'continuing_education_hours': float(data.get('continuingEducation', 0)),
            'years_experience': float(data.get('yearsExperience', 0)),
            'previous_certifications': float(data.get('previousCertifications', 0))
        }
        
        # Create DataFrame for prediction
        X = pd.DataFrame([features])
        
        if certification_model is not None:
            # Use trained model
            if hasattr(certification_model, 'predict_proba'):
                prediction_proba = certification_model.predict_proba(X)[0]
                prediction = certification_model.predict(X)[0]
                confidence = max(prediction_proba) * 100
            else:
                prediction = certification_model.predict(X)[0]
                confidence = 85.0  # Default confidence
            
            approved = bool(prediction)
        else:
            # Fallback logic when model not available
            score = 0
            if features['training_hours'] >= 40: score += 25
            if features['assessment_score'] >= 70: score += 30
            if features['compliance_history'] >= 80: score += 20
            if features['continuing_education_hours'] >= 20: score += 15
            if features['years_experience'] >= 3: score += 10
            
            approved = score >= 70
            confidence = min(score, 100)
        
        # Generate factors based on input
        factors = []
        if features['training_hours'] >= 40:
            factors.append("Sufficient training hours completed")
        else:
            factors.append("Need more training hours")
            
        if features['assessment_score'] >= 70:
            factors.append("Strong assessment performance")
        else:
            factors.append("Assessment score needs improvement")
            
        if features['compliance_history'] >= 80:
            factors.append("Excellent compliance record")
        else:
            factors.append("Compliance history needs attention")
        
        return jsonify({
            'approved': approved,
            'confidence': round(confidence, 1),
            'factors': factors,
            'model_used': 'trained' if certification_model else 'fallback'
        })
        
    except Exception as e:
        logger.error(f"Error in certification prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk/analyze', methods=['POST'])
def analyze_risk():
    """Analyze performance risk"""
    try:
        data = request.json
        
        # Extract features from request
        features = {
            'learning_hours': float(data.get('learningHours', 0)),
            'engagement_score': float(data.get('engagementScore', 0)),
            'assessment_attempts': float(data.get('assessmentAttempts', 0)),
            'courses_completed': float(data.get('coursesCompleted', 0)),
            'last_activity_days': float(data.get('lastActivityDays', 0)),
            'compliance_rating': float(data.get('complianceRating', 0))
        }
        
        # Create DataFrame for prediction
        X = pd.DataFrame([features])
        
        if risk_model is not None:
            # Use trained model
            risk_prediction = risk_model.predict(X)[0]
            if hasattr(risk_model, 'predict_proba'):
                risk_proba = risk_model.predict_proba(X)[0]
                risk_score = max(risk_proba) * 100
            else:
                risk_score = 75.0  # Default score
        else:
            # Fallback logic
            risk_score = 0
            if features['learning_hours'] < 10: risk_score += 25
            if features['engagement_score'] < 50: risk_score += 20
            if features['assessment_attempts'] > 3: risk_score += 15
            if features['courses_completed'] < 2: risk_score += 20
            if features['last_activity_days'] > 14: risk_score += 15
            if features['compliance_rating'] < 70: risk_score += 20
            
            risk_score = min(risk_score, 100)
        
        # Determine risk level
        if risk_score <= 30:
            risk_level = "LOW"
        elif risk_score <= 60:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Generate factors and interventions
        factors = []
        interventions = []
        
        if features['learning_hours'] < 10:
            factors.append("Low learning hours per month")
            interventions.append("Recommend increased learning schedule")
        
        if features['engagement_score'] < 50:
            factors.append("Poor engagement with learning materials")
            interventions.append("Personalized content recommendations")
        
        if features['assessment_attempts'] > 3:
            factors.append("Multiple assessment attempts indicating struggle")
            interventions.append("Additional tutoring or remedial courses")
        
        if features['courses_completed'] < 2:
            factors.append("Low course completion rate")
            interventions.append("Course difficulty adjustment or extended timelines")
        
        if features['last_activity_days'] > 14:
            factors.append("Inactive for extended period")
            interventions.append("Re-engagement campaign and motivation strategies")
        
        if features['compliance_rating'] < 70:
            factors.append("Below-average compliance rating")
            interventions.append("Compliance training and monitoring")
        
        if not factors:
            factors.append("All indicators within normal range")
            interventions.append("Continue current learning path")
        
        return jsonify({
            'riskLevel': risk_level,
            'riskScore': round(risk_score, 1),
            'factors': factors,
            'interventions': interventions,
            'model_used': 'trained' if risk_model else 'fallback'
        })
        
    except Exception as e:
        logger.error(f"Error in risk analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations/courses', methods=['POST'])
def recommend_courses():
    """Get course recommendations"""
    try:
        data = request.json
        
        # For demo purposes, return mock recommendations
        # In real implementation, this would use the trained recommendation models
        member_profile = {
            'specialization': data.get('specialization', 'General'),
            'experience_level': data.get('experienceLevel', 'Intermediate'),
            'learning_goals': data.get('learningGoals', [])
        }
        
        # Mock course recommendations based on profile
        mock_courses = [
            {
                'id': 1,
                'title': 'Advanced Power Systems Analysis',
                'category': 'Power Engineering',
                'difficulty': 'Advanced',
                'duration': 40,
                'rating': 4.8,
                'match_score': 95.2,
                'description': 'Comprehensive course on modern power system analysis techniques'
            },
            {
                'id': 2,
                'title': 'Digital Signal Processing Fundamentals',
                'category': 'Electronics',
                'difficulty': 'Intermediate',
                'duration': 30,
                'rating': 4.6,
                'match_score': 87.3,
                'description': 'Essential DSP concepts for electrical engineers'
            },
            {
                'id': 3,
                'title': 'Renewable Energy Systems Design',
                'category': 'Renewable Energy',
                'difficulty': 'Intermediate',
                'duration': 35,
                'rating': 4.7,
                'match_score': 82.1,
                'description': 'Design and implementation of renewable energy systems'
            },
            {
                'id': 4,
                'title': 'Control Systems Engineering',
                'category': 'Control Systems',
                'difficulty': 'Advanced',
                'duration': 45,
                'rating': 4.9,
                'match_score': 78.5,
                'description': 'Modern control theory and applications'
            },
            {
                'id': 5,
                'title': 'Professional Ethics in Engineering',
                'category': 'Professional Development',
                'difficulty': 'Beginner',
                'duration': 15,
                'rating': 4.4,
                'match_score': 75.0,
                'description': 'Ethical considerations in engineering practice'
            }
        ]
        
        return jsonify({
            'recommendations': mock_courses,
            'member_profile': member_profile,
            'total_matches': len(mock_courses),
            'model_used': 'mock'  # Would be 'trained' when using actual models
        })
        
    except Exception as e:
        logger.error(f"Error in course recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
