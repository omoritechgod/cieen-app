
"""
Script to download and prepare datasets for CIEEEN AI models
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import os

def create_certification_dataset():
    """Create synthetic certification dataset based on real patterns"""
    np.random.seed(42)
    
    # Generate base features
    n_samples = 2000
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.3, 0.7],  # Imbalanced classes
        flip_y=0.05,
        random_state=42
    )
    
    # Create meaningful feature names and data
    features = {
        'member_id': range(1, n_samples + 1),
        'training_hours': np.random.exponential(30, n_samples) + 10,
        'assessment_score': np.random.beta(2, 1, n_samples) * 100,
        'compliance_history': np.random.gamma(2, 20, n_samples),
        'continuing_education': np.random.poisson(15, n_samples),
        'years_experience': np.random.exponential(5, n_samples) + 1,
        'previous_certifications': np.random.poisson(2, n_samples),
        'course_completion_rate': np.random.beta(3, 1, n_samples) * 100,
        'engagement_score': np.random.normal(70, 15, n_samples),
        'specialization_area': np.random.choice(['Power', 'Electronics', 'Telecommunications', 'Control'], n_samples),
        'age_group': np.random.choice(['25-35', '36-45', '46-55', '56+'], n_samples),
        'certification_approved': y
    }
    
    df = pd.DataFrame(features)
    
    # Ensure realistic bounds
    df['assessment_score'] = df['assessment_score'].clip(0, 100)
    df['compliance_history'] = df['compliance_history'].clip(0, 100)
    df['course_completion_rate'] = df['course_completion_rate'].clip(0, 100)
    df['engagement_score'] = df['engagement_score'].clip(0, 100)
    df['training_hours'] = df['training_hours'].clip(5, 200)
    df['years_experience'] = df['years_experience'].clip(1, 40)
    
    return df

def create_risk_assessment_dataset():
    """Create synthetic risk assessment dataset"""
    np.random.seed(123)
    
    n_samples = 1800
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.7, 0.3],  # 30% at risk
        flip_y=0.03,
        random_state=123
    )
    
    features = {
        'member_id': range(1, n_samples + 1),
        'learning_hours_monthly': np.random.exponential(12, n_samples) + 2,
        'engagement_score': np.random.beta(2, 2, n_samples) * 100,
        'assessment_attempts': np.random.poisson(2, n_samples) + 1,
        'courses_completed': np.random.poisson(3, n_samples),
        'days_since_last_activity': np.random.exponential(7, n_samples),
        'compliance_rating': np.random.beta(3, 1, n_samples) * 100,
        'forum_participation': np.random.poisson(5, n_samples),
        'mentor_sessions': np.random.poisson(2, n_samples),
        'at_risk': y  # 1 = at risk, 0 = not at risk
    }
    
    df = pd.DataFrame(features)
    
    # Ensure realistic bounds
    df['engagement_score'] = df['engagement_score'].clip(0, 100)
    df['compliance_rating'] = df['compliance_rating'].clip(0, 100)
    df['learning_hours_monthly'] = df['learning_hours_monthly'].clip(1, 80)
    df['days_since_last_activity'] = df['days_since_last_activity'].clip(0, 90)
    df['assessment_attempts'] = df['assessment_attempts'].clip(1, 10)
    
    return df

def create_course_recommendation_dataset():
    """Create course interaction dataset for recommendation system"""
    np.random.seed(456)
    
    n_members = 500
    n_courses = 100
    n_interactions = 5000
    
    # Create courses
    course_categories = ['Power Systems', 'Electronics', 'Control Systems', 
                        'Telecommunications', 'Renewable Energy', 'AI/ML', 
                        'Project Management', 'Safety Standards']
    
    courses = pd.DataFrame({
        'course_id': range(1, n_courses + 1),
        'course_name': [f"Course_{i}" for i in range(1, n_courses + 1)],
        'category': np.random.choice(course_categories, n_courses),
        'difficulty_level': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], n_courses),
        'duration_hours': np.random.choice([10, 20, 30, 40, 50], n_courses),
        'rating': np.random.uniform(3.5, 5.0, n_courses)
    })
    
    # Create member-course interactions
    interactions = []
    for _ in range(n_interactions):
        member_id = np.random.randint(1, n_members + 1)
        course_id = np.random.randint(1, n_courses + 1)
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.4, 0.25])
        completed = np.random.choice([0, 1], p=[0.3, 0.7])
        
        interactions.append({
            'member_id': member_id,
            'course_id': course_id,
            'rating': rating,
            'completed': completed,
            'time_spent_hours': np.random.exponential(10) if completed else np.random.exponential(3)
        })
    
    interactions_df = pd.DataFrame(interactions).drop_duplicates(['member_id', 'course_id'])
    
    return courses, interactions_df

def main():
    """Download and save all datasets"""
    os.makedirs('ai_models/datasets', exist_ok=True)
    
    print("Creating certification dataset...")
    cert_data = create_certification_dataset()
    cert_data.to_csv('ai_models/datasets/certification_data.csv', index=False)
    print(f"Certification dataset created with {len(cert_data)} records")
    
    print("Creating risk assessment dataset...")
    risk_data = create_risk_assessment_dataset()
    risk_data.to_csv('ai_models/datasets/risk_assessment_data.csv', index=False)
    print(f"Risk assessment dataset created with {len(risk_data)} records")
    
    print("Creating course recommendation datasets...")
    courses, interactions = create_course_recommendation_dataset()
    courses.to_csv('ai_models/datasets/courses.csv', index=False)
    interactions.to_csv('ai_models/datasets/course_interactions.csv', index=False)
    print(f"Course recommendation datasets created: {len(courses)} courses, {len(interactions)} interactions")
    
    print("\nAll datasets created successfully!")
    print("Run the training scripts to build the AI models.")

if __name__ == "__main__":
    main()
