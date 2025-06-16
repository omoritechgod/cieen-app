
"""
Course Recommendation System AI Model
Recommends courses to members based on collaborative and content-based filtering
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import joblib
import os

class CourseRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarity = None
        self.user_similarity = None
        self.nmf_model = None
        self.svd_model = None
        self.course_features = None
        
    def load_data(self, courses_path, interactions_path):
        """Load course and interaction data"""
        print("Loading course recommendation data...")
        
        self.courses = pd.read_csv(courses_path)
        self.interactions = pd.read_csv(interactions_path)
        
        print(f"Courses: {len(self.courses)}")
        print(f"Interactions: {len(self.interactions)}")
        print(f"Unique members: {self.interactions['member_id'].nunique()}")
        print(f"Unique courses: {self.interactions['course_id'].nunique()}")
        
        return self.courses, self.interactions
    
    def create_user_item_matrix(self):
        """Create user-item interaction matrix"""
        print("Creating user-item matrix...")
        
        # Create rating matrix
        self.user_item_matrix = self.interactions.pivot_table(
            index='member_id', 
            columns='course_id', 
            values='rating',
            fill_value=0
        )
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        
        # Calculate sparsity
        sparsity = (self.user_item_matrix == 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])
        print(f"Matrix sparsity: {sparsity:.2%}")
        
        return self.user_item_matrix
    
    def analyze_interaction_patterns(self):
        """Analyze user interaction patterns"""
        print("\nAnalyzing interaction patterns...")
        
        # User statistics
        user_stats = self.interactions.groupby('member_id').agg({
            'rating': ['count', 'mean', 'std'],
            'completed': 'sum',
            'time_spent_hours': 'sum'
        }).round(2)
        
        # Course statistics  
        course_stats = self.interactions.groupby('course_id').agg({
            'rating': ['count', 'mean', 'std'],
            'completed': 'mean',
            'time_spent_hours': 'mean'
        }).round(2)
        
        # Merge with course info
        course_analysis = course_stats.merge(self.courses, left_index=True, right_on='course_id')
        
        # Visualize patterns
        plt.figure(figsize=(15, 10))
        
        # Rating distribution
        plt.subplot(2, 3, 1)
        self.interactions['rating'].hist(bins=5, alpha=0.7)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        
        # Course popularity
        plt.subplot(2, 3, 2)
        course_popularity = self.interactions['course_id'].value_counts().head(10)
        course_popularity.plot(kind='bar')
        plt.title('Top 10 Most Popular Courses')
        plt.xlabel('Course ID')
        plt.ylabel('Number of Interactions')
        plt.xticks(rotation=45)
        
        # Average rating by category
        plt.subplot(2, 3, 3)
        category_ratings = self.interactions.merge(self.courses, on='course_id').groupby('category')['rating'].mean().sort_values(ascending=False)
        category_ratings.plot(kind='bar')
        plt.title('Average Rating by Category')
        plt.xlabel('Course Category')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45)
        
        # Completion rate by difficulty
        plt.subplot(2, 3, 4)
        completion_by_difficulty = self.interactions.merge(self.courses, on='course_id').groupby('difficulty_level')['completed'].mean()
        completion_by_difficulty.plot(kind='bar')
        plt.title('Completion Rate by Difficulty')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Completion Rate')
        
        # User engagement distribution
        plt.subplot(2, 3, 5)
        user_engagement = self.interactions.groupby('member_id')['rating'].count()
        user_engagement.hist(bins=20, alpha=0.7)
        plt.title('User Engagement Distribution')
        plt.xlabel('Number of Course Interactions')
        plt.ylabel('Number of Users')
        
        # Time spent vs rating correlation
        plt.subplot(2, 3, 6)
        plt.scatter(self.interactions['time_spent_hours'], self.interactions['rating'], alpha=0.5)
        plt.xlabel('Time Spent (Hours)')
        plt.ylabel('Rating')
        plt.title('Time Spent vs Rating')
        
        # Add correlation coefficient
        correlation = self.interactions['time_spent_hours'].corr(self.interactions['rating'])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig('ai_models/plots/course_interaction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return user_stats, course_analysis
    
    def build_collaborative_filtering(self):
        """Build collaborative filtering model"""
        print("\nBuilding collaborative filtering models...")
        
        # Item-based collaborative filtering
        item_matrix = self.user_item_matrix.T  # Transpose for item-item similarity
        self.item_similarity = cosine_similarity(item_matrix)
        
        # User-based collaborative filtering  
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        # Matrix factorization with NMF
        self.nmf_model = NMF(n_components=20, random_state=42)
        user_features = self.nmf_model.fit_transform(self.user_item_matrix)
        item_features = self.nmf_model.components_
        
        # SVD for dimensionality reduction
        self.svd_model = TruncatedSVD(n_components=20, random_state=42)
        user_features_svd = self.svd_model.fit_transform(self.user_item_matrix)
        
        print("Collaborative filtering models built successfully!")
        
        return user_features, item_features, user_features_svd
    
    def build_content_based_filtering(self):
        """Build content-based filtering model"""
        print("Building content-based filtering model...")
        
        # Create course feature matrix
        # Combine textual features
        self.courses['combined_features'] = (
            self.courses['category'] + ' ' + 
            self.courses['difficulty_level'] + ' ' + 
            self.courses['duration_hours'].astype(str)
        )
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.courses['combined_features'])
        
        # Calculate content similarity
        content_similarity = cosine_similarity(tfidf_matrix)
        
        print("Content-based filtering model built successfully!")
        
        return content_similarity, tfidf
    
    def evaluate_recommendations(self):
        """Evaluate recommendation system performance"""
        print("\nEvaluating recommendation system...")
        
        # Split data for evaluation
        train_data, test_data = train_test_split(self.interactions, test_size=0.2, random_state=42)
        
        # Create train matrix
        train_matrix = train_data.pivot_table(
            index='member_id', 
            columns='course_id', 
            values='rating',
            fill_value=0
        )
        
        # NMF prediction
        nmf_pred = NMF(n_components=20, random_state=42)
        user_features = nmf_pred.fit_transform(train_matrix)
        item_features = nmf_pred.components_
        predictions = np.dot(user_features, item_features)
        
        # Calculate RMSE for users and items in test set
        test_predictions = []
        test_actuals = []
        
        for _, row in test_data.iterrows():
            user_idx = train_matrix.index.get_loc(row['member_id']) if row['member_id'] in train_matrix.index else None
            item_idx = train_matrix.columns.get_loc(row['course_id']) if row['course_id'] in train_matrix.columns else None
            
            if user_idx is not None and item_idx is not None:
                pred_rating = predictions[user_idx, item_idx]
                test_predictions.append(pred_rating)
                test_actuals.append(row['rating'])
        
        if test_predictions:
            rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
            mae = mean_absolute_error(test_actuals, test_predictions)
            
            print(f"RMSE: {rmse:.3f}")
            print(f"MAE: {mae:.3f}")
            
            # Visualize prediction accuracy
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.scatter(test_actuals, test_predictions, alpha=0.6)
            plt.plot([1, 5], [1, 5], 'r--')
            plt.xlabel('Actual Rating')
            plt.ylabel('Predicted Rating')
            plt.title(f'Prediction Accuracy (RMSE: {rmse:.3f})')
            
            plt.subplot(1, 2, 2)
            residuals = np.array(test_actuals) - np.array(test_predictions)
            plt.hist(residuals, bins=20, alpha=0.7)
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Residual Distribution')
            
            plt.tight_layout()
            plt.savefig('ai_models/plots/recommendation_evaluation.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return rmse, mae
        else:
            print("No valid predictions could be made for evaluation.")
            return None, None
    
    def generate_recommendations(self, user_id, n_recommendations=5):
        """Generate course recommendations for a user"""
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Find similar users
        user_similarities = self.user_similarity[user_idx]
        similar_users = np.argsort(user_similarities)[::-1][1:6]  # Top 5 similar users
        
        # Get recommendations based on similar users
        recommendations = {}
        for similar_user in similar_users:
            similar_user_ratings = self.user_item_matrix.iloc[similar_user]
            for course_id, rating in similar_user_ratings.items():
                if rating > 0 and user_ratings[course_id] == 0:  # User hasn't taken this course
                    if course_id not in recommendations:
                        recommendations[course_id] = []
                    recommendations[course_id].append(rating * user_similarities[similar_user])
        
        # Average the weighted ratings
        final_recommendations = {}
        for course_id, weighted_ratings in recommendations.items():
            final_recommendations[course_id] = np.mean(weighted_ratings)
        
        # Sort and return top N
        sorted_recommendations = sorted(final_recommendations.items(), key=lambda x: x[1], reverse=True)
        top_recommendations = sorted_recommendations[:n_recommendations]
        
        return top_recommendations
    
    def save_models(self):
        """Save all trained models"""
        os.makedirs('ai_models/trained_models', exist_ok=True)
        
        joblib.dump(self.nmf_model, 'ai_models/trained_models/course_nmf_model.pkl')
        joblib.dump(self.svd_model, 'ai_models/trained_models/course_svd_model.pkl')
        joblib.dump(self.user_item_matrix, 'ai_models/trained_models/user_item_matrix.pkl')
        joblib.dump(self.item_similarity, 'ai_models/trained_models/item_similarity.pkl')
        joblib.dump(self.user_similarity, 'ai_models/trained_models/user_similarity.pkl')
        
        print("Course recommendation models saved successfully!")

def main():
    """Main function to run course recommendation training"""
    recommender = CourseRecommender()
    
    # Load data
    courses, interactions = recommender.load_data(
        'ai_models/datasets/courses.csv',
        'ai_models/datasets/course_interactions.csv'
    )
    
    # Create user-item matrix
    user_item_matrix = recommender.create_user_item_matrix()
    
    # Analyze patterns
    user_stats, course_analysis = recommender.analyze_interaction_patterns()
    
    # Build models
    user_features, item_features, user_features_svd = recommender.build_collaborative_filtering()
    content_similarity, tfidf = recommender.build_content_based_filtering()
    
    # Evaluate
    rmse, mae = recommender.evaluate_recommendations()
    
    # Save models
    recommender.save_models()
    
    # Test recommendation
    print("\nTesting recommendation for user 1:")
    recommendations = recommender.generate_recommendations(1, n_recommendations=5)
    for course_id, score in recommendations:
        course_info = courses[courses['course_id'] == course_id].iloc[0]
        print(f"Course: {course_info['course_name']} (Category: {course_info['category']}) - Score: {score:.3f}")
    
    print("\nCourse Recommendation System training completed!")
    print("Check 'ai_models/plots/' for visualization outputs")
    print("Check 'ai_models/trained_models/' for saved model files")

if __name__ == "__main__":
    main()
