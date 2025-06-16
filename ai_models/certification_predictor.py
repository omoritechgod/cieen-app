
"""
Certification Eligibility Predictor AI Model
Predicts whether a member is eligible for certification based on their profile
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import chi2, SelectKBest
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import joblib
import os

class CertificationPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the certification data"""
        print("Loading certification data...")
        df = pd.read_csv(data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Target distribution:\n{df['certification_approved'].value_counts()}")
        
        # Handle categorical variables
        categorical_cols = ['specialization_area', 'age_group']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Separate features and target
        X = df.drop(['member_id', 'certification_approved'], axis=1)
        y = df['certification_approved']
        
        return X, y, df
    
    def perform_chi_square_test(self, X, y):
        """Perform chi-square test for feature selection"""
        print("\nPerforming Chi-square test for feature importance...")
        
        # For continuous variables, bin them for chi-square test
        X_binned = X.copy()
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X_binned[col] = pd.cut(X[col], bins=5, labels=False)
        
        chi2_stats = []
        p_values = []
        
        for col in X_binned.columns:
            contingency_table = pd.crosstab(X_binned[col], y)
            chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
            chi2_stats.append(chi2_stat)
            p_values.append(p_val)
        
        # Create results DataFrame
        chi2_results = pd.DataFrame({
            'Feature': X.columns,
            'Chi2_Statistic': chi2_stats,
            'P_Value': p_values
        }).sort_values('Chi2_Statistic', ascending=False)
        
        print("Chi-square test results:")
        print(chi2_results.to_string(index=False))
        
        # Visualize chi-square results
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.barplot(data=chi2_results.head(10), x='Chi2_Statistic', y='Feature')
        plt.title('Top 10 Features by Chi-square Statistic')
        
        plt.subplot(1, 2, 2)
        plt.scatter(chi2_results['Chi2_Statistic'], -np.log10(chi2_results['P_Value']))
        plt.xlabel('Chi-square Statistic')
        plt.ylabel('-log10(P-value)')
        plt.title('Chi-square Statistics vs P-values')
        for i, txt in enumerate(chi2_results['Feature']):
            if chi2_results.iloc[i]['P_Value'] < 0.05:
                plt.annotate(txt, (chi2_results.iloc[i]['Chi2_Statistic'], 
                                 -np.log10(chi2_results.iloc[i]['P_Value'])))
        
        plt.tight_layout()
        plt.savefig('ai_models/plots/certification_chi_square_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return chi2_results
    
    def train_model(self, X, y):
        """Train the certification prediction model"""
        print("\nTraining certification prediction models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_score = 0
        best_model_name = ""
        
        # Compare models
        for name, model in models.items():
            if name == 'Logistic Regression':
                scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            else:
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_model_name = name
        
        # Train the best model
        self.model = models[best_model_name]
        if best_model_name == 'Logistic Regression':
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        else:
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print(f"\nBest model: {best_model_name}")
        print(f"Test accuracy: {(y_pred == y_test).mean():.4f}")
        
        # Generate evaluation plots
        self.generate_evaluation_plots(y_test, y_pred, y_pred_proba, X.columns)
        
        return X_test, y_test, y_pred, y_pred_proba
    
    def generate_evaluation_plots(self, y_true, y_pred, y_pred_proba, feature_names):
        """Generate comprehensive evaluation plots"""
        os.makedirs('ai_models/plots', exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # Confusion Matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Certification Predictor')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # ROC Curve
        plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Certification Predictor')
        plt.legend(loc="lower right")
        
        # Feature Importance (if available)
        plt.subplot(2, 3, 3)
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.title('Feature Importance')
            plt.xlabel('Importance Score')
        
        # Prediction Distribution
        plt.subplot(2, 3, 4)
        plt.hist(y_pred_proba[y_true == 0], alpha=0.5, label='Not Approved', bins=20)
        plt.hist(y_pred_proba[y_true == 1], alpha=0.5, label='Approved', bins=20)
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        
        # Classification Report Heatmap
        plt.subplot(2, 3, 5)
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).iloc[:-1, :].T
        sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='Blues')
        plt.title('Classification Report')
        
        # Precision-Recall Curve
        plt.subplot(2, 3, 6)
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        plt.plot(recall, precision, lw=2, label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('ai_models/plots/certification_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred))
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs('ai_models/trained_models', exist_ok=True)
        joblib.dump(self.model, 'ai_models/trained_models/certification_predictor.pkl')
        joblib.dump(self.scaler, 'ai_models/trained_models/certification_scaler.pkl')
        joblib.dump(self.label_encoders, 'ai_models/trained_models/certification_encoders.pkl')
        print("Certification prediction model saved successfully!")

def main():
    """Main function to run certification prediction training"""
    predictor = CertificationPredictor()
    
    # Load and preprocess data
    X, y, df = predictor.load_and_preprocess_data('ai_models/datasets/certification_data.csv')
    
    # Perform chi-square analysis
    chi2_results = predictor.perform_chi_square_test(X, y)
    
    # Train model
    X_test, y_test, y_pred, y_pred_proba = predictor.train_model(X, y)
    
    # Save model
    predictor.save_model()
    
    print("\nCertification Predictor training completed!")
    print("Check 'ai_models/plots/' for visualization outputs")
    print("Check 'ai_models/trained_models/' for saved model files")

if __name__ == "__main__":
    main()
