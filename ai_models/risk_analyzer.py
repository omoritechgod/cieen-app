
"""
Performance Risk Analyzer AI Model
Identifies members at risk of non-compliance or poor performance
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import joblib
import os

class RiskAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=8)
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the risk assessment data"""
        print("Loading risk assessment data...")
        df = pd.read_csv(data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Risk distribution:\n{df['at_risk'].value_counts()}")
        
        # Create additional risk indicators
        df['engagement_risk'] = (df['engagement_score'] < 50).astype(int)
        df['activity_risk'] = (df['days_since_last_activity'] > 14).astype(int)
        df['learning_risk'] = (df['learning_hours_monthly'] < 8).astype(int)
        
        # Separate features and target
        X = df.drop(['member_id', 'at_risk'], axis=1)
        y = df['at_risk']
        
        return X, y, df
    
    def analyze_risk_factors(self, X, y):
        """Analyze risk factors using statistical tests"""
        print("\nAnalyzing risk factors...")
        
        # Calculate correlation with risk
        correlations = []
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                corr = X[col].corr(y)
                correlations.append({'Feature': col, 'Correlation': abs(corr)})
        
        corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
        print("Feature correlations with risk:")
        print(corr_df.to_string(index=False))
        
        # Chi-square test for categorical relationships
        chi2_results = []
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                # Bin continuous variables
                X_binned = pd.cut(X[col], bins=5, labels=False)
                contingency_table = pd.crosstab(X_binned, y)
                chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
                chi2_results.append({
                    'Feature': col,
                    'Chi2_Statistic': chi2_stat,
                    'P_Value': p_val
                })
        
        chi2_df = pd.DataFrame(chi2_results).sort_values('Chi2_Statistic', ascending=False)
        
        # Visualize risk factor analysis
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 3, 1)
        sns.barplot(data=corr_df.head(8), x='Correlation', y='Feature')
        plt.title('Feature Correlation with Risk')
        
        plt.subplot(2, 3, 2)
        sns.barplot(data=chi2_df.head(8), x='Chi2_Statistic', y='Feature')
        plt.title('Chi-square Statistics for Risk Factors')
        
        # Risk distribution by key factors
        plt.subplot(2, 3, 3)
        risk_by_engagement = X.groupby(pd.cut(X['engagement_score'], bins=5))['engagement_risk'].mean()
        risk_by_engagement.plot(kind='bar')
        plt.title('Risk by Engagement Score')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 4)
        risk_by_activity = X.groupby(pd.cut(X['days_since_last_activity'], bins=5))['activity_risk'].mean()
        risk_by_activity.plot(kind='bar')
        plt.title('Risk by Days Since Last Activity')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 5)
        plt.scatter(X['learning_hours_monthly'], X['engagement_score'], c=y, alpha=0.6)
        plt.xlabel('Learning Hours Monthly')
        plt.ylabel('Engagement Score')
        plt.title('Risk Distribution: Learning vs Engagement')
        plt.colorbar(label='At Risk')
        
        plt.subplot(2, 3, 6)
        # Heatmap of risk factors
        risk_matrix = X[['engagement_score', 'learning_hours_monthly', 'compliance_rating', 
                        'days_since_last_activity']].corr()
        sns.heatmap(risk_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Risk Factor Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('ai_models/plots/risk_factor_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_df, chi2_df
    
    def train_model(self, X, y):
        """Train the risk analysis model"""
        print("\nTraining risk analysis models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Try multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        best_score = 0
        best_model_name = ""
        
        # Compare models
        for name, model in models.items():
            if name in ['SVM']:
                scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='roc_auc')
            else:
                scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_model_name = name
        
        # Train the best model
        self.model = models[best_model_name]
        if best_model_name == 'SVM':
            self.model.fit(X_train_selected, y_train)
            y_pred = self.model.predict(X_test_selected)
            y_pred_proba = self.model.predict_proba(X_test_selected)[:, 1]
        else:
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"\nBest model: {best_model_name}")
        print(f"Test AUC: {roc_curve(y_test, y_pred_proba)[2].max():.4f}")
        
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
        plt.title('Confusion Matrix - Risk Analyzer')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # ROC Curve
        plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Risk Analyzer')
        plt.legend(loc="lower right")
        
        # Feature Importance
        plt.subplot(2, 3, 3)
        if hasattr(self.model, 'feature_importances_'):
            # Get selected features
            selected_features = self.feature_selector.get_support()
            selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features[i]]
            
            importance_df = pd.DataFrame({
                'feature': selected_feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.title('Feature Importance')
            plt.xlabel('Importance Score')
        
        # Risk Score Distribution
        plt.subplot(2, 3, 4)
        plt.hist(y_pred_proba[y_true == 0], alpha=0.5, label='Not At Risk', bins=20, color='green')
        plt.hist(y_pred_proba[y_true == 1], alpha=0.5, label='At Risk', bins=20, color='red')
        plt.xlabel('Risk Probability')
        plt.ylabel('Count')
        plt.title('Risk Score Distribution')
        plt.legend()
        
        # Precision-Recall Curve
        plt.subplot(2, 3, 5)
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        plt.plot(recall, precision, lw=2, label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        # Risk Threshold Analysis
        plt.subplot(2, 3, 6)
        thresholds = np.arange(0.1, 1.0, 0.1)
        precision_scores = []
        recall_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            from sklearn.metrics import precision_score, recall_score
            precision_scores.append(precision_score(y_true, y_pred_thresh))
            recall_scores.append(recall_score(y_true, y_pred_thresh))
        
        plt.plot(thresholds, precision_scores, label='Precision', marker='o')
        plt.plot(thresholds, recall_scores, label='Recall', marker='s')
        plt.xlabel('Risk Threshold')
        plt.ylabel('Score')
        plt.title('Precision/Recall vs Threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('ai_models/plots/risk_analyzer_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred))
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs('ai_models/trained_models', exist_ok=True)
        joblib.dump(self.model, 'ai_models/trained_models/risk_analyzer.pkl')
        joblib.dump(self.scaler, 'ai_models/trained_models/risk_scaler.pkl')
        joblib.dump(self.feature_selector, 'ai_models/trained_models/risk_feature_selector.pkl')
        print("Risk analyzer model saved successfully!")

def main():
    """Main function to run risk analysis training"""
    analyzer = RiskAnalyzer()
    
    # Load and preprocess data
    X, y, df = analyzer.load_and_preprocess_data('ai_models/datasets/risk_assessment_data.csv')
    
    # Analyze risk factors
    corr_results, chi2_results = analyzer.analyze_risk_factors(X, y)
    
    # Train model
    X_test, y_test, y_pred, y_pred_proba = analyzer.train_model(X, y)
    
    # Save model
    analyzer.save_model()
    
    print("\nRisk Analyzer training completed!")
    print("Check 'ai_models/plots/' for visualization outputs")
    print("Check 'ai_models/trained_models/' for saved model files")

if __name__ == "__main__":
    main()
