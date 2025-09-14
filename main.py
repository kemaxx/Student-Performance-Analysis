import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import sqlite3
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StudentPerformanceAnalyzer:
    def __init__(self):
        self.df = None
        self.db_connection = None
        self.setup_database()
    
    def setup_database(self):
        """Initialize SQLite database"""
        self.db_connection = sqlite3.connect('student_performance.db')
        
    def generate_sample_data(self, n_students=1000):
        """Generate realistic sample student data"""
        np.random.seed(42)
        
        # Student demographics
        student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, n_students + 1)]
        names = [f"Student_{i}" for i in range(1, n_students + 1)]
        ages = np.random.normal(20, 2, n_students).clip(17, 25).astype(int)
        genders = np.random.choice(['Male', 'Female'], n_students)
        locations = np.random.choice(['Urban', 'Rural', 'Suburban'], n_students, p=[0.4, 0.3, 0.3])
        
        # Socioeconomic factors
        parent_education = np.random.choice(['Primary', 'Secondary', 'NCE/OND', 'HND/BSc', 'Postgraduate'],n_students, p=[0.2, 0.35, 0.25, 0.15, 0.05])
        family_income = np.random.choice(['Low', 'Middle', 'High'], n_students, p=[0.3, 0.5, 0.2])
        
        # Academic factors
        study_hours = np.random.normal(6, 2, n_students).clip(1, 12)
        attendance_rate = np.random.normal(85, 15, n_students).clip(40, 100)

        ## Corrected location effects - more realistic for Nigerian context
        location_effects = np.zeros(n_students)
        for i, loc in enumerate(locations):
            if loc == 'Urban':
                location_effects[i] = np.random.normal(5, 2)  # Urban advantage
            elif loc == 'Rural':
                location_effects[i] = np.random.normal(-4, 2)  # Rural disadvantage
            else:  # Suburban
                location_effects[i] = np.random.normal(1, 2)   # Slight suburban advantage
        
        # Subject grades (influenced by various factors)
        base_performance = np.random.normal(70, 15, n_students)
        
        # Adjust performance based on factors
        performance_adjustment = np.zeros(n_students)
        performance_adjustment += (study_hours - 6) * 2  # Study hours effect
        performance_adjustment += (attendance_rate - 85) * 0.3  # Attendance effect
        performance_adjustment += location_effects ## Apply location effects
        # Parent education effects
        for i, edu in enumerate(parent_education):
            if edu == 'Postgraduate':
                performance_adjustment[i] += 10
            elif edu == 'HND/BSc':
                performance_adjustment[i] += 6
            elif edu == 'NCE/OND':
                performance_adjustment[i] += 2
            elif edu == 'Secondary':
                performance_adjustment[i] += 0
            elif edu == 'Primary':
                performance_adjustment[i] += -1

        performance_adjustment += np.where(family_income == 'High', 5, 0)
        performance_adjustment += np.where(family_income == 'Middle', 2, 0)
        
        adjusted_performance = (base_performance + performance_adjustment).clip(0, 100)

        # Corrected subject-specific generation with realistic difficulty and student aptitudes
        # Define subject difficulty (relative to average)
        subject_difficulty = {
            'Mathematics': -6,        # Harder
            'Physics': -4,           # Harder  
            'Computer Science': -2,   # Slightly harder
            'English': -3,           # Moderate (ESL factor)
            'Statistics': 0          # Average difficulty
        }
        

        # Generate student aptitude patterns
        science_aptitude = np.random.normal(0, 6, n_students)
        language_aptitude = np.random.normal(0, 5, n_students)
        analytical_aptitude = np.random.normal(0, 7, n_students)


        # Generate subject-specific grades with some correlation
        subjects = ['Mathematics', 'English', 'Computer Science', 'Physics', 'Statistics','Biology','Chemistry']
        subject_data = {}
        
        # for subject in subjects:
        #     # Add subject-specific variation
        #     subject_variation = np.random.normal(0, 5, n_students)
        #     subject_grades = (adjusted_performance + subject_variation).clip(0, 100)
        #     subject_data[f'{subject}_Grade'] = subject_grades.round(1)

       

        for subject in subjects:
            # Start with adjusted performance
            base_subject_score = adjusted_performance.copy()
            
            # Apply subject difficulty
            difficulty_adj = subject_difficulty.get(subject, 0)
            base_subject_score += difficulty_adj
            
            # Apply aptitude bonuses based on subject type
            if subject in ['Mathematics', 'Physics','Chemistry','Biology']:
                base_subject_score += science_aptitude * 0.7  # Strong science correlation
            elif subject == 'English':
                base_subject_score += language_aptitude * 0.8  # Language skills matter most
            elif subject in ['Computer Science', 'Statistics']:
                base_subject_score += analytical_aptitude * 0.6  # Analytical thinking

            # Add location-specific subject effects
            for i, loc in enumerate(locations):
                if subject in ['Computer Science', 'Physics'] and loc == 'Rural':
                    base_subject_score[i] -= 3  # Rural schools may lack resources for these subjects
                elif subject == 'English' and loc == 'Urban':
                    base_subject_score[i] += 2  # Better English exposure in urban areas

            # Add small random variation (reduced from 5 to 3 since we have systematic factors now)
            subject_variation = np.random.normal(0, 3, n_students)
            
            # Calculate final subject grades
            subject_grades = (base_subject_score + subject_variation).clip(0, 100)
            subject_data[f'{subject}_Grade'] = subject_grades.round(1)

        
        # Calculate overall GPA
        grade_columns = [f'{subject}_Grade' for subject in subjects]
        overall_gpa = np.mean([subject_data[col] for col in grade_columns], axis=0)
        
         
        # Determine at-risk status (GPA < 60 or attendance < 70%)
        at_risk = ((overall_gpa < 60) | (attendance_rate < 70)).astype(int)
        
        # Extracurricular activities
        extracurricular = np.random.choice(['Sports', 'Arts', 'Science Club', 'None'], n_students, p=[0.3, 0.2, 0.2, 0.3])
        
        # Create DataFrame
        data = {
            'Student_ID': student_ids,
            'Name': names,
            'Age': ages,
            'Gender': genders,
            'Location': locations,
            'Parent_Education': parent_education,
            'Family_Income': family_income,
            'Study_Hours_Weekly': study_hours.round(1),
            'Attendance_Rate': attendance_rate.round(1),
            'Extracurricular': extracurricular,
            'Overall_GPA': overall_gpa.round(2),
            'At_Risk': at_risk
        }
        
        # Add subject grades
        data.update(subject_data)
        
        self.df = pd.DataFrame(data)
        return self.df
    
    def save_to_database(self):
        """Save data to SQLite database"""
        if self.df is not None:
            self.df.to_sql('students', self.db_connection, if_exists='replace', index=False)
            print("Data saved to database successfully!")
    
    def load_from_database(self):
        """Load data from SQLite database"""
        try:
            self.df = pd.read_sql_query("SELECT * FROM students", self.db_connection)
            return self.df
        except:
            print("No existing data found. Generating new sample data...")
            return self.generate_sample_data()
    
    def basic_statistics(self):
        """Generate basic statistical summary"""
        if self.df is None:
            print("No data available. Please load or generate data first.")
            return
        
        print("=== STUDENT PERFORMANCE ANALYSIS SUMMARY ===\n")
        print(f"Total Students: {len(self.df)}")
        print(f"Average GPA: {self.df['Overall_GPA'].mean():.2f}")
        print(f"Students at Risk: {self.df['At_Risk'].sum()} ({(self.df['At_Risk'].sum()/len(self.df)*100):.1f}%)")
        print(f"Average Attendance: {self.df['Attendance_Rate'].mean():.1f}%")
        print(f"Average Study Hours: {self.df['Study_Hours_Weekly'].mean():.1f} hours/week")
        
        print("\n=== GRADE DISTRIBUTION BY SUBJECT ===")
        subjects = ['Mathematics', 'English', 'Computer Science', 'Physics', 'Statistics']
        for subject in subjects:
            avg_grade = self.df[f'{subject}_Grade'].mean()
            print(f"{subject}: {avg_grade:.1f}")
    
    def perform_correlation_analysis(self):
        """Analyze correlations between different factors"""
        numeric_columns = ['Age', 'Study_Hours_Weekly', 'Attendance_Rate', 'Overall_GPA',
                          'Mathematics_Grade', 'English_Grade', 'Computer Science_Grade', 
                          'Physics_Grade', 'Statistics_Grade']
        
        correlation_matrix = self.df[numeric_columns].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix - Student Performance Factors')
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def analyze_performance_by_demographics(self):
        """Analyze performance across different demographic groups"""
        print("=== PERFORMANCE BY DEMOGRAPHICS ===\n")
        
        # Performance by Gender
        gender_performance = self.df.groupby('Gender')['Overall_GPA'].agg(['mean', 'std', 'count'])
        print("Performance by Gender:")
        print(gender_performance)
        print()
        
        # Performance by Location
        location_performance = self.df.groupby('Location')['Overall_GPA'].agg(['mean', 'std', 'count'])
        print("Performance by Location:")
        print(location_performance)
        print()
        
        # Performance by Parent Education
        education_performance = self.df.groupby('Parent_Education')['Overall_GPA'].agg(['mean', 'std', 'count'])
        print("Performance by Parent Education:")
        print(education_performance)
        print()
        
        # Performance by Family Income
        income_performance = self.df.groupby('Family_Income')['Overall_GPA'].agg(['mean', 'std', 'count'])
        print("Performance by Family Income:")
        print(income_performance)
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Student Performance Analysis Dashboard', fontsize=16, y=0.95)
        
        # 1. GPA Distribution
        axes[0,0].hist(self.df['Overall_GPA'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0,0].axvline(self.df['Overall_GPA'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.df["Overall_GPA"].mean():.1f}')
        axes[0,0].set_title('GPA Distribution')
        axes[0,0].set_xlabel('GPA')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        
        # 2. Attendance vs GPA
        axes[0,1].scatter(self.df['Attendance_Rate'], self.df['Overall_GPA'], alpha=0.6, color='green')
        axes[0,1].set_title('Attendance Rate vs GPA')
        axes[0,1].set_xlabel('Attendance Rate (%)')
        axes[0,1].set_ylabel('Overall GPA')
        
        # 3. Study Hours vs GPA
        axes[0,2].scatter(self.df['Study_Hours_Weekly'], self.df['Overall_GPA'], alpha=0.6, color='orange')
        axes[0,2].set_title('Study Hours vs GPA')
        axes[0,2].set_xlabel('Study Hours per Week')
        axes[0,2].set_ylabel('Overall GPA')
        
        # 4. Performance by Gender
        gender_data = self.df.groupby('Gender')['Overall_GPA'].mean()
        axes[1,0].bar(gender_data.index, gender_data.values, color=['lightcoral', 'lightblue'])
        axes[1,0].set_title('Average GPA by Gender')
        axes[1,0].set_ylabel('Average GPA')
        
        # 5. At-Risk Students by Location
        risk_by_location = self.df.groupby('Location')['At_Risk'].mean() * 100
        axes[1,1].bar(risk_by_location.index, risk_by_location.values, color='red', alpha=0.7)
        axes[1,1].set_title('At-Risk Students by Location (%)')
        axes[1,1].set_ylabel('Percentage At-Risk')
        
        # 6. Subject Performance Comparison
        subjects = ['Mathematics_Grade', 'English_Grade', 'Computer Science_Grade', 'Physics_Grade', 'Statistics_Grade']
        subject_means = [self.df[subject].mean() for subject in subjects]
        subject_names = [s.replace('_Grade', '') for s in subjects]
        axes[1,2].bar(subject_names, subject_means, color='purple', alpha=0.7)
        axes[1,2].set_title('Average Performance by Subject')
        axes[1,2].set_ylabel('Average Grade')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def build_prediction_models(self):
        """Build machine learning models for prediction"""
        print("=== BUILDING PREDICTION MODELS ===\n")
        
        # Prepare features for modeling
        feature_columns = ['Age', 'Study_Hours_Weekly', 'Attendance_Rate']
        
        # Encode categorical variables
        gender_encoded = pd.get_dummies(self.df['Gender'], prefix='Gender')
        location_encoded = pd.get_dummies(self.df['Location'], prefix='Location')
        parent_ed_encoded = pd.get_dummies(self.df['Parent_Education'], prefix='ParentEd')
        income_encoded = pd.get_dummies(self.df['Family_Income'], prefix='Income')
        
        # Combine features
        X = pd.concat([
            self.df[feature_columns],
            gender_encoded,
            location_encoded,
            parent_ed_encoded,
            income_encoded
        ], axis=1)
        
        # Model 1: At-Risk Student Classification
        y_risk = self.df['At_Risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
        
        # Random Forest Classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)
        risk_predictions = rf_classifier.predict(X_test)
        risk_accuracy = accuracy_score(y_test, risk_predictions)
        
        print(f"At-Risk Student Prediction Accuracy: {risk_accuracy:.3f}")
        print("\nClassification Report for At-Risk Prediction:")
        print(classification_report(y_test, risk_predictions))
        
        # Model 2: GPA Prediction
        y_gpa = self.df['Overall_GPA']
        X_train_gpa, X_test_gpa, y_train_gpa, y_test_gpa = train_test_split(X, y_gpa, test_size=0.2, random_state=42)
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_gpa, y_train_gpa)
        gpa_predictions = lr_model.predict(X_test_gpa)
        gpa_mse = mean_squared_error(y_test_gpa, gpa_predictions)
        gpa_rmse = np.sqrt(gpa_mse)
        
        print(f"\nGPA Prediction RMSE: {gpa_rmse:.3f}")
        
        # Feature Importance for At-Risk Prediction
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_classifier.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features for At-Risk Prediction:")
        print(feature_importance.head(10))
        
        return rf_classifier, lr_model, feature_importance
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n" + "="*60)
        print("STUDENT PERFORMANCE ANALYSIS - INSIGHTS REPORT")
        print("="*60)
        
        # Key findings
        avg_gpa = self.df['Overall_GPA'].mean()
        at_risk_pct = (self.df['At_Risk'].sum() / len(self.df)) * 100
        high_performers = len(self.df[self.df['Overall_GPA'] >= 80])
        low_attendance_risk = len(self.df[(self.df['Attendance_Rate'] < 70) & (self.df['At_Risk'] == 1)])
        
        print(f"\n📊 KEY METRICS:")
        print(f"   • Average GPA: {avg_gpa:.2f}")
        print(f"   • Students at risk: {self.df['At_Risk'].sum()} ({at_risk_pct:.1f}%)")
        print(f"   • High performers (GPA ≥ 80): {high_performers}")
        print(f"   • At-risk due to low attendance: {low_attendance_risk}")
        
        print(f"\n🎯 KEY INSIGHTS:")
        
        # Correlation insights
        corr_attendance_gpa = self.df['Attendance_Rate'].corr(self.df['Overall_GPA'])
        corr_study_gpa = self.df['Study_Hours_Weekly'].corr(self.df['Overall_GPA'])
        
        print(f"   • Attendance has a {corr_attendance_gpa:.3f} correlation with GPA")
        print(f"   • Study hours have a {corr_study_gpa:.3f} correlation with GPA")
        
        # Demographics insights
        best_location = self.df.groupby('Location')['Overall_GPA'].mean().idxmax()
        best_parent_ed = self.df.groupby('Parent_Education')['Overall_GPA'].mean().idxmax()
        
        print(f"   • Students from {best_location} areas perform best on average")
        print(f"   • Students with {best_parent_ed} parent education perform best")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("   • Implement early warning system for students with <70% attendance")
        print("   • Provide additional support for at-risk students")
        print("   • Encourage study groups to improve study hours")
        print("   • Consider location-based support programs")
        print("   • Develop parent engagement programs")

def main():
    """Main function to run the analysis"""
    # Initialize the analyzer
    analyzer = StudentPerformanceAnalyzer()
    
    print("Student Performance Analysis System")
    print("="*50)
    
    # Generate or load data
    print("Generating sample data...")
    df = analyzer.generate_sample_data(1000)
    print(f"Generated data for {len(df)} students")
    
    # Save to database
    analyzer.save_to_database()
    
    # Perform basic analysis
    analyzer.basic_statistics()
    print("\n")
    
    # Demographic analysis
    analyzer.analyze_performance_by_demographics()
    print("\n")
    
    # Correlation analysis
    print("Performing correlation analysis...")
    correlation_matrix = analyzer.perform_correlation_analysis()
    
    # Create visualizations
    print("Creating visualizations...")
    analyzer.create_visualizations()
    
    # Build prediction models
    rf_model, lr_model, feature_importance = analyzer.build_prediction_models()
    
    # Generate insights report
    analyzer.generate_insights_report()
    
    print("\n" + "="*60)
    print("Analysis completed successfully!")
    print("Check the generated visualizations and database file.")
    print("="*60)

if __name__ == "__main__":
    main()