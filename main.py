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
        ages = np.random.normal(15.5, 1.5, n_students).clip(13, 18).astype(int)
        genders = np.random.choice(['Male', 'Female'], n_students)
        locations = np.random.choice(['Urban', 'Rural', 'Suburban'], n_students, p=[0.4, 0.3, 0.3])
        
        # Add class levels first (needed for subject assignment)
        classes = np.random.choice(['JSS1', 'JSS2', 'JSS3', 'SS1', 'SS2', 'SS3'],
                                   size=n_students, p=[0.15, 0.15, 0.15, 0.15, 0.2, 0.2])

        # Add academic track only for senior classes
        academic_track = []
        for c in classes:
            if c in ['SS1', 'SS2', 'SS3']:
                academic_track.append(np.random.choice(['Science', 'Arts', 'Commercial']))
            else:
                academic_track.append('General')
        
        # Socioeconomic factors
        parent_education = np.random.choice(['Primary', 'Secondary', 'NCE/OND', 'HND/BSc', 'Postgraduate'],
                                            n_students, p=[0.2, 0.35, 0.25, 0.15, 0.05])
        family_income = np.random.choice(['Low', 'Middle', 'High'], n_students, p=[0.3, 0.5, 0.2])
        
        # Academic factors
        study_hours = np.random.normal(6, 2, n_students).clip(1, 12)
        attendance_rate = np.random.normal(85, 15, n_students).clip(40, 100)

        # Location effects - more realistic for Nigerian context
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
        performance_adjustment += (attendance_rate - 75) * 0.3  # Attendance effect
        performance_adjustment += location_effects  # Apply location effects
        
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

        # Nigerian secondary school subject structure by class and track
        track_subjects = {
            'JSS': ['Mathematics', 'English', 'Basic_Science', 'Basic_Technology', 'Social_Studies', 
                    'Civic_Education', 'Physical_Health_Education'],
            'Science': ['Mathematics', 'English', 'Physics', 'Chemistry', 'Biology', 'Further_Mathematics'],
            'Arts': ['English', 'Literature', 'History', 'Government', 'Geography', 'CRK_IRK', 'Economics'],
            'Commercial': ['Mathematics', 'English', 'Economics', 'Accounting', 'Commerce', 'Government']
        }

        # Subject difficulty adjustments
        subject_difficulty = {
            'Mathematics': -6, 'Further_Mathematics': -12, 'Physics': -4, 'Chemistry': -5,
            'Biology': -3, 'English': -3, 'Literature': -5, 'Economics': -4,
            'Accounting': -6, 'History': -2, 'Government': -1, 'Geography': -1,
            'Basic_Science': 3, 'Social_Studies': 5, 'Civic_Education': 8,
            'Basic_Technology': 2, 'Physical_Health_Education': 10, 'Commerce': -2,
            'CRK_IRK': 3
        }

        # Generate student aptitude patterns
        science_aptitude = np.random.normal(0, 6, n_students)
        language_aptitude = np.random.normal(0, 5, n_students)
        analytical_aptitude = np.random.normal(0, 7, n_students)

        # Initialize all possible subjects with NaN
        all_possible_subjects = set()
        for subjects_list in track_subjects.values():
            all_possible_subjects.update(subjects_list)

        subject_data = {}
        # Initialize all subjects for all students with NaN
        for subject in all_possible_subjects:
            subject_data[f'{subject}_Grade'] = np.full(n_students, np.nan)
            for term in ['Term1', 'Term2', 'Term3']:
                subject_data[f'{subject}_{term}'] = np.full(n_students, np.nan)

        # Generate scores only for relevant subjects per student
        for i, (class_level, track) in enumerate(zip(classes, academic_track)):
            # Determine which subjects this student takes
            if class_level in ['JSS1', 'JSS2', 'JSS3']:
                student_subjects = track_subjects['JSS']
            else:  # SS1, SS2, SS3
                student_subjects = track_subjects[track]
            
            for subject in student_subjects:
                # Start with this student's adjusted performance
                base_subject_score = adjusted_performance[i]
                
                # Apply subject difficulty
                difficulty_adj = subject_difficulty.get(subject, 0)
                base_subject_score += difficulty_adj
                
                # Apply aptitude bonuses based on subject type
                if subject in ['Mathematics', 'Further_Mathematics', 'Physics', 'Chemistry', 'Biology', 'Basic_Science']:
                    base_subject_score += science_aptitude[i] * 0.7
                elif subject in ['English', 'Literature', 'CRK_IRK']:
                    base_subject_score += language_aptitude[i] * 0.8
                elif subject in ['Economics', 'Accounting', 'Commerce', 'Government', 'Basic_Technology']:
                    base_subject_score += analytical_aptitude[i] * 0.6
                
                # Location-specific subject effects
                if subject in ['Physics', 'Chemistry', 'Basic_Technology'] and locations[i] == 'Rural':
                    base_subject_score -= 3  # Rural schools may lack resources
                elif subject == 'English' and locations[i] == 'Urban':
                    base_subject_score += 2  # Better English exposure in urban areas
                
                # Generate overall subject grade
                subject_variation = np.random.normal(0, 3)
                subject_grade = np.clip(base_subject_score + subject_variation, 0, 100)
                subject_data[f'{subject}_Grade'][i] = round(subject_grade, 1)
                
                # Generate term-specific scores with realistic progression
                for term_idx, term in enumerate(['Term1', 'Term2', 'Term3']):
                    term_variation = np.random.normal(term_idx * 0.5, 3)
                    term_score = np.clip(base_subject_score + term_variation, 0, 100)
                    subject_data[f'{subject}_{term}'][i] = round(term_score, 1)

        # Calculate term averages only from subjects each student actually takes
        term_averages = {}
        for term in ['Term1', 'Term2', 'Term3']:
            term_averages[f'{term}_Average'] = np.full(n_students, np.nan)
            
            for i, (class_level, track) in enumerate(zip(classes, academic_track)):
                if class_level in ['JSS1', 'JSS2', 'JSS3']:
                    student_subjects = track_subjects['JSS']
                else:
                    student_subjects = track_subjects[track]
                
                # Calculate average only from subjects this student takes
                student_term_scores = []
                for subject in student_subjects:
                    score = subject_data[f'{subject}_{term}'][i]
                    if not np.isnan(score):
                        student_term_scores.append(score)
                
                if student_term_scores:
                    term_averages[f'{term}_Average'][i] = np.mean(student_term_scores)

        # Calculate annual average from term averages
        annual_average = np.full(n_students, np.nan)
        for i in range(n_students):
            student_term_avgs = []
            for term in ['Term1', 'Term2', 'Term3']:
                if not np.isnan(term_averages[f'{term}_Average'][i]):
                    student_term_avgs.append(term_averages[f'{term}_Average'][i])
            
            if student_term_avgs:
                annual_average[i] = np.mean(student_term_avgs)

        # Calculate overall GPA (same as annual average for this system)
        overall_gpa = annual_average.copy()
         
        # Determine at-risk status (GPA < 60 or attendance < 70%)
        at_risk = ((overall_gpa < 60) | (attendance_rate < 70)).astype(int)
        
        # Extracurricular activities
        extracurricular = np.random.choice(['Sports', 'Arts', 'Science Club', 'None'], 
                                           n_students, p=[0.3, 0.2, 0.2, 0.3])

        # Add school type (Public vs Private)
        school_types = np.random.choice(['Public', 'Private'], size=n_students, p=[0.7, 0.3])
        
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
            'Overall_GPA': np.nan_to_num(overall_gpa, nan=0).round(2),
            'At_Risk': at_risk,
            'Class': classes,
            'Academic_Track': academic_track,
            'School_Type': school_types,
            'Annual_Average': np.nan_to_num(annual_average, nan=0).round(2)
        }
        
        # Add term averages
        for term, averages in term_averages.items():
            data[term] = np.nan_to_num(averages, nan=0).round(1)

        # Add subject grades (with NaN values preserved for subjects not taken)
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
        
        print("=== NIGERIAN SECONDARY SCHOOL PERFORMANCE ANALYSIS ===\n")
        print(f"Total Students: {len(self.df)}")
        print(f"Average Annual Score: {self.df['Annual_Average'].mean():.1f}%")
        print(f"Students at Risk: {self.df['At_Risk'].sum()} ({(self.df['At_Risk'].sum()/len(self.df)*100):.1f}%)")
        print(f"Average Attendance: {self.df['Attendance_Rate'].mean():.1f}%")
        print(f"Average Study Hours: {self.df['Study_Hours_Weekly'].mean():.1f} hours/week")
        
        # Show class distribution
        print("\n=== STUDENTS BY CLASS LEVEL ===")
        class_distribution = self.df['Class'].value_counts().sort_index()
        for class_level, count in class_distribution.items():
            percentage = (count / len(self.df)) * 100
            print(f"{class_level}: {count} students ({percentage:.1f}%)")

        # Show track distribution for senior students
        print("\n=== ACADEMIC TRACK DISTRIBUTION (SS1-SS3) ===")
        senior_students = self.df[self.df['Class'].isin(['SS1', 'SS2', 'SS3'])]
        if len(senior_students) > 0:
            track_distribution = senior_students['Academic_Track'].value_counts()
            for track, count in track_distribution.items():
                percentage = (count / len(senior_students)) * 100
                print(f"{track}: {count} students ({percentage:.1f}%)")
    
    def perform_correlation_analysis(self):
        """Analyze correlations between different factors"""
        # Use only core subjects that most students have
        numeric_columns = ['Age', 'Study_Hours_Weekly', 'Attendance_Rate', 'Annual_Average']
        
        # Add term averages
        numeric_columns.extend(['Term1_Average', 'Term2_Average', 'Term3_Average'])
        
        # Add common subjects (Mathematics and English are in all tracks)
        common_subjects = ['Mathematics_Grade', 'English_Grade']
        for subject in common_subjects:
            if subject in self.df.columns:
                numeric_columns.append(subject)
        
        correlation_data = self.df[numeric_columns].dropna()
        correlation_matrix = correlation_data.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, fmt='.2f')
        plt.title('Correlation Matrix - Nigerian Secondary School Performance Factors')
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix
    
    def analyze_performance_by_demographics(self):
        """Analyze performance across different demographic groups"""
        print("=== PERFORMANCE BY DEMOGRAPHICS (Nigerian Context) ===\n")
        
        # Performance by Gender
        gender_performance = self.df.groupby('Gender')['Annual_Average'].agg(['mean', 'std', 'count'])
        print("Performance by Gender:")
        print(gender_performance)
        print()
        
        # Performance by Location
        location_performance = self.df.groupby('Location')['Annual_Average'].agg(['mean', 'std', 'count'])
        print("Performance by Location:")
        print(location_performance)
        print()
        
        # Performance by Parent Education
        education_performance = self.df.groupby('Parent_Education')['Annual_Average'].agg(['mean', 'std', 'count'])
        print("Performance by Parent Education:")
        print(education_performance)
        print()
        
        # Performance by Family Income
        income_performance = self.df.groupby('Family_Income')['Annual_Average'].agg(['mean', 'std', 'count'])
        print("Performance by Family Income:")
        print(income_performance)
        print()

        # Performance by Academic Track (for senior students)
        senior_students = self.df[self.df['Class'].isin(['SS1', 'SS2', 'SS3'])]
        if len(senior_students) > 0:
            track_performance = senior_students.groupby('Academic_Track')['Annual_Average'].agg(['mean', 'std', 'count'])
            print("Performance by Academic Track (Senior Classes):")
            print(track_performance)
    
    def create_visualizations(self):
        """Create comprehensive visualizations for Nigerian secondary school data"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Nigerian Secondary School Performance Analysis Dashboard', fontsize=18, y=0.95)
        
        # 1. Annual Average Distribution
        axes[0,0].hist(self.df['Annual_Average'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0,0].axvline(self.df['Annual_Average'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.df["Annual_Average"].mean():.1f}%')
        axes[0,0].axvline(50, color='orange', linestyle=':', label='Pass Mark: 50%')
        axes[0,0].set_title('Annual Average Distribution')
        axes[0,0].set_xlabel('Annual Average (%)')
        axes[0,0].set_ylabel('Number of Students')
        axes[0,0].legend()
        
        # 2. Attendance vs Annual Average
        axes[0,1].scatter(self.df['Attendance_Rate'], self.df['Annual_Average'], alpha=0.6, color='green')
        axes[0,1].set_title('Attendance Rate vs Annual Average')
        axes[0,1].set_xlabel('Attendance Rate (%)')
        axes[0,1].set_ylabel('Annual Average (%)')
        
        # 3. Study Hours vs Annual Average
        axes[0,2].scatter(self.df['Study_Hours_Weekly'], self.df['Annual_Average'], alpha=0.6, color='orange')
        axes[0,2].set_title('Study Hours vs Annual Average')
        axes[0,2].set_xlabel('Study Hours per Week')
        axes[0,2].set_ylabel('Annual Average (%)')
        
        # 4. Performance by Class Level
        class_data = self.df.groupby('Class')['Annual_Average'].mean().sort_index()
        axes[1,0].bar(class_data.index, class_data.values, color='lightcoral', alpha=0.8)
        axes[1,0].set_title('Average Performance by Class Level')
        axes[1,0].set_ylabel('Annual Average (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. At-Risk Students by Location
        risk_by_location = self.df.groupby('Location')['At_Risk'].mean() * 100
        axes[1,1].bar(risk_by_location.index, risk_by_location.values, color='red', alpha=0.7)
        axes[1,1].set_title('At-Risk Students by Location (%)')
        axes[1,1].set_ylabel('Percentage At-Risk')
        
        # 6. Performance by School Type
        school_data = self.df.groupby('School_Type')['Annual_Average'].mean()
        axes[1,2].bar(school_data.index, school_data.values, color='purple', alpha=0.7)
        axes[1,2].set_title('Performance by School Type')
        axes[1,2].set_ylabel('Annual Average (%)')
        
        # 7. Term Progression Analysis
        term_means = []
        term_labels = []
        for term in ['Term1_Average', 'Term2_Average', 'Term3_Average']:
            if term in self.df.columns:
                term_means.append(self.df[term].mean())
                term_labels.append(term.replace('_Average', ''))

        if term_means:
            axes[2,0].plot(term_labels, term_means, marker='o', linewidth=2, markersize=8, color='blue')
            axes[2,0].set_title('Term Progression Analysis')
            axes[2,0].set_ylabel('Average Score (%)')
            axes[2,0].grid(True, alpha=0.3)
        
        # 8. Performance by Academic Track (Senior Students Only)
        senior_students = self.df[self.df['Class'].isin(['SS1', 'SS2', 'SS3'])]
        if len(senior_students) > 0:
            track_data = senior_students.groupby('Academic_Track')['Annual_Average'].mean()
            axes[2,1].bar(track_data.index, track_data.values, color='darkgreen', alpha=0.8)
            axes[2,1].set_title('Performance by Track (Senior Classes)')
            axes[2,1].set_ylabel('Annual Average (%)')
            axes[2,1].tick_params(axis='x', rotation=45)
        else:
            axes[2,1].text(0.5, 0.5, 'No Senior Students\nData Available', 
                          ha='center', va='center', transform=axes[2,1].transAxes)
        
        # 9. Performance by Family Income
        income_data = self.df.groupby('Family_Income')['Annual_Average'].mean()
        axes[2,2].bar(income_data.index, income_data.values, color='gold', alpha=0.8)
        axes[2,2].set_title('Performance by Family Income')
        axes[2,2].set_ylabel('Annual Average (%)')
        axes[2,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()
    
    def build_prediction_models(self):
        """Build machine learning models for Nigerian secondary school prediction"""
        print("=== BUILDING PREDICTION MODELS (Nigerian Context) ===\n")
        
        # Prepare features for modeling
        feature_columns = ['Age', 'Study_Hours_Weekly', 'Attendance_Rate']
        
        # Encode categorical variables
        df_encoded = self.df.copy()
        categorical_columns = ['Gender', 'Location', 'Class', 'Academic_Track', 
                              'School_Type', 'Parent_Education', 'Family_Income']
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                feature_columns.extend(dummies.columns.tolist())
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Prepare feature matrix
        X = df_encoded[feature_columns].fillna(0)
        
        print("### At-Risk Student Prediction")
        
        # At-Risk Prediction Model
        y_risk = self.df['At_Risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
        
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)
        risk_predictions = rf_classifier.predict(X_test)
        risk_accuracy = accuracy_score(y_test, risk_predictions)
        
        print(f"At-Risk Student Prediction Accuracy: {risk_accuracy:.3f}")
        print("\nClassification Report for At-Risk Prediction:")
        print(classification_report(y_test, risk_predictions))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_classifier.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 15 Most Important Features for At-Risk Prediction:")
        print(feature_importance.head(15))
        
        print("\n### Annual Average Prediction")
        
        # Annual Average Prediction Model
        y_avg = self.df['Annual_Average'].dropna()
        X_avg = X.loc[y_avg.index]
        
        X_train_avg, X_test_avg, y_train_avg, y_test_avg = train_test_split(
            X_avg, y_avg, test_size=0.2, random_state=42)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train_avg, y_train_avg)
        avg_predictions = lr_model.predict(X_test_avg)
        
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test_avg, avg_predictions)
        rmse = np.sqrt(mean_squared_error(y_test_avg, avg_predictions))
        
        print(f"Annual Average Prediction R² Score: {r2:.3f}")
        print(f"Annual Average Prediction RMSE: {rmse:.3f}%")
        
        return rf_classifier, lr_model, feature_importance
    
    def generate_insights_report(self):
        """Generate comprehensive insights report for Nigerian secondary schools"""
        print("\n" + "="*60)
        print("NIGERIAN SECONDARY SCHOOL PERFORMANCE ANALYSIS - INSIGHTS REPORT")
        print("="*60)
        
        # Key findings
        avg_score = self.df['Annual_Average'].mean()
        at_risk_pct = (self.df['At_Risk'].sum() / len(self.df)) * 100
        high_performers = len(self.df[self.df['Annual_Average'] >= 80])
        pass_rate = (len(self.df[self.df['Annual_Average'] >= 50]) / len(self.df)) * 100
        
        print(f"\nKEY METRICS:")
        print(f"   • Average Annual Score: {avg_score:.1f}%")
        print(f"   • Students at risk: {self.df['At_Risk'].sum()} ({at_risk_pct:.1f}%)")
        print(f"   • High performers (Score ≥ 80%): {high_performers}")
        print(f"   • Pass rate (Score ≥ 50%): {pass_rate:.1f}%")
        
        print(f"\nKEY INSIGHTS:")
        
        # Correlation insights
        corr_attendance_score = self.df['Attendance_Rate'].corr(self.df['Annual_Average'])
        corr_study_score = self.df['Study_Hours_Weekly'].corr(self.df['Annual_Average'])
        
        print(f"   • Attendance has a {corr_attendance_score:.3f} correlation with Annual Average")
        print(f"   • Study hours have a {corr_study_score:.3f} correlation with Annual Average")
        
        # Demographics insights
        best_location = self.df.groupby('Location')['Annual_Average'].mean().idxmax()
        best_parent_ed = self.df.groupby('Parent_Education')['Annual_Average'].mean().idxmax()
        
        print(f"   • Students from {best_location} areas perform best on average")
        print(f"   • Students with {best_parent_ed} parent education perform best")
        
        # Academic track insights (for senior students)
        senior_students = self.df[self.df['Class'].isin(['SS1', 'SS2', 'SS3'])]
        if len(senior_students) > 0:
            best_track = senior_students.groupby('Academic_Track')['Annual_Average'].mean().idxmax()
            print(f"   • {best_track} track students perform best among senior classes")
        
        print(f"\nRECOMMENDATIONS:")
        print("   • Implement early warning system for students with <75% attendance")
        print("   • Provide additional support for at-risk students")
        print("   • Strengthen rural school resources, especially for science subjects")
        print("   • Develop parent engagement programs")
        print("   • Consider track-specific support programs")
        print("   • Focus on WAEC/NECO preparation for SS3 students")

def main():
    """Main function to run the analysis"""
    # Initialize the analyzer
    analyzer = StudentPerformanceAnalyzer()
    
    print("Nigerian Secondary School Performance Analysis System")
    print("="*60)
    
    # Generate or load data
    print("Generating sample data with Nigerian curriculum structure...")
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
    print("Data now reflects authentic Nigerian secondary school curriculum structure!")
    print("Run 'streamlit run dashboard.py' to view the enhanced dashboard.")
    print("="*60)

if __name__ == "__main__":
    main()