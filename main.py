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
        """Generate realistic Nigerian secondary school student data with percentage system"""
        np.random.seed(42)

        # Student demographics - Nigerian secondary school appropriate
        student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, n_students + 1)]
        names = [f"Student_{i}" for i in range(1, n_students + 1)]
        ages = np.random.normal(15.5, 1.5, n_students).clip(13, 18).astype(int)  # Nigerian secondary ages
        genders = np.random.choice(['Male', 'Female'], n_students)
        locations = np.random.choice(['Urban', 'Rural', 'Suburban'], n_students, p=[0.35, 0.4, 0.25])

        # Nigerian secondary school classes
        classes = np.random.choice(['JSS1', 'JSS2', 'JSS3', 'SS1', 'SS2', 'SS3'],
                                   n_students, p=[0.17, 0.17, 0.16, 0.17, 0.17, 0.16])

        # Academic track system for senior classes (SS1-SS3)
        academic_tracks = []
        for i, class_level in enumerate(classes):
            if class_level in ['JSS1', 'JSS2', 'JSS3']:
                academic_tracks.append('General')  # Junior classes take all subjects
            else:
                # Senior classes choose tracks
                track = np.random.choice(['Science', 'Arts', 'Commercial'], p=[0.4, 0.35, 0.25])
                academic_tracks.append(track)

        # Nigerian socioeconomic factors
        parent_education = np.random.choice(['Primary', 'Secondary', 'NCE/OND', 'HND/BSc', 'Postgraduate'],
                                            n_students, p=[0.2, 0.35, 0.25, 0.15, 0.05])
        family_income = np.random.choice(['Low', 'Middle', 'Upper Middle', 'High'],
                                         n_students, p=[0.4, 0.35, 0.2, 0.05])
        school_type = np.random.choice(['Public', 'Private'], n_students, p=[0.7, 0.3])

        # Academic factors
        study_hours = np.random.normal(3.5, 1.2, n_students).clip(1, 7)  # More realistic for Nigerian students
        attendance_rate = np.random.normal(85, 15, n_students).clip(40, 100)

        # Base performance influenced by various factors
        base_performance = np.random.normal(65, 18, n_students)  # Nigerian grade distribution

        # Adjust performance based on realistic factors
        performance_adjustment = np.zeros(n_students)
        performance_adjustment += (study_hours - 3.5) * 4  # Study hours effect
        performance_adjustment += (attendance_rate - 85) * 0.2  # Attendance effect
        performance_adjustment += np.where(parent_education == 'Postgraduate', 12, 0)
        performance_adjustment += np.where(parent_education == 'HND/BSc', 8, 0)
        performance_adjustment += np.where(parent_education == 'NCE/OND', 5, 0)
        performance_adjustment += np.where(parent_education == 'Secondary', 2, 0)
        performance_adjustment += np.where(family_income == 'High', 8, 0)
        performance_adjustment += np.where(family_income == 'Upper Middle', 5, 0)
        performance_adjustment += np.where(family_income == 'Middle', 2, 0)
        performance_adjustment += np.where(school_type == 'Private', 6, 0)  # Private school advantage

        adjusted_performance = (base_performance + performance_adjustment).clip(20, 95)

        # Nigerian secondary school subject structure
        track_subjects = {
            'General': ['Mathematics', 'English', 'Basic_Science', 'Basic_Technology', 'Social_Studies',
                        'Civic_Education', 'Physical_Health_Education'],
            'Science': ['Mathematics', 'English', 'Physics', 'Chemistry', 'Biology', 'Further_Mathematics',
                        'Agricultural_Science'],
            'Arts': ['English', 'Literature', 'History', 'Government', 'Geography', 'CRK_IRK', 'Economics'],
            'Commercial': ['Mathematics', 'English', 'Economics', 'Accounting', 'Commerce', 'Government',
                           'Data_Processing']
        }

        # Generate term scores (3 terms per academic session)
        subject_data = {}
        all_possible_subjects = set()
        for subjects_list in track_subjects.values():
            all_possible_subjects.update(subjects_list)

        # Initialize all subjects for all terms with NaN
        for subject in all_possible_subjects:
            for term in ['Term1', 'Term2', 'Term3']:
                subject_data[f'{subject}_{term}'] = np.full(n_students, np.nan)

        # Generate realistic term progression and subject scores
        for i, (class_level, track) in enumerate(zip(classes, academic_tracks)):
            track_subject_list = track_subjects[track]
            student_base_performance = adjusted_performance[i]

            for subject in track_subject_list:
                # Subject-specific difficulty adjustments
                subject_difficulty = {
                    'Mathematics': -8, 'Further_Mathematics': -15, 'Physics': -5, 'Chemistry': -7,
                    'English': -3, 'Literature': -5, 'Economics': -4, 'Accounting': -6,
                    'Basic_Science': 5, 'Social_Studies': 8, 'Civic_Education': 10
                }

                difficulty_adj = subject_difficulty.get(subject, 0)

                # Generate three term scores with realistic progression
                term1_score = (student_base_performance + difficulty_adj + np.random.normal(0, 8)).clip(20, 95)

                # Term 2 usually shows slight improvement
                term2_score = (term1_score + np.random.normal(2, 6)).clip(20, 95)

                # Term 3 might show fatigue or continued improvement
                term3_score = (term2_score + np.random.normal(-1, 7)).clip(20, 95)

                subject_data[f'{subject}_Term1'][i] = round(term1_score, 1)
                subject_data[f'{subject}_Term2'][i] = round(term2_score, 1)
                subject_data[f'{subject}_Term3'][i] = round(term3_score, 1)

        # Calculate term averages and annual average
        term_averages = {}
        for term in ['Term1', 'Term2', 'Term3']:
            term_averages[f'{term}_Average'] = np.zeros(n_students)

            for i, track in enumerate(academic_tracks):
                track_subject_list = track_subjects[track]
                term_scores = []
                for subject in track_subject_list:
                    score = subject_data[f'{subject}_{term}'][i]
                    if not np.isnan(score):
                        term_scores.append(score)

                if term_scores:
                    term_averages[f'{term}_Average'][i] = np.mean(term_scores)
                else:
                    term_averages[f'{term}_Average'][i] = np.nan

        # Calculate annual average
        annual_average = np.zeros(n_students)
        for i in range(n_students):
            term_scores = [term_averages[f'{term}_Average'][i] for term in ['Term1', 'Term2', 'Term3']]
            term_scores = [score for score in term_scores if not np.isnan(score)]
            if term_scores:
                annual_average[i] = np.mean(term_scores)
            else:
                annual_average[i] = np.nan

        # Determine at-risk status (Nigerian context: below 50% or poor attendance)
        at_risk = ((annual_average < 50) | (attendance_rate < 75)).astype(int)

        # Nigerian secondary school appropriate extracurriculars
        extracurricular = np.random.choice(['Sports', 'Drama/Cultural', 'Debate', 'Science Club',
                                            'Literary Society', 'School Band', 'Prefectship', 'None'],
                                           n_students, p=[0.2, 0.15, 0.1, 0.15, 0.1, 0.05, 0.1, 0.15])

        # Post-secondary aspirations
        post_secondary_aspiration = np.random.choice(['University', 'Polytechnic', 'College_of_Education',
                                                      'Vocational_Training', 'Work', 'Undecided'],
                                                     n_students, p=[0.45, 0.2, 0.1, 0.1, 0.05, 0.1])

        # Create DataFrame
        data = {
            'Student_ID': student_ids,
            'Name': names,
            'Age': ages,
            'Gender': genders,
            'Location': locations,
            'Class': classes,
            'Academic_Track': academic_tracks,
            'School_Type': school_type,
            'Parent_Education': parent_education,
            'Family_Income': family_income,
            'Study_Hours_Weekly': study_hours.round(1),
            'Attendance_Rate': attendance_rate.round(1),
            'Extracurricular': extracurricular,
            'Post_Secondary_Aspiration': post_secondary_aspiration,
            'Annual_Average': annual_average.round(1),
            'At_Risk': at_risk
        }

        # Add term averages
        for term, averages in term_averages.items():
            data[term] = averages.round(1)

        # Add subject scores for all terms
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
        """Generate basic statistical summary for Nigerian secondary school system"""
        if self.df is None:
            print("No data available. Please load or generate data first.")
            return

        print("=== NIGERIAN SECONDARY SCHOOL PERFORMANCE ANALYSIS ===\n")
        print(f"Total Students: {len(self.df)}")
        print(f"Average Annual Score: {self.df['Annual_Average'].mean():.1f}%")
        print(
            f"Students at Risk (Below 50%): {self.df['At_Risk'].sum()} ({(self.df['At_Risk'].sum() / len(self.df) * 100):.1f}%)")
        print(f"Average Attendance: {self.df['Attendance_Rate'].mean():.1f}%")
        print(f"Average Study Hours: {self.df['Study_Hours_Weekly'].mean():.1f} hours/week")

        print("\n=== STUDENTS BY CLASS LEVEL ===")
        class_distribution = self.df['Class'].value_counts().sort_index()
        for class_level, count in class_distribution.items():
            percentage = (count / len(self.df)) * 100
            class_students = self.df[self.df['Class'] == class_level]
            avg_score = class_students['Annual_Average'].mean()
            print(f"{class_level}: {count} students ({percentage:.1f}%) - Avg Score: {avg_score:.1f}%")

        print("\n=== STUDENTS BY ACADEMIC TRACK (Senior Classes Only) ===")
        senior_students = self.df[self.df['Class'].isin(['SS1', 'SS2', 'SS3'])]
        if len(senior_students) > 0:
            track_distribution = senior_students['Academic_Track'].value_counts()
            for track, count in track_distribution.items():
                percentage = (count / len(senior_students)) * 100
                avg_score = senior_students[senior_students['Academic_Track'] == track]['Annual_Average'].mean()
                print(f"{track}: {count} students ({percentage:.1f}%) - Avg Score: {avg_score:.1f}%")

        print("\n=== TERM PROGRESSION ANALYSIS ===")
        term_cols = ['Term1_Average', 'Term2_Average', 'Term3_Average']
        for term in term_cols:
            if term in self.df.columns:
                avg_score = self.df[term].mean()
                print(f"{term.replace('_', ' ')}: {avg_score:.1f}%")

        print("\n=== PERFORMANCE BY SCHOOL TYPE ===")
        school_performance = self.df.groupby('School_Type')['Annual_Average'].agg(['mean', 'std', 'count'])
        for school_type, data in school_performance.iterrows():
            print(f"{school_type}: {data['count']} students - Avg Score: {data['mean']:.1f}% (±{data['std']:.1f})")

        print("\n=== POST-SECONDARY ASPIRATION ANALYSIS ===")
        aspiration_analysis = self.df.groupby('Post_Secondary_Aspiration')['Annual_Average'].agg(['mean', 'count'])
        for aspiration, data in aspiration_analysis.iterrows():
            print(f"{aspiration.replace('_', ' ')}: {data['count']} students - Avg Score: {data['mean']:.1f}%")

    def perform_correlation_analysis(self):
        """Analyze correlations between different factors"""
        # Core numeric columns that exist for all students
        core_numeric_columns = ['Age', 'Study_Hours_Weekly', 'Attendance_Rate', 'Annual_Average']

        # Add term averages for progression analysis
        term_columns = ['Term1_Average', 'Term2_Average', 'Term3_Average']

        # Universal subjects (taken by most students)
        universal_subjects = []
        if 'Mathematics_Term1' in self.df.columns:
            universal_subjects.extend(['Mathematics_Term1', 'Mathematics_Term2', 'Mathematics_Term3'])
        if 'English_Term1' in self.df.columns:
            universal_subjects.extend(['English_Term1', 'English_Term2', 'English_Term3'])

        # Combine for correlation analysis
        correlation_columns = core_numeric_columns + term_columns + universal_subjects[:6]  # Limit for readability

        # Only include columns that exist and have sufficient non-null values
        existing_columns = [col for col in correlation_columns if col in self.df.columns]
        correlation_data = self.df[existing_columns].dropna()

        if len(correlation_data) < 50:
            print("Insufficient data for correlation analysis")
            return None

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

        # Performance by Class Level
        class_performance = self.df.groupby('Class')['Annual_Average'].agg(['mean', 'std', 'count'])
        print("Performance by Class Level:")
        print(class_performance)
        print()

        # Performance by Academic Track (for senior students only)
        senior_students = self.df[self.df['Class'].isin(['SS1', 'SS2', 'SS3'])]
        if len(senior_students) > 0:
            track_performance = senior_students.groupby('Academic_Track')['Annual_Average'].agg(
                ['mean', 'std', 'count'])
            print("Performance by Academic Track (Senior Classes):")
            print(track_performance)
            print()

        # Performance by Location
        location_performance = self.df.groupby('Location')['Annual_Average'].agg(['mean', 'std', 'count'])
        print("Performance by Location:")
        print(location_performance)
        print()

        # Performance by School Type
        school_performance = self.df.groupby('School_Type')['Annual_Average'].agg(['mean', 'std', 'count'])
        print("Performance by School Type:")
        print(school_performance)
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

        # Post-Secondary Aspiration vs Performance
        aspiration_performance = self.df.groupby('Post_Secondary_Aspiration')['Annual_Average'].agg(
            ['mean', 'std', 'count'])
        print("Performance by Post-Secondary Aspiration:")
        print(aspiration_performance)

    def create_visualizations(self):
        """Create comprehensive visualizations for Nigerian secondary school data"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Nigerian Secondary School Performance Analysis Dashboard', fontsize=18, y=0.95)

        # 1. Annual Average Distribution
        axes[0, 0].hist(self.df['Annual_Average'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(self.df['Annual_Average'].mean(), color='red', linestyle='--',
                           label=f'Mean: {self.df["Annual_Average"].mean():.1f}%')
        axes[0, 0].set_title('Annual Average Distribution')
        axes[0, 0].set_xlabel('Annual Average (%)')
        axes[0, 0].set_ylabel('Number of Students')
        axes[0, 0].legend()

        # 2. Attendance vs Annual Average
        axes[0, 1].scatter(self.df['Attendance_Rate'], self.df['Annual_Average'], alpha=0.6, color='green')
        axes[0, 1].set_title('Attendance Rate vs Annual Average')
        axes[0, 1].set_xlabel('Attendance Rate (%)')
        axes[0, 1].set_ylabel('Annual Average (%)')

        # 3. Study Hours vs Annual Average
        axes[0, 2].scatter(self.df['Study_Hours_Weekly'], self.df['Annual_Average'], alpha=0.6, color='orange')
        axes[0, 2].set_title('Study Hours vs Annual Average')
        axes[0, 2].set_xlabel('Study Hours per Week')
        axes[0, 2].set_ylabel('Annual Average (%)')

        # 4. Performance by Class Level
        class_data = self.df.groupby('Class')['Annual_Average'].mean().sort_index()
        axes[1, 0].bar(class_data.index, class_data.values, color='lightcoral', alpha=0.8)
        axes[1, 0].set_title('Average Performance by Class Level')
        axes[1, 0].set_ylabel('Annual Average (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 5. At-Risk Students by Location
        risk_by_location = self.df.groupby('Location')['At_Risk'].mean() * 100
        axes[1, 1].bar(risk_by_location.index, risk_by_location.values, color='red', alpha=0.7)
        axes[1, 1].set_title('At-Risk Students by Location (%)')
        axes[1, 1].set_ylabel('Percentage At-Risk')

        # 6. Performance by School Type
        school_data = self.df.groupby('School_Type')['Annual_Average'].mean()
        axes[1, 2].bar(school_data.index, school_data.values, color='purple', alpha=0.7)
        axes[1, 2].set_title('Performance by School Type')
        axes[1, 2].set_ylabel('Annual Average (%)')

        # 7. Term Progression Analysis
        term_means = []
        term_labels = []
        for term in ['Term1_Average', 'Term2_Average', 'Term3_Average']:
            if term in self.df.columns:
                term_means.append(self.df[term].mean())
                term_labels.append(term.replace('_Average', ''))

        if term_means:
            axes[2, 0].plot(term_labels, term_means, marker='o', linewidth=2, markersize=8, color='blue')
            axes[2, 0].set_title('Term Progression Analysis')
            axes[2, 0].set_ylabel('Average Score (%)')
            axes[2, 0].grid(True, alpha=0.3)

        # 8. Performance by Academic Track (Senior Students Only)
        senior_students = self.df[self.df['Class'].isin(['SS1', 'SS2', 'SS3'])]
        if len(senior_students) > 0:
            track_data = senior_students.groupby('Academic_Track')['Annual_Average'].mean()
            axes[2, 1].bar(track_data.index, track_data.values, color='darkgreen', alpha=0.8)
            axes[2, 1].set_title('Performance by Track (Senior Classes)')
            axes[2, 1].set_ylabel('Annual Average (%)')
            axes[2, 1].tick_params(axis='x', rotation=45)
        else:
            axes[2, 1].text(0.5, 0.5, 'No Senior Students\nData Available',
                            ha='center', va='center', transform=axes[2, 1].transAxes)

        # 9. Performance by Family Income
        income_data = self.df.groupby('Family_Income')['Annual_Average'].mean()
        # Sort by income level logically
        income_order = ['Low', 'Middle', 'Upper Middle', 'High']
        income_data = income_data.reindex([inc for inc in income_order if inc in income_data.index])

        axes[2, 2].bar(income_data.index, income_data.values, color='gold', alpha=0.8)
        axes[2, 2].set_title('Performance by Family Income')
        axes[2, 2].set_ylabel('Annual Average (%)')
        axes[2, 2].tick_params(axis='x', rotation=45)

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

        # Handle categorical encoding with proper column names
        categorical_columns = ['Gender', 'Location', 'Class', 'Academic_Track',
                               'School_Type', 'Parent_Education', 'Family_Income']

        for col in categorical_columns:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                feature_columns.extend(dummies.columns.tolist())
                df_encoded = pd.concat([df_encoded, dummies], axis=1)

        # Prepare feature matrix
        X = df_encoded[feature_columns].fillna(0)

        print("### 🚨 At-Risk Student Prediction (Below 50%)")

        # At-Risk Prediction Model
        y_risk = self.df['At_Risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        risk_predictions = rf_model.predict(X_test)
        risk_accuracy = accuracy_score(y_test, risk_predictions)

        print(f"At-Risk Student Prediction Accuracy: {risk_accuracy:.3f}")
        print("\nClassification Report for At-Risk Prediction:")
        print(classification_report(y_test, risk_predictions))

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 15 Most Important Features for At-Risk Prediction:")
        print(feature_importance.head(15))

        print("\n### 📊 Annual Average Prediction")

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

        # Term progression prediction
        if all(col in self.df.columns for col in ['Term1_Average', 'Term2_Average', 'Term3_Average']):
            print("\n### 📈 Term Progression Analysis")

            # Predict Term 3 performance based on Term 1 and Term 2
            term_data = self.df[['Term1_Average', 'Term2_Average', 'Term3_Average']].dropna()

            if len(term_data) > 50:
                X_term = term_data[['Term1_Average', 'Term2_Average']]
                y_term3 = term_data['Term3_Average']

                X_train_term, X_test_term, y_train_term, y_test_term = train_test_split(
                    X_term, y_term3, test_size=0.2, random_state=42)

                term_model = LinearRegression()
                term_model.fit(X_train_term, y_train_term)
                term_predictions = term_model.predict(X_test_term)

                term_r2 = r2_score(y_test_term, term_predictions)
                term_rmse = np.sqrt(mean_squared_error(y_test_term, term_predictions))

                print(f"Term 3 Prediction (from Terms 1 & 2) R² Score: {term_r2:.3f}")
                print(f"Term 3 Prediction RMSE: {term_rmse:.3f}%")

                # Show term progression patterns
                avg_term1 = term_data['Term1_Average'].mean()
                avg_term2 = term_data['Term2_Average'].mean()
                avg_term3 = term_data['Term3_Average'].mean()

                print(f"\nTerm Progression Pattern:")
                print(f"Term 1 Average: {avg_term1:.1f}%")
                print(f"Term 2 Average: {avg_term2:.1f}% (Change: {avg_term2 - avg_term1:+.1f}%)")
                print(f"Term 3 Average: {avg_term3:.1f}% (Change: {avg_term3 - avg_term2:+.1f}%)")

        return rf_model, lr_model, feature_importance

    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n" + "=" * 60)
        print("STUDENT PERFORMANCE ANALYSIS - INSIGHTS REPORT")
        print("=" * 60)

        # Key findings
        avg_score = self.df['Annual_Average'].mean()
        at_risk_pct = (self.df['At_Risk'].sum() / len(self.df)) * 100
        high_performers = len(self.df[self.df['Annual_Average'] >= 80])
        low_attendance_risk = len(self.df[(self.df['Attendance_Rate'] < 70) & (self.df['At_Risk'] == 1)])

        print(f"\n📊 KEY METRICS:")
        print(f"   • Average Annual Score: {avg_score:.1f}%")
        print(f"   • Students at risk: {self.df['At_Risk'].sum()} ({at_risk_pct:.1f}%)")
        print(f"   • High performers (Score ≥ 80%): {high_performers}")
        print(f"   • At-risk due to low attendance: {low_attendance_risk}")

        print(f"\n🎯 KEY INSIGHTS:")

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
    print("=" * 50)

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

    print("\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("Check the generated visualizations and database file.")
    print("=" * 60)


if __name__ == "__main__":
    main()