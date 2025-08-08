import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Student Performance Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load student data from database or generate sample data"""
    try:
        conn = sqlite3.connect('student_performance.db')
        df = pd.read_sql_query("SELECT * FROM students", conn)
        conn.close()
        return df
    except:
        # Generate sample data if database doesn't exist
        return generate_sample_data()


@st.cache_data
def generate_sample_data(n_students=1000):
    """Generate sample student data"""
    np.random.seed(42)

    # Student demographics
    student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, n_students + 1)]
    names = [f"Student_{i}" for i in range(1, n_students + 1)]
    ages = np.random.normal(20, 2, n_students).clip(17, 25).astype(int)
    genders = np.random.choice(['Male', 'Female'], n_students)
    locations = np.random.choice(['Urban', 'Rural', 'Suburban'], n_students, p=[0.4, 0.3, 0.3])

    # Socioeconomic factors
    parent_education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_students,
                                        p=[0.3, 0.4, 0.2, 0.1])
    family_income = np.random.choice(['Low', 'Middle', 'High'], n_students, p=[0.3, 0.5, 0.2])

    # Academic factors
    study_hours = np.random.normal(6, 2, n_students).clip(1, 12)
    attendance_rate = np.random.normal(85, 15, n_students).clip(40, 100)

    # Subject grades
    base_performance = np.random.normal(70, 15, n_students)
    performance_adjustment = np.zeros(n_students)
    performance_adjustment += (study_hours - 6) * 2
    performance_adjustment += (attendance_rate - 85) * 0.3
    performance_adjustment += np.where(parent_education == 'PhD', 8, 0)
    performance_adjustment += np.where(parent_education == 'Master', 5, 0)
    performance_adjustment += np.where(parent_education == 'Bachelor', 2, 0)
    performance_adjustment += np.where(family_income == 'High', 5, 0)
    performance_adjustment += np.where(family_income == 'Middle', 2, 0)

    adjusted_performance = (base_performance + performance_adjustment).clip(0, 100)

    subjects = ['Mathematics', 'English', 'Computer Science', 'Physics', 'Statistics']
    subject_data = {}

    for subject in subjects:
        subject_variation = np.random.normal(0, 5, n_students)
        subject_grades = (adjusted_performance + subject_variation).clip(0, 100)
        subject_data[f'{subject}_Grade'] = subject_grades.round(1)

    grade_columns = [f'{subject}_Grade' for subject in subjects]
    overall_gpa = np.mean([subject_data[col] for col in grade_columns], axis=0)
    at_risk = ((overall_gpa < 60) | (attendance_rate < 70)).astype(int)
    extracurricular = np.random.choice(['Sports', 'Arts', 'Science Club', 'None'], n_students, p=[0.3, 0.2, 0.2, 0.3])

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

    data.update(subject_data)
    return pd.DataFrame(data)


def create_overview_metrics(df):
    """Create overview metrics cards"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📚 Total Students",
            value=f"{len(df):,}",
            delta=None
        )

    with col2:
        avg_gpa = df['Overall_GPA'].mean()
        st.metric(
            label="📊 Average GPA",
            value=f"{avg_gpa:.2f}",
            delta=f"{avg_gpa - 70:.1f} vs Target (70)"
        )

    with col3:
        at_risk_count = df['At_Risk'].sum()
        at_risk_pct = (at_risk_count / len(df)) * 100
        st.metric(
            label="⚠️ At-Risk Students",
            value=f"{at_risk_count}",
            delta=f"-{at_risk_pct:.1f}% of total"
        )

    with col4:
        avg_attendance = df['Attendance_Rate'].mean()
        st.metric(
            label="🎯 Avg Attendance",
            value=f"{avg_attendance:.1f}%",
            delta=f"{avg_attendance - 80:.1f}% vs Target (80%)"
        )


def create_performance_charts(df):
    """Create performance visualization charts"""

    # GPA Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 GPA Distribution")
        fig_hist = px.histogram(df, x='Overall_GPA', nbins=20,
                                title='Student GPA Distribution',
                                color_discrete_sequence=['#1f77b4'])
        fig_hist.add_vline(x=df['Overall_GPA'].mean(), line_dash="dash",
                           line_color="red", annotation_text=f"Mean: {df['Overall_GPA'].mean():.1f}")
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("🎯 At-Risk Analysis")
        at_risk_counts = df['At_Risk'].value_counts()
        fig_pie = px.pie(values=at_risk_counts.values,
                         names=['Not At-Risk', 'At-Risk'],
                         title='At-Risk Student Distribution',
                         color_discrete_sequence=['#2ca02c', '#d62728'])
        st.plotly_chart(fig_pie, use_container_width=True)

    # Subject Performance Comparison
    st.subheader("📚 Subject Performance Comparison")
    subjects = ['Mathematics_Grade', 'English_Grade', 'Computer Science_Grade', 'Physics_Grade', 'Statistics_Grade']
    subject_means = [df[subject].mean() for subject in subjects]
    subject_names = [s.replace('_Grade', '') for s in subjects]

    fig_bar = px.bar(x=subject_names, y=subject_means,
                     title='Average Performance by Subject',
                     color=subject_means,
                     color_continuous_scale='viridis')
    fig_bar.update_layout(showlegend=False, xaxis_title="Subjects", yaxis_title="Average Grade")
    st.plotly_chart(fig_bar, use_container_width=True)


def create_demographic_analysis(df):
    """Create demographic analysis charts"""
    st.subheader("👥 Performance by Demographics")

    col1, col2 = st.columns(2)

    with col1:
        # Performance by Gender
        gender_performance = df.groupby('Gender')['Overall_GPA'].mean().reset_index()
        fig_gender = px.bar(gender_performance, x='Gender', y='Overall_GPA',
                            title='Average GPA by Gender',
                            color='Overall_GPA',
                            color_continuous_scale='blues')
        st.plotly_chart(fig_gender, use_container_width=True)

        # Performance by Location
        location_performance = df.groupby('Location')['Overall_GPA'].mean().reset_index()
        fig_location = px.bar(location_performance, x='Location', y='Overall_GPA',
                              title='Average GPA by Location',
                              color='Overall_GPA',
                              color_continuous_scale='greens')
        st.plotly_chart(fig_location, use_container_width=True)

    with col2:
        # Performance by Parent Education
        parent_ed_performance = df.groupby('Parent_Education')['Overall_GPA'].mean().reset_index()
        fig_parent_ed = px.bar(parent_ed_performance, x='Parent_Education', y='Overall_GPA',
                               title='Average GPA by Parent Education',
                               color='Overall_GPA',
                               color_continuous_scale='oranges')
        fig_parent_ed.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_parent_ed, use_container_width=True)

        # At-Risk by Income Level
        income_risk = df.groupby('Family_Income')['At_Risk'].mean() * 100
        fig_income_risk = px.bar(x=income_risk.index, y=income_risk.values,
                                 title='At-Risk Percentage by Family Income',
                                 color=income_risk.values,
                                 color_continuous_scale='reds')
        fig_income_risk.update_layout(xaxis_title="Family Income", yaxis_title="At-Risk Percentage (%)")
        st.plotly_chart(fig_income_risk, use_container_width=True)


def create_correlation_analysis(df):
    """Create correlation analysis visualizations"""
    st.subheader("🔗 Factor Correlation Analysis")

    # Correlation heatmap
    numeric_columns = ['Age', 'Study_Hours_Weekly', 'Attendance_Rate', 'Overall_GPA',
                       'Mathematics_Grade', 'English_Grade', 'Computer Science_Grade',
                       'Physics_Grade', 'Statistics_Grade']

    corr_matrix = df[numeric_columns].corr()

    fig_corr = px.imshow(corr_matrix,
                         title='Correlation Matrix - Performance Factors',
                         color_continuous_scale='RdBu',
                         aspect="auto")
    fig_corr.update_layout(width=800, height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Scatter plots for key relationships
    col1, col2 = st.columns(2)

    with col1:
        fig_scatter1 = px.scatter(df, x='Attendance_Rate', y='Overall_GPA',
                                  color='At_Risk',
                                  title='Attendance Rate vs GPA',
                                  color_discrete_map={0: 'green', 1: 'red'},
                                  hover_data=['Study_Hours_Weekly'])
        st.plotly_chart(fig_scatter1, use_container_width=True)

    with col2:
        fig_scatter2 = px.scatter(df, x='Study_Hours_Weekly', y='Overall_GPA',
                                  color='At_Risk',
                                  title='Study Hours vs GPA',
                                  color_discrete_map={0: 'green', 1: 'red'},
                                  hover_data=['Attendance_Rate'])
        st.plotly_chart(fig_scatter2, use_container_width=True)


def create_prediction_models(df):
    """Build and display prediction models"""
    st.subheader("🤖 Predictive Modeling")

    # Prepare features
    feature_columns = ['Age', 'Study_Hours_Weekly', 'Attendance_Rate']

    # Encode categorical variables
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=['Gender', 'Location', 'Parent_Education', 'Family_Income'],
                                prefix=['Gender', 'Location', 'ParentEd', 'Income'])

    # Get feature columns after encoding
    feature_cols = [col for col in df_encoded.columns if
                    col.startswith(('Gender_', 'Location_', 'ParentEd_', 'Income_'))]
    feature_cols.extend(feature_columns)

    X = df_encoded[feature_cols]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚨 At-Risk Student Prediction")

        # At-Risk Prediction Model
        y_risk = df['At_Risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        risk_predictions = rf_model.predict(X_test)
        risk_accuracy = accuracy_score(y_test, risk_predictions)

        st.metric("Model Accuracy", f"{risk_accuracy:.3f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)

        fig_importance = px.bar(feature_importance, x='Importance', y='Feature',
                                title='Top 10 Important Features',
                                orientation='h')
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)

    with col2:
        st.markdown("### 📊 GPA Prediction")

        # GPA Prediction Model
        y_gpa = df['Overall_GPA']
        X_train_gpa, X_test_gpa, y_train_gpa, y_test_gpa = train_test_split(X, y_gpa, test_size=0.2, random_state=42)

        lr_model = LinearRegression()
        lr_model.fit(X_train_gpa, y_train_gpa)
        gpa_predictions = lr_model.predict(X_test_gpa)

        # Calculate R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test_gpa, gpa_predictions)
        rmse = np.sqrt(((y_test_gpa - gpa_predictions) ** 2).mean())

        st.metric("R² Score", f"{r2:.3f}")
        st.metric("RMSE", f"{rmse:.3f}")

        # Actual vs Predicted plot
        fig_pred = px.scatter(x=y_test_gpa, y=gpa_predictions,
                              title='Actual vs Predicted GPA',
                              labels={'x': 'Actual GPA', 'y': 'Predicted GPA'})

        # Add perfect prediction line
        min_val = min(y_test_gpa.min(), gpa_predictions.min())
        max_val = max(y_test_gpa.max(), gpa_predictions.max())
        fig_pred.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                           line=dict(color="red", dash="dash"))

        st.plotly_chart(fig_pred, use_container_width=True)


def create_insights_and_recommendations(df):
    """Generate insights and recommendations"""
    st.subheader("💡 Key Insights & Recommendations")

    # Calculate key statistics
    avg_gpa = df['Overall_GPA'].mean()
    at_risk_pct = (df['At_Risk'].sum() / len(df)) * 100
    high_performers = len(df[df['Overall_GPA'] >= 80])
    low_attendance_risk = len(df[(df['Attendance_Rate'] < 70) & (df['At_Risk'] == 1)])

    corr_attendance_gpa = df['Attendance_Rate'].corr(df['Overall_GPA'])
    corr_study_gpa = df['Study_Hours_Weekly'].corr(df['Overall_GPA'])

    best_location = df.groupby('Location')['Overall_GPA'].mean().idxmax()
    best_parent_ed = df.groupby('Parent_Education')['Overall_GPA'].mean().idxmax()

    # Key Insights
    st.markdown("""
    <div class="insight-box">
    <h4>🔍 Key Findings:</h4>
    <ul>
        <li><strong>Attendance Impact:</strong> Correlation of {:.3f} with GPA - Strong positive relationship</li>
        <li><strong>Study Hours Effect:</strong> Correlation of {:.3f} with GPA - Moderate positive relationship</li>
        <li><strong>Location Factor:</strong> Students from {} areas perform best on average</li>
        <li><strong>Education Impact:</strong> Students with {} parent education show highest performance</li>
        <li><strong>Risk Identification:</strong> {:.1f}% of students are currently at-risk</li>
    </ul>
    </div>
    """.format(corr_attendance_gpa, corr_study_gpa, best_location, best_parent_ed, at_risk_pct),
                unsafe_allow_html=True)

    # Recommendations
    st.markdown("""
    <div class="insight-box">
    <h4>📋 Strategic Recommendations:</h4>
    <ol>
        <li><strong>Early Warning System:</strong> Implement automated alerts for students with <70% attendance</li>
        <li><strong>Targeted Support:</strong> Provide additional resources for at-risk students identified by the model</li>
        <li><strong>Study Skills Program:</strong> Develop workshops to help students optimize study hours effectively</li>
        <li><strong>Location-Based Support:</strong> Create specialized programs for students from underperforming areas</li>
        <li><strong>Parent Engagement:</strong> Establish programs to involve parents in academic planning</li>
        <li><strong>Attendance Incentives:</strong> Implement reward systems to improve attendance rates</li>
        <li><strong>Subject-Specific Support:</strong> Focus additional resources on subjects with lower average performance</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)


def student_lookup_tool(df):
    """Individual student lookup and analysis"""
    st.subheader("🔍 Individual Student Analysis")

    # Student selection
    student_list = ['Select a student...'] + df['Student_ID'].tolist()
    selected_student = st.selectbox("Choose a student:", student_list)

    if selected_student != 'Select a student...':
        student_data = df[df['Student_ID'] == selected_student].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 👤 Student Profile")
            st.write(f"**ID:** {student_data['Student_ID']}")
            st.write(f"**Name:** {student_data['Name']}")
            st.write(f"**Age:** {student_data['Age']}")
            st.write(f"**Gender:** {student_data['Gender']}")
            st.write(f"**Location:** {student_data['Location']}")

        with col2:
            st.markdown("### 📊 Academic Performance")
            st.metric("Overall GPA", f"{student_data['Overall_GPA']:.2f}")
            st.metric("Attendance Rate", f"{student_data['Attendance_Rate']:.1f}%")
            st.metric("Study Hours/Week", f"{student_data['Study_Hours_Weekly']:.1f}")

            if student_data['At_Risk'] == 1:
                st.error("⚠️ Student is AT RISK")
            else:
                st.success("✅ Student is performing well")

        with col3:
            st.markdown("### 📚 Subject Grades")
            subjects = ['Mathematics', 'English', 'Computer Science', 'Physics', 'Statistics']
            for subject in subjects:
                grade = student_data[f'{subject}_Grade']
                st.write(f"**{subject}:** {grade:.1f}")

        # Subject performance radar chart
        subjects = ['Mathematics', 'English', 'Computer Science', 'Physics', 'Statistics']
        grades = [student_data[f'{subject}_Grade'] for subject in subjects]

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=grades,
            theta=subjects,
            fill='toself',
            name='Student Performance',
            line_color='blue'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            title=f"Subject Performance Radar - {selected_student}",
            showlegend=True
        )

        st.plotly_chart(fig_radar, use_container_width=True)


def main():
    """Main dashboard application"""

    # Header
    st.markdown('<h1 class="main-header">🎓 Student Performance Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Load data
    with st.spinner('Loading student data...'):
        df = load_data()

    st.success(f"✅ Data loaded successfully! Analyzing {len(df)} students.")

    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Section:",
        ["Overview", "Performance Analysis", "Demographics", "Correlations",
         "Predictive Models", "Individual Students", "Insights & Recommendations"]
    )

    # Sidebar filters
    st.sidebar.title("🔍 Filters")

    # Location filter
    locations = ['All'] + df['Location'].unique().tolist()
    selected_location = st.sidebar.selectbox("Filter by Location:", locations)

    # Gender filter
    genders = ['All'] + df['Gender'].unique().tolist()
    selected_gender = st.sidebar.selectbox("Filter by Gender:", genders)

    # GPA range filter
    gpa_range = st.sidebar.slider("GPA Range:",
                                  float(df['Overall_GPA'].min()),
                                  float(df['Overall_GPA'].max()),
                                  (float(df['Overall_GPA'].min()), float(df['Overall_GPA'].max())))

    # Apply filters
    filtered_df = df.copy()
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['Location'] == selected_location]
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    filtered_df = filtered_df[(filtered_df['Overall_GPA'] >= gpa_range[0]) &
                              (filtered_df['Overall_GPA'] <= gpa_range[1])]

    st.sidebar.write(f"Filtered dataset: {len(filtered_df)} students")

    # Main content based on page selection
    if page == "Overview":
        create_overview_metrics(filtered_df)
        st.markdown("---")
        create_performance_charts(filtered_df)

    elif page == "Performance Analysis":
        create_performance_charts(filtered_df)

    elif page == "Demographics":
        create_demographic_analysis(filtered_df)

    elif page == "Correlations":
        create_correlation_analysis(filtered_df)

    elif page == "Predictive Models":
        create_prediction_models(filtered_df)

    elif page == "Individual Students":
        student_lookup_tool(filtered_df)

    elif page == "Insights & Recommendations":
        create_insights_and_recommendations(filtered_df)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
        "Student Performance Analysis System | Built with Streamlit | "
        f"Data last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()