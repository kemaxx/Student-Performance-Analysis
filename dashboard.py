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
from sklearn.metrics import accuracy_score, classification_report, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Nigerian Secondary School Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Nigerian theme - FIXED FOR BETTER TEXT VISIBILITY
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #008751;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #008751;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #008751;
    }
    .insight-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #008751;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .insight-box h4 {
        color: #008751 !important;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .insight-box ul, .insight-box ol {
        color: #333333 !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    .insight-box li {
        color: #333333 !important;
        margin-bottom: 0.5rem;
    }
    .insight-box li strong {
        color: #008751 !important;
        font-weight: bold;
    }
    .at-risk-alert {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        color: #721c24 !important;
        font-weight: bold;
    }
    .high-risk-row {
        background-color: #ffebee !important;
        color: #c62828 !important;
    }
    .medium-risk-row {
        background-color: #fff3e0 !important;
        color: #ef6c00 !important;
    }
    .low-risk-row {
        background-color: #f3e5f5 !important;
        color: #7b1fa2 !important;
    }
    /* Force text color in Streamlit containers */
    .stMarkdown .insight-box * {
        color: #333333 !important;
    }
    .stMarkdown .insight-box h4 {
        color: #008751 !important;
    }
    .stMarkdown .insight-box strong {
        color: #008751 !important;
    }
    /* Custom table styling */
    .at-risk-table {
        border-collapse: collapse;
        width: 100%;
        margin-top: 20px;
    }
    .at-risk-table th, .at-risk-table td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    .at-risk-table th {
        background-color: #008751;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load student data from database"""
    try:
        conn = sqlite3.connect('student_performance.db')
        df = pd.read_sql_query("SELECT * FROM students", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        st.info("Please run main.py first to generate the database.")
        return None


def calculate_risk_severity(df):
    """Calculate risk severity score for at-risk students"""
    risk_students = df[df['At_Risk'] == 1].copy()

    if len(risk_students) == 0:
        return risk_students

    # Initialize severity score
    risk_students['Risk_Score'] = 0.0

    # Academic performance factor (40% weight)
    performance_score = (50 - risk_students['Annual_Average'].clip(upper=50)) / 50 * 40
    risk_students['Risk_Score'] += performance_score

    # Attendance factor (30% weight)
    attendance_score = (75 - risk_students['Attendance_Rate'].clip(upper=75)) / 75 * 30
    risk_students['Risk_Score'] += attendance_score

    # Study hours factor (15% weight)
    avg_study_hours = df['Study_Hours_Weekly'].mean()
    study_deficit = np.maximum(0, avg_study_hours - risk_students['Study_Hours_Weekly'])
    study_score = (study_deficit / avg_study_hours) * 15
    risk_students['Risk_Score'] += study_score

    # Age factor (10% weight) - older students in same class more at risk
    for class_level in risk_students['Class'].unique():
        class_students = risk_students[risk_students['Class'] == class_level]
        if len(class_students) > 1:
            avg_age = df[df['Class'] == class_level]['Age'].mean()
            age_score = np.maximum(0, class_students['Age'] - avg_age) * 5
            risk_students.loc[risk_students['Class'] == class_level, 'Risk_Score'] += age_score

    # Socio-economic factors (5% weight)
    socio_score = 0
    socio_score += np.where(risk_students['Location'] == 'Rural', 2, 0)
    socio_score += np.where(risk_students['School_Type'] == 'Public', 1, 0)
    socio_score += np.where(risk_students['Family_Income'] == 'Low', 2, 0)
    risk_students['Risk_Score'] += socio_score

    # Cap the score at 100
    risk_students['Risk_Score'] = risk_students['Risk_Score'].clip(upper=100)

    # Classify severity levels
    risk_students['Risk_Level'] = pd.cut(
        risk_students['Risk_Score'],
        bins=[0, 30, 60, 100],
        labels=['Low Risk', 'Medium Risk', 'High Risk'],
        include_lowest=True
    )

    return risk_students


def create_at_risk_students_table(df):
    """Create comprehensive at-risk students table sorted by severity"""
    st.subheader("⚠️ At-Risk Students Analysis - Severity Ranking")

    # Calculate risk severity
    risk_students = calculate_risk_severity(df)

    if len(risk_students) == 0:
        st.success("🎉 Excellent! No students are currently identified as at-risk.")
        st.info("All students are performing above 50% with adequate attendance rates.")
        return

    # Sort by risk score (highest risk first)
    risk_students_sorted = risk_students.sort_values('Risk_Score', ascending=False).reset_index(drop=True)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        high_risk_count = len(risk_students_sorted[risk_students_sorted['Risk_Level'] == 'High Risk'])
        st.metric(
            label="🔴 High Risk",
            value=high_risk_count,
            delta=f"{high_risk_count / len(df) * 100:.1f}% of total students"
        )

    with col2:
        medium_risk_count = len(risk_students_sorted[risk_students_sorted['Risk_Level'] == 'Medium Risk'])
        st.metric(
            label="🟠 Medium Risk",
            value=medium_risk_count,
            delta=f"{medium_risk_count / len(df) * 100:.1f}% of total students"
        )

    with col3:
        low_risk_count = len(risk_students_sorted[risk_students_sorted['Risk_Level'] == 'Low Risk'])
        st.metric(
            label="🟡 Low Risk",
            value=low_risk_count,
            delta=f"{low_risk_count / len(df) * 100:.1f}% of total students"
        )

    with col4:
        avg_risk_score = risk_students_sorted['Risk_Score'].mean()
        st.metric(
            label="📊 Avg Risk Score",
            value=f"{avg_risk_score:.1f}",
            delta="Out of 100"
        )

    # Risk level distribution chart
    st.subheader("📊 Risk Level Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Risk level pie chart
        risk_counts = risk_students_sorted['Risk_Level'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='At-Risk Students by Severity Level',
            color_discrete_map={
                'High Risk': '#dc3545',
                'Medium Risk': '#fd7e14',
                'Low Risk': '#ffc107'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Risk score distribution
        fig_hist = px.histogram(
            risk_students_sorted,
            x='Risk_Score',
            nbins=15,
            title='Risk Score Distribution',
            color='Risk_Level',
            color_discrete_map={
                'High Risk': '#dc3545',
                'Medium Risk': '#fd7e14',
                'Low Risk': '#ffc107'
            }
        )
        fig_hist.update_layout(xaxis_title="Risk Score", yaxis_title="Number of Students")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Filters for the table
    st.subheader("🔍 Filter At-Risk Students")

    col1, col2, col3 = st.columns(3)

    with col1:
        risk_level_filter = st.multiselect(
            "Filter by Risk Level:",
            options=['High Risk', 'Medium Risk', 'Low Risk'],
            default=['High Risk', 'Medium Risk', 'Low Risk']
        )

    with col2:
        class_filter = st.multiselect(
            "Filter by Class:",
            options=sorted(risk_students_sorted['Class'].unique()),
            default=sorted(risk_students_sorted['Class'].unique())
        )

    with col3:
        risk_score_range = st.slider(
            "Risk Score Range:",
            min_value=0.0,
            max_value=100.0,
            value=(0.0, 100.0),
            step=5.0
        )

    # Apply filters
    filtered_students = risk_students_sorted[
        (risk_students_sorted['Risk_Level'].isin(risk_level_filter)) &
        (risk_students_sorted['Class'].isin(class_filter)) &
        (risk_students_sorted['Risk_Score'] >= risk_score_range[0]) &
        (risk_students_sorted['Risk_Score'] <= risk_score_range[1])
        ]

    # Main at-risk students table
    st.subheader(f"📋 At-Risk Students Table ({len(filtered_students)} students)")

    if len(filtered_students) == 0:
        st.info("No students match the selected filters.")
        return

    # Prepare display columns
    display_columns = [
        'Student_ID', 'Name', 'Class', 'Risk_Level', 'Risk_Score',
        'Annual_Average', 'Attendance_Rate', 'Study_Hours_Weekly',
        'Location', 'School_Type', 'Parent_Education', 'Family_Income'
    ]

    display_data = filtered_students[display_columns].copy()

    # Format the data for better display
    display_data['Risk_Score'] = display_data['Risk_Score'].round(1)
    display_data['Annual_Average'] = display_data['Annual_Average'].round(1).astype(str) + '%'
    display_data['Attendance_Rate'] = display_data['Attendance_Rate'].round(1).astype(str) + '%'
    display_data['Study_Hours_Weekly'] = display_data['Study_Hours_Weekly'].round(1).astype(str) + 'h'

    # Rename columns for display
    display_data.columns = [
        'Student ID', 'Name', 'Class', 'Risk Level', 'Risk Score',
        'Annual Average', 'Attendance Rate', 'Study Hours/Week',
        'Location', 'School Type', 'Parent Education', 'Family Income'
    ]

    # Color-code the dataframe based on risk level
    def highlight_risk_level(row):
        risk_level = row['Risk Level']
        if risk_level == 'High Risk':
            return ['background-color: #ffebee; color: #c62828'] * len(row)
        elif risk_level == 'Medium Risk':
            return ['background-color: #fff3e0; color: #ef6c00'] * len(row)
        else:  # Low Risk
            return ['background-color: #f3e5f5; color: #7b1fa2'] * len(row)

    # Display the styled table
    styled_table = display_data.style.apply(highlight_risk_level, axis=1)
    st.dataframe(styled_table, use_container_width=True, height=400)

    # Detailed intervention recommendations
    st.subheader("💡 Intervention Recommendations by Risk Level")

    # High Risk Students
    high_risk_students = filtered_students[filtered_students['Risk_Level'] == 'High Risk']
    if len(high_risk_students) > 0:
        st.markdown("""
        <div class="insight-box">
            <h4>🔴 High Risk Students - Immediate Action Required</h4>
            <ul>
                <li><strong>Immediate Parent Conference:</strong> Schedule urgent meetings with parents/guardians</li>
                <li><strong>Individual Learning Plan:</strong> Develop personalized academic support plans</li>
                <li><strong>Daily Monitoring:</strong> Track attendance and academic progress daily</li>
                <li><strong>Peer Mentoring:</strong> Pair with high-performing students for support</li>
                <li><strong>Counseling Support:</strong> Assess for personal/social issues affecting performance</li>
                <li><strong>Additional Classes:</strong> Provide extra lessons in weak subject areas</li>
                <li><strong>Study Skills Training:</strong> Intensive study techniques and time management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Medium Risk Students
    medium_risk_students = filtered_students[filtered_students['Risk_Level'] == 'Medium Risk']
    if len(medium_risk_students) > 0:
        st.markdown("""
        <div class="insight-box">
            <h4>🟠 Medium Risk Students - Close Monitoring Required</h4>
            <ul>
                <li><strong>Weekly Check-ins:</strong> Regular progress monitoring with class teachers</li>
                <li><strong>Parent Communication:</strong> Monthly progress reports to parents</li>
                <li><strong>Study Groups:</strong> Encourage participation in peer study groups</li>
                <li><strong>Homework Support:</strong> After-school homework assistance programs</li>
                <li><strong>Subject-Specific Help:</strong> Targeted support in weakest subjects</li>
                <li><strong>Attendance Incentives:</strong> Rewards for improved attendance patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Low Risk Students
    low_risk_students = filtered_students[filtered_students['Risk_Level'] == 'Low Risk']
    if len(low_risk_students) > 0:
        st.markdown("""
        <div class="insight-box">
            <h4>🟡 Low Risk Students - Preventive Measures</h4>
            <ul>
                <li><strong>Regular Monitoring:</strong> Bi-weekly progress checks</li>
                <li><strong>Motivational Support:</strong> Encourage continued effort and improvement</li>
                <li><strong>Goal Setting:</strong> Help establish realistic academic goals</li>
                <li><strong>Skill Building:</strong> Focus on study skills and exam techniques</li>
                <li><strong>Positive Reinforcement:</strong> Recognize improvements and efforts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Export functionality
    st.subheader("📤 Export At-Risk Students Data")

    col1, col2 = st.columns(2)

    with col1:
        # Create CSV download
        csv_data = filtered_students[display_columns].to_csv(index=False)
        st.download_button(
            label="📊 Download Full Data (CSV)",
            data=csv_data,
            file_name=f"at_risk_students_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    with col2:
        # Create summary report
        summary_data = {
            'Risk_Level': ['High Risk', 'Medium Risk', 'Low Risk', 'Total'],
            'Count': [
                len(filtered_students[filtered_students['Risk_Level'] == 'High Risk']),
                len(filtered_students[filtered_students['Risk_Level'] == 'Medium Risk']),
                len(filtered_students[filtered_students['Risk_Level'] == 'Low Risk']),
                len(filtered_students)
            ],
            'Percentage': [
                f"{len(filtered_students[filtered_students['Risk_Level'] == 'High Risk']) / len(df) * 100:.1f}%",
                f"{len(filtered_students[filtered_students['Risk_Level'] == 'Medium Risk']) / len(df) * 100:.1f}%",
                f"{len(filtered_students[filtered_students['Risk_Level'] == 'Low Risk']) / len(df) * 100:.1f}%",
                f"{len(filtered_students) / len(df) * 100:.1f}%"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)

        st.download_button(
            label="📈 Download Summary (CSV)",
            data=summary_csv,
            file_name=f"at_risk_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    # Class-wise at-risk analysis
    st.subheader("🏫 Class-wise At-Risk Analysis")

    class_risk_analysis = filtered_students.groupby(['Class', 'Risk_Level']).size().unstack(fill_value=0)

    if not class_risk_analysis.empty:
        fig_class_risk = px.bar(
            class_risk_analysis.reset_index(),
            x='Class',
            y=['High Risk', 'Medium Risk', 'Low Risk'],
            title='At-Risk Students by Class and Severity Level',
            color_discrete_map={
                'High Risk': '#dc3545',
                'Medium Risk': '#fd7e14',
                'Low Risk': '#ffc107'
            }
        )
        fig_class_risk.update_layout(
            xaxis_title="Class Level",
            yaxis_title="Number of Students",
            legend_title="Risk Level"
        )
        st.plotly_chart(fig_class_risk, use_container_width=True)


def create_overview_metrics(df):
    """Create overview metrics cards for Nigerian secondary school system"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📚 Total Students",
            value=f"{len(df):,}",
            delta=None
        )

    with col2:
        avg_score = df['Annual_Average'].mean()
        st.metric(
            label="📊 Average Score",
            value=f"{avg_score:.1f}%",
            delta=f"{avg_score - 50:.1f}% vs Pass Mark"
        )

    with col3:
        at_risk_count = df['At_Risk'].sum()
        at_risk_pct = (at_risk_count / len(df)) * 100
        st.metric(
            label="⚠️ At-Risk Students",
            value=f"{at_risk_count}",
            delta=f"-{at_risk_pct:.1f}% of total",
            delta_color="inverse"
        )

    with col4:
        avg_attendance = df['Attendance_Rate'].mean()
        st.metric(
            label="🎯 Avg Attendance",
            value=f"{avg_attendance:.1f}%",
            delta=f"{avg_attendance - 75:.1f}% vs Target"
        )


def create_class_distribution(df):
    """Create class level distribution charts"""
    st.subheader("🏫 Student Distribution by Class Level")

    col1, col2 = st.columns(2)

    with col1:
        # Class distribution
        class_counts = df['Class'].value_counts().sort_index()
        fig_class = px.bar(x=class_counts.index, y=class_counts.values,
                           title='Students by Class Level',
                           color=class_counts.values,
                           color_continuous_scale='viridis')
        fig_class.update_layout(xaxis_title="Class Level", yaxis_title="Number of Students")
        st.plotly_chart(fig_class, use_container_width=True)

    with col2:
        # Academic track distribution for senior students
        senior_students = df[df['Class'].isin(['SS1', 'SS2', 'SS3'])]
        if len(senior_students) > 0:
            track_counts = senior_students['Academic_Track'].value_counts()
            fig_track = px.pie(values=track_counts.values, names=track_counts.index,
                               title='Academic Track Distribution (Senior Classes)',
                               color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_track, use_container_width=True)
        else:
            st.info("No senior students (SS1-SS3) in current dataset")


def create_performance_charts(df):
    """Create performance visualization charts"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Annual Average Distribution")
        fig_hist = px.histogram(df, x='Annual_Average', nbins=20,
                                title='Student Annual Average Distribution',
                                color_discrete_sequence=['#008751'])
        fig_hist.add_vline(x=df['Annual_Average'].mean(), line_dash="dash",
                           line_color="red", annotation_text=f"Mean: {df['Annual_Average'].mean():.1f}%")
        fig_hist.add_vline(x=50, line_dash="dot",
                           line_color="orange", annotation_text="Pass Mark: 50%")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("🎯 At-Risk Analysis")
        at_risk_counts = df['At_Risk'].value_counts()
        fig_pie = px.pie(values=at_risk_counts.values,
                         names=['Not At-Risk', 'At-Risk'],
                         title='At-Risk Student Distribution',
                         color_discrete_sequence=['#28a745', '#dc3545'])
        st.plotly_chart(fig_pie, use_container_width=True)


def create_term_progression_analysis(df):
    """Analyze term progression"""
    st.subheader("📅 Term Progression Analysis")

    term_columns = ['Term1_Average', 'Term2_Average', 'Term3_Average']
    available_terms = [col for col in term_columns if col in df.columns]

    if len(available_terms) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            # Term progression line chart
            term_means = []
            term_labels = []
            for term in available_terms:
                term_means.append(df[term].mean())
                term_labels.append(term.replace('_Average', ''))

            fig_progression = px.line(x=term_labels, y=term_means,
                                      title='Average Performance Across Terms',
                                      markers=True,
                                      line_shape='linear')
            fig_progression.update_traces(line_color='#008751', line_width=3, marker_size=8)
            fig_progression.update_layout(xaxis_title="Term", yaxis_title="Average Score (%)")
            st.plotly_chart(fig_progression, use_container_width=True)

        with col2:
            # Term comparison box plot
            term_data = df[available_terms].melt(var_name='Term', value_name='Score')
            term_data['Term'] = term_data['Term'].str.replace('_Average', '')

            fig_box = px.box(term_data, x='Term', y='Score',
                             title='Score Distribution by Term',
                             color='Term')
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("Term progression data not available")


def create_subject_performance_analysis(df):
    """Analyze subject performance by track"""
    st.subheader("📚 Subject Performance Analysis")

    # Get all subject columns
    subject_columns = [col for col in df.columns if
                       col.endswith('_Term1') or col.endswith('_Term2') or col.endswith('_Term3')]

    if subject_columns:
        # Extract unique subjects
        subjects = list(set([col.split('_')[0] for col in subject_columns]))
        subjects.sort()

        # Core subjects analysis (Mathematics and English - taken by most students)
        core_subjects = ['Mathematics', 'English']
        available_core = [subj for subj in core_subjects if any(col.startswith(subj) for col in subject_columns)]

        if available_core:
            col1, col2 = st.columns(2)

            with col1:
                # Core subjects performance
                core_performance = []
                for subject in available_core:
                    # Get Term 3 scores (final term) or available term
                    term_cols = [col for col in subject_columns if col.startswith(subject)]
                    if term_cols:
                        latest_term = sorted(term_cols)[-1]  # Get latest term
                        avg_score = df[latest_term].mean()
                        core_performance.append({'Subject': subject, 'Average': avg_score})

                if core_performance:
                    core_df = pd.DataFrame(core_performance)
                    fig_core = px.bar(core_df, x='Subject', y='Average',
                                      title='Core Subjects Performance',
                                      color='Average',
                                      color_continuous_scale='viridis')
                    st.plotly_chart(fig_core, use_container_width=True)

            with col2:
                # Performance by academic track (for senior students)
                senior_students = df[df['Class'].isin(['SS1', 'SS2', 'SS3'])]
                if len(senior_students) > 0:
                    track_performance = senior_students.groupby('Academic_Track')['Annual_Average'].mean().reset_index()
                    fig_track = px.bar(track_performance, x='Academic_Track', y='Annual_Average',
                                       title='Performance by Academic Track',
                                       color='Annual_Average',
                                       color_continuous_scale='oranges')
                    st.plotly_chart(fig_track, use_container_width=True)


def create_demographic_analysis(df):
    """Create demographic analysis charts"""
    st.subheader("👥 Performance by Demographics")

    col1, col2 = st.columns(2)

    with col1:
        # Performance by Gender
        gender_performance = df.groupby('Gender')['Annual_Average'].mean().reset_index()
        fig_gender = px.bar(gender_performance, x='Gender', y='Annual_Average',
                            title='Average Performance by Gender',
                            color='Annual_Average',
                            color_continuous_scale='blues')
        st.plotly_chart(fig_gender, use_container_width=True)

        # Performance by Location
        location_performance = df.groupby('Location')['Annual_Average'].mean().reset_index()
        fig_location = px.bar(location_performance, x='Location', y='Annual_Average',
                              title='Average Performance by Location',
                              color='Annual_Average',
                              color_continuous_scale='greens')
        st.plotly_chart(fig_location, use_container_width=True)

    with col2:
        # Performance by Parent Education
        parent_ed_performance = df.groupby('Parent_Education')['Annual_Average'].mean().reset_index()
        fig_parent_ed = px.bar(parent_ed_performance, x='Parent_Education', y='Annual_Average',
                               title='Average Performance by Parent Education',
                               color='Annual_Average',
                               color_continuous_scale='oranges')
        fig_parent_ed.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_parent_ed, use_container_width=True)

        # Performance by School Type
        school_performance = df.groupby('School_Type')['Annual_Average'].mean().reset_index()
        fig_school = px.bar(school_performance, x='School_Type', y='Annual_Average',
                            title='Performance by School Type',
                            color='Annual_Average',
                            color_continuous_scale='purples')
        st.plotly_chart(fig_school, use_container_width=True)


def create_correlation_analysis(df):
    """Create correlation analysis visualizations"""
    st.subheader("🔗 Factor Correlation Analysis")

    # Core numeric columns
    numeric_columns = ['Age', 'Study_Hours_Weekly', 'Attendance_Rate', 'Annual_Average']

    # Add term averages if available
    term_columns = ['Term1_Average', 'Term2_Average', 'Term3_Average']
    available_terms = [col for col in term_columns if col in df.columns]
    numeric_columns.extend(available_terms)

    # Add core subject scores if available
    core_subjects = ['Mathematics_Term3', 'English_Term3']  # Final term scores
    available_subjects = [col for col in core_subjects if col in df.columns]
    numeric_columns.extend(available_subjects)

    # Create correlation matrix
    correlation_data = df[numeric_columns].dropna()

    if len(correlation_data) > 10:
        corr_matrix = correlation_data.corr()

        fig_corr = px.imshow(corr_matrix,
                             title='Correlation Matrix - Nigerian Secondary School Factors',
                             color_continuous_scale='RdBu',
                             aspect="auto",
                             text_auto=True)
        fig_corr.update_layout(width=800, height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Key scatter plots
        col1, col2 = st.columns(2)

        with col1:
            fig_scatter1 = px.scatter(df, x='Attendance_Rate', y='Annual_Average',
                                      color='At_Risk',
                                      title='Attendance Rate vs Annual Average',
                                      color_discrete_map={0: 'green', 1: 'red'},
                                      hover_data=['Study_Hours_Weekly', 'Class'])
            st.plotly_chart(fig_scatter1, use_container_width=True)

        with col2:
            fig_scatter2 = px.scatter(df, x='Study_Hours_Weekly', y='Annual_Average',
                                      color='At_Risk',
                                      title='Study Hours vs Annual Average',
                                      color_discrete_map={0: 'green', 1: 'red'},
                                      hover_data=['Attendance_Rate', 'Class'])
            st.plotly_chart(fig_scatter2, use_container_width=True)


def create_prediction_models(df):
    """Build and display prediction models for Nigerian context"""
    st.subheader("🤖 Predictive Modeling - Nigerian Secondary School")

    # Prepare features
    feature_columns = ['Age', 'Study_Hours_Weekly', 'Attendance_Rate']

    # Encode categorical variables
    df_encoded = df.copy()
    categorical_columns = ['Gender', 'Location', 'Class', 'Academic_Track',
                           'School_Type', 'Parent_Education', 'Family_Income']

    for col in categorical_columns:
        if col in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            feature_columns.extend(dummies.columns.tolist())
            df_encoded = pd.concat([df_encoded, dummies], axis=1)

    X = df_encoded[feature_columns].fillna(0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚨 At-Risk Student Prediction")

        try:
            # At-Risk Prediction Model
            y_risk = df['At_Risk']
            X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            risk_predictions = rf_model.predict(X_test)
            risk_accuracy = accuracy_score(y_test, risk_predictions)

            st.metric("Model Accuracy", f"{risk_accuracy:.3f}")
            st.metric("At-Risk Students in Test", f"{y_test.sum()}")

            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)

            fig_importance = px.bar(feature_importance, x='Importance', y='Feature',
                                    title='Top 10 Important Features for Risk Prediction',
                                    orientation='h')
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)

        except Exception as e:
            st.error(f"Error in at-risk prediction: {e}")

    with col2:
        st.markdown("### 📊 Annual Average Prediction")

        try:
            # Annual Average Prediction Model
            y_avg = df['Annual_Average'].dropna()
            X_avg = X.loc[y_avg.index]

            X_train_avg, X_test_avg, y_train_avg, y_test_avg = train_test_split(
                X_avg, y_avg, test_size=0.2, random_state=42)

            lr_model = LinearRegression()
            lr_model.fit(X_train_avg, y_train_avg)
            avg_predictions = lr_model.predict(X_test_avg)

            r2 = r2_score(y_test_avg, avg_predictions)
            rmse = np.sqrt(((y_test_avg - avg_predictions) ** 2).mean())

            st.metric("R² Score", f"{r2:.3f}")
            st.metric("RMSE", f"{rmse:.2f}%")

            # Actual vs Predicted plot
            fig_pred = px.scatter(x=y_test_avg, y=avg_predictions,
                                  title='Actual vs Predicted Annual Average',
                                  labels={'x': 'Actual Average (%)', 'y': 'Predicted Average (%)'})

            # Add perfect prediction line
            min_val = min(y_test_avg.min(), avg_predictions.min())
            max_val = max(y_test_avg.max(), avg_predictions.max())
            fig_pred.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                               line=dict(color="red", dash="dash"))

            st.plotly_chart(fig_pred, use_container_width=True)

        except Exception as e:
            st.error(f"Error in average prediction: {e}")


def create_insights_and_recommendations(df):
    """Generate insights and recommendations for Nigerian context"""
    st.subheader("💡 Key Insights & Recommendations")

    # Calculate key statistics
    avg_score = df['Annual_Average'].mean()
    at_risk_pct = (df['At_Risk'].sum() / len(df)) * 100
    high_performers = len(df[df['Annual_Average'] >= 80])
    pass_rate = (len(df[df['Annual_Average'] >= 50]) / len(df)) * 100

    corr_attendance_score = df['Attendance_Rate'].corr(df['Annual_Average'])
    corr_study_score = df['Study_Hours_Weekly'].corr(df['Annual_Average'])

    best_location = df.groupby('Location')['Annual_Average'].mean().idxmax()
    best_parent_ed = df.groupby('Parent_Education')['Annual_Average'].mean().idxmax()
    best_school_type = df.groupby('School_Type')['Annual_Average'].mean().idxmax()

    # Key Insights with better visibility
    st.markdown(f"""
    <div class="insight-box">
        <h4>🔍 Key Findings - Nigerian Secondary School Performance:</h4>
        <ul>
            <li><strong>Overall Pass Rate:</strong> {pass_rate:.1f}% of students scoring above 50%</li>
            <li><strong>Attendance Impact:</strong> Correlation of {corr_attendance_score:.3f} with performance</li>
            <li><strong>Study Hours Effect:</strong> Correlation of {corr_study_score:.3f} with performance</li>
            <li><strong>Location Factor:</strong> {best_location} students perform best ({df.groupby('Location')['Annual_Average'].mean().loc[best_location]:.1f}% average)</li>
            <li><strong>School Type Impact:</strong> {best_school_type} schools show better performance</li>
            <li><strong>Parent Education:</strong> {best_parent_ed} level shows highest student performance</li>
            <li><strong>Risk Level:</strong> {at_risk_pct:.1f}% of students are at-risk (below 50% or poor attendance)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Recommendations specific to Nigerian education system with better visibility
    st.markdown("""
    <div class="insight-box">
        <h4>📋 Strategic Recommendations for Nigerian Secondary Schools:</h4>
        <ol>
            <li><strong>Attendance Monitoring:</strong> Implement digital attendance tracking and parent notification system</li>
            <li><strong>At-Risk Intervention:</strong> Early identification program for students below 50% performance</li>
            <li><strong>Teacher Training:</strong> Focus on pedagogical skills for subjects with low performance</li>
            <li><strong>Infrastructure Support:</strong> Prioritize schools in underperforming locations</li>
            <li><strong>Parent Engagement:</strong> Community education programs to involve parents in academic planning</li>
            <li><strong>Study Skills Workshop:</strong> Train students on effective study techniques and time management</li>
            <li><strong>Term Monitoring:</strong> Track performance trends across the three terms</li>
            <li><strong>Track-Specific Support:</strong> Tailored resources for Science, Arts, and Commercial tracks</li>
            <li><strong>WAEC/NECO Preparation:</strong> Focused preparation for external examinations</li>
            <li><strong>Career Guidance:</strong> Post-secondary planning aligned with academic performance</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


def student_lookup_tool(df):
    """Individual student lookup and analysis for Nigerian students"""
    st.subheader("🔍 Individual Student Analysis")

    # Student selection
    student_list = ['Select a student...'] + df['Student_ID'].tolist()
    selected_student = st.selectbox("Choose a student:", student_list)

    if selected_student != 'Select a student...':
        student_data = df[df['Student_ID'] == selected_student].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 👤 Student Profile")
            st.write(f"**Student ID:** {student_data['Student_ID']}")
            st.write(f"**Name:** {student_data['Name']}")
            st.write(f"**Age:** {student_data['Age']} years")
            st.write(f"**Gender:** {student_data['Gender']}")
            st.write(f"**Location:** {student_data['Location']}")
            st.write(f"**Class:** {student_data['Class']}")
            if 'Academic_Track' in student_data and student_data['Academic_Track'] != 'General':
                st.write(f"**Academic Track:** {student_data['Academic_Track']}")

        with col2:
            st.markdown("### 📊 Academic Performance")
            st.metric("Annual Average", f"{student_data['Annual_Average']:.1f}%")
            st.metric("Attendance Rate", f"{student_data['Attendance_Rate']:.1f}%")
            st.metric("Study Hours/Week", f"{student_data['Study_Hours_Weekly']:.1f}")
            st.write(f"**School Type:** {student_data['School_Type']}")
            st.write(f"**Parent Education:** {student_data['Parent_Education']}")

            if student_data['At_Risk'] == 1:
                st.markdown(
                    '<div class="at-risk-alert">⚠️ <strong>Student is AT RISK</strong><br/>Requires immediate intervention</div>',
                    unsafe_allow_html=True)
            else:
                st.success("✅ Student is performing satisfactorily")

        with col3:
            st.markdown("### 📚 Term Performance")
            term_columns = ['Term1_Average', 'Term2_Average', 'Term3_Average']
            available_terms = [col for col in term_columns if col in df.columns]

            for term_col in available_terms:
                if not pd.isna(student_data[term_col]):
                    term_name = term_col.replace('_Average', '')
                    st.write(f"**{term_name}:** {student_data[term_col]:.1f}%")

        # Subject performance analysis
        st.markdown("### 📖 Subject Performance Analysis")

        # Get subject scores for this student
        subject_columns = [col for col in df.columns if '_Term' in col and not col.endswith('_Average')]
        student_subjects = {}

        for col in subject_columns:
            if not pd.isna(student_data[col]):
                subject_name = col.split('_')[0]
                term = col.split('_')[1]
                if subject_name not in student_subjects:
                    student_subjects[subject_name] = {}
                student_subjects[subject_name][term] = student_data[col]

        if student_subjects:
            # Create subject performance chart
            subjects_for_chart = []
            scores_for_chart = []

            for subject, terms in student_subjects.items():
                # Get the latest available term score
                if 'Term3' in terms:
                    subjects_for_chart.append(subject)
                    scores_for_chart.append(terms['Term3'])
                elif 'Term2' in terms:
                    subjects_for_chart.append(subject)
                    scores_for_chart.append(terms['Term2'])
                elif 'Term1' in terms:
                    subjects_for_chart.append(subject)
                    scores_for_chart.append(terms['Term1'])

            if subjects_for_chart:
                fig_subjects = px.bar(x=subjects_for_chart, y=scores_for_chart,
                                      title=f'Subject Performance - {selected_student}',
                                      color=scores_for_chart,
                                      color_continuous_scale='RdYlGn')
                fig_subjects.update_layout(xaxis_title="Subjects", yaxis_title="Score (%)")
                fig_subjects.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_subjects, use_container_width=True)


def main():
    """Main dashboard application for Nigerian Secondary School System"""

    # Header
    st.markdown('<h1 class="main-header">🇳🇬 Nigerian Secondary School Performance Dashboard</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
    Comprehensive analysis of student performance in Nigerian secondary schools (JSS1 - SS3)
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner('Loading Nigerian secondary school data...'):
        df = load_data()

    if df is None:
        st.stop()

    st.success(f"✅ Data loaded successfully! Analyzing {len(df)} students from Nigerian secondary schools.")

    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Section:",
        ["Overview", "Class Distribution", "Performance Analysis", "Term Progression",
         "Subject Analysis", "Demographics", "Correlations", "Predictive Models",
         "At-Risk Students", "Individual Students", "Insights & Recommendations"]
    )

    # Sidebar filters
    st.sidebar.title("🔍 Filters")

    # Class filter
    classes = ['All'] + sorted(df['Class'].unique().tolist())
    selected_class = st.sidebar.selectbox("Filter by Class:", classes)

    # Location filter
    locations = ['All'] + df['Location'].unique().tolist()
    selected_location = st.sidebar.selectbox("Filter by Location:", locations)

    # School Type filter
    school_types = ['All'] + df['School_Type'].unique().tolist()
    selected_school_type = st.sidebar.selectbox("Filter by School Type:", school_types)

    # Performance range filter
    score_range = st.sidebar.slider("Annual Average Range (%):",
                                    float(df['Annual_Average'].min()),
                                    float(df['Annual_Average'].max()),
                                    (float(df['Annual_Average'].min()), float(df['Annual_Average'].max())))

    # Apply filters
    filtered_df = df.copy()
    if selected_class != 'All':
        filtered_df = filtered_df[filtered_df['Class'] == selected_class]
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['Location'] == selected_location]
    if selected_school_type != 'All':
        filtered_df = filtered_df[filtered_df['School_Type'] == selected_school_type]
    filtered_df = filtered_df[(filtered_df['Annual_Average'] >= score_range[0]) &
                              (filtered_df['Annual_Average'] <= score_range[1])]

    st.sidebar.write(f"Filtered dataset: {len(filtered_df)} students")

    # Main content based on page selection
    if page == "Overview":
        create_overview_metrics(filtered_df)
        st.markdown("---")
        create_performance_charts(filtered_df)

    elif page == "Class Distribution":
        create_class_distribution(filtered_df)

    elif page == "Performance Analysis":
        create_performance_charts(filtered_df)

    elif page == "Term Progression":
        create_term_progression_analysis(filtered_df)

    elif page == "Subject Analysis":
        create_subject_performance_analysis(filtered_df)

    elif page == "Demographics":
        create_demographic_analysis(filtered_df)

    elif page == "Correlations":
        create_correlation_analysis(filtered_df)

    elif page == "Predictive Models":
        create_prediction_models(filtered_df)

    elif page == "At-Risk Students":
        create_at_risk_students_table(filtered_df)

    elif page == "Individual Students":
        student_lookup_tool(filtered_df)

    elif page == "Insights & Recommendations":
        create_insights_and_recommendations(filtered_df)

    # Footer with Nigerian context
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
        "🇳🇬 Nigerian Secondary School Performance Analysis System | "
        "Supporting JSS1-SS3 Students | Built with Streamlit | "
        f"Data last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()