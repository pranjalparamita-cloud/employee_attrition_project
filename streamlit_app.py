"""
============================================================
STREAMLIT DASHBOARD
Employee Attrition Risk Prediction & Scoring System
============================================================
Run: streamlit run streamlit_app.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Employee Attrition Risk System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #e74c3c;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        border-left: 4px solid #3498db;
    }
    .risk-badge-high {
        background-color: #e74c3c;
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-weight: bold;
    }
    .risk-badge-medium {
        background-color: #f39c12;
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-weight: bold;
    }
    .risk-badge-low {
        background-color: #2ecc71;
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER: FILE PATH (same directory)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(SCRIPT_DIR, filename)


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    risk_df = pd.read_csv(get_path('employee_risk_scores.csv'))
    feature_importance = pd.read_csv(get_path('feature_importance.csv'))
    model_comparison = pd.read_csv(get_path('model_comparison.csv'), index_col=0)
    return risk_df, feature_importance, model_comparison


@st.cache_resource
def load_models():
    model = joblib.load(get_path('best_model.pkl'))
    scaler = joblib.load(get_path('scaler.pkl'))
    label_encoders = joblib.load(get_path('label_encoders.pkl'))
    feature_names = joblib.load(get_path('feature_names.pkl'))
    return model, scaler, label_encoders, feature_names


# Try loading
try:
    risk_df, feature_importance, model_comparison = load_data()
    model, scaler, label_encoders, feature_names = load_models()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    st.error(f"❌ Required file not found: {e}")
    st.warning("⚠️ Please run `python attrition_analysis.py` first to generate all required files.")
    st.stop()
except Exception as e:
    data_loaded = False
    st.error(f"❌ Error loading data: {e}")
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_risk_cat(prob, low_thresh, high_thresh):
    if prob < low_thresh:
        return 'Low Risk'
    elif prob < high_thresh:
        return 'Medium Risk'
    else:
        return 'High Risk'


RISK_COLORS = {
    'High Risk': '#e74c3c',
    'Medium Risk': '#f39c12',
    'Low Risk': '#2ecc71'
}

RISK_EMOJI = {
    'High Risk': '🔴',
    'Medium Risk': '🟡',
    'Low Risk': '🟢'
}


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🎯 Controls & Filters")

# Department filter
dept_options = ['All Departments']
if 'Department' in risk_df.columns:
    dept_options += sorted(risk_df['Department'].astype(str).unique().tolist())
selected_dept = st.sidebar.selectbox("📂 Department", dept_options)

# Job Role filter
role_options = ['All Roles']
if 'JobRole' in risk_df.columns:
    role_options += sorted(risk_df['JobRole'].astype(str).unique().tolist())
selected_role = st.sidebar.selectbox("💼 Job Role", role_options)

# Risk threshold sliders
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Risk Thresholds")
low_threshold = st.sidebar.slider("Low → Medium threshold (%)", 10, 50, 30)
high_threshold = st.sidebar.slider("Medium → High threshold (%)", 40, 90, 60)

if low_threshold >= high_threshold:
    st.sidebar.error("Low threshold must be less than High threshold!")
    low_threshold, high_threshold = 30, 60

# Recalculate risk categories
risk_df['Risk_Category'] = risk_df['Risk_Percentage'].apply(
    lambda x: get_risk_cat(x, low_threshold, high_threshold)
)

# Apply filters
filtered_df = risk_df.copy()
if selected_dept != 'All Departments':
    filtered_df = filtered_df[filtered_df['Department'].astype(str) == selected_dept]
if selected_role != 'All Roles':
    filtered_df = filtered_df[filtered_df['JobRole'].astype(str) == selected_role]

# Employee search
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Employee Lookup")
emp_ids = ['-- Select --'] + sorted(filtered_df['EmployeeID'].astype(str).unique().tolist())
selected_employee = st.sidebar.selectbox("Employee ID", emp_ids)

# Sidebar stats
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing:** {len(filtered_df)} / {len(risk_df)} employees")


# ============================================================
# MAIN HEADER
# ============================================================
st.markdown(
    '<div class="main-header">👥 Employee Attrition Risk Intelligence System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Palo Alto Networks | ML-Powered Predictive HR Analytics</div>',
    unsafe_allow_html=True
)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Risk Dashboard",
    "👤 Employee Profile",
    "🏢 Department View",
    "🔬 Explainability",
    "📈 Model Performance"
])


# ============================================================
# TAB 1: RISK DASHBOARD
# ============================================================
with tab1:
    st.header("📊 Attrition Risk Dashboard")

    # --- KPI Row ---
    total = len(filtered_df)
    high = len(filtered_df[filtered_df['Risk_Category'] == 'High Risk'])
    medium = len(filtered_df[filtered_df['Risk_Category'] == 'Medium Risk'])
    low = len(filtered_df[filtered_df['Risk_Category'] == 'Low Risk'])
    avg_risk = filtered_df['Risk_Percentage'].mean() if total > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("👥 Total Employees", f"{total:,}")
    col2.metric("🔴 High Risk", f"{high}", f"{high / total * 100:.1f}%" if total > 0 else "0%")
    col3.metric("🟡 Medium Risk", f"{medium}", f"{medium / total * 100:.1f}%" if total > 0 else "0%")
    col4.metric("🟢 Low Risk", f"{low}", f"{low / total * 100:.1f}%" if total > 0 else "0%")
    col5.metric("📊 Avg Risk Score", f"{avg_risk:.1f}%")

    st.markdown("---")

    # --- Charts Row 1 ---
    col1, col2 = st.columns(2)

    with col1:
        risk_counts = filtered_df['Risk_Category'].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Category Distribution",
            color=risk_counts.index,
            color_discrete_map=RISK_COLORS,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label+value')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            filtered_df, x='Risk_Percentage', nbins=50,
            title="Risk Score Distribution",
            color='Risk_Category',
            color_discrete_map=RISK_COLORS,
            labels={'Risk_Percentage': 'Risk Score (%)'}
        )
        fig.add_vline(x=low_threshold, line_dash="dash", line_color="#f39c12",
                      annotation_text=f"Medium ({low_threshold}%)")
        fig.add_vline(x=high_threshold, line_dash="dash", line_color="#e74c3c",
                      annotation_text=f"High ({high_threshold}%)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Charts Row 2 ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Top High-Risk Employees")
        high_risk_emp = filtered_df[filtered_df['Risk_Category'] == 'High Risk'] \
            .sort_values('Risk_Percentage', ascending=False).head(15)

        if len(high_risk_emp) > 0:
            show_cols = ['EmployeeID', 'Risk_Percentage', 'Risk_Category']
            for c in ['Department', 'JobRole', 'MonthlyIncome', 'OverTime',
                       'JobSatisfaction', 'YearsAtCompany']:
                if c in high_risk_emp.columns:
                    show_cols.append(c)

            st.dataframe(
                high_risk_emp[show_cols].style.background_gradient(
                    subset=['Risk_Percentage'], cmap='Reds'
                ).format({'Risk_Percentage': '{:.1f}%'}),
                use_container_width=True, height=400
            )
        else:
            st.success("✅ No high-risk employees in current filter!")

    with col2:
        if 'JobRole' in filtered_df.columns and len(filtered_df) > 0:
            role_risk = filtered_df.groupby('JobRole')['Risk_Percentage'].agg(
                ['mean', 'count']
            ).sort_values('mean', ascending=True).reset_index()
            role_risk.columns = ['JobRole', 'Avg_Risk', 'Count']

            fig = px.bar(
                role_risk, y='JobRole', x='Avg_Risk', orientation='h',
                title="Average Risk by Job Role",
                color='Avg_Risk', color_continuous_scale='RdYlGn_r',
                text='Count',
                labels={'Avg_Risk': 'Average Risk (%)'}
            )
            fig.update_traces(texttemplate='n=%{text}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # --- Attrition actual vs predicted ---
    if 'Attrition' in filtered_df.columns:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            actual_rate = filtered_df['Attrition'].mean() * 100
            predicted_high = high / total * 100 if total > 0 else 0
            st.metric("Actual Attrition Rate", f"{actual_rate:.1f}%")
        with col2:
            st.metric("Predicted High Risk Rate", f"{predicted_high:.1f}%")


# ============================================================
# TAB 2: EMPLOYEE PROFILE
# ============================================================
with tab2:
    st.header("👤 Employee Risk Profile")

    if selected_employee != '-- Select --':
        emp_data = filtered_df[filtered_df['EmployeeID'].astype(str) == selected_employee]

        if len(emp_data) > 0:
            emp = emp_data.iloc[0]

            # --- Header Section ---
            col1, col2, col3 = st.columns([1.5, 1, 1.5])

            with col1:
                st.markdown(f"### 🧑‍💼 Employee ID: {selected_employee}")
                detail_pairs = [
                    ('Department', '📂'), ('JobRole', '💼'), ('JobLevel', '📊'),
                    ('Age', '🎂'), ('Gender', '👤'), ('MaritalStatus', '💍')
                ]
                for field, icon in detail_pairs:
                    if field in emp.index:
                        st.write(f"{icon} **{field}:** {emp[field]}")

            with col2:
                st.metric("⚠️ Risk Score", f"{emp['Risk_Percentage']}%")
                cat = emp['Risk_Category']
                st.markdown(
                    f"**Category:** {RISK_EMOJI.get(cat, '')} "
                    f"<span class='risk-badge-{'high' if cat == 'High Risk' else 'medium' if cat == 'Medium Risk' else 'low'}'>"
                    f"{cat}</span>",
                    unsafe_allow_html=True
                )

            with col3:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=emp['Risk_Percentage'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Attrition Risk", 'font': {'size': 16}},
                    number={'suffix': '%', 'font': {'size': 28}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#2c3e50"},
                        'steps': [
                            {'range': [0, low_threshold], 'color': '#2ecc71'},
                            {'range': [low_threshold, high_threshold], 'color': '#f39c12'},
                            {'range': [high_threshold, 100], 'color': '#e74c3c'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.8,
                            'value': emp['Risk_Percentage']
                        }
                    }
                ))
                fig.update_layout(height=280, margin=dict(t=60, b=20, l=40, r=40))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # --- Details Grid ---
            st.subheader("📋 Employee Details")

            detail_sections = {
                'Compensation': ['MonthlyIncome', 'DailyRate', 'HourlyRate',
                                 'MonthlyRate', 'PercentSalaryHike', 'StockOptionLevel'],
                'Experience': ['TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
                               'YearsWithCurrManager', 'YearsSinceLastPromotion',
                               'NumCompaniesWorked'],
                'Satisfaction': ['JobSatisfaction', 'EnvironmentSatisfaction',
                                 'RelationshipSatisfaction', 'WorkLifeBalance',
                                 'JobInvolvement'],
                'Other': ['OverTime', 'BusinessTravel', 'DistanceFromHome',
                          'Education', 'EducationField', 'TrainingTimesLastYear',
                          'PerformanceRating']
            }

            for section, fields in detail_sections.items():
                available = [f for f in fields if f in emp.index]
                if available:
                    with st.expander(f"📁 {section}", expanded=True):
                        cols = st.columns(len(available) if len(available) <= 4 else 4)
                        for i, field in enumerate(available):
                            cols[i % len(cols)].metric(field, emp[field])

            # --- Engineered Features ---
            st.markdown("---")
            st.subheader("🧪 Engineered Features")
            eng_features = ['IncomePerYear', 'EngagementScore', 'WorkloadStress',
                            'TenureRatio', 'RoleStagnation', 'CompanyHopper',
                            'IsOverduePromotion', 'SatisfactionIncomeRatio', 'PromotionDelay']
            available_eng = [f for f in eng_features if f in emp.index]

            if available_eng:
                cols = st.columns(min(len(available_eng), 5))
                for i, f in enumerate(available_eng):
                    cols[i % len(cols)].metric(f, f"{emp[f]:.2f}" if isinstance(emp[f], float) else emp[f])

            # --- Contributing Factors ---
            st.markdown("---")
            st.subheader("📊 Key Contributing Factors")

            factor_data = []
            for _, row in feature_importance.head(12).iterrows():
                feat = row['Feature']
                if feat in emp.index:
                    factor_data.append({
                        'Feature': feat,
                        'Importance': row['Importance'],
                        'Employee Value': emp[feat]
                    })

            if factor_data:
                factor_df = pd.DataFrame(factor_data)
                fig = px.bar(
                    factor_df, x='Importance', y='Feature', orientation='h',
                    title="Top Feature Importances (with this employee's values)",
                    color='Importance', color_continuous_scale='Reds',
                    text='Employee Value',
                    height=400
                )
                fig.update_traces(texttemplate='Val: %{text}', textposition='outside')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Employee not found in the filtered data.")
    else:
        st.info("👈 Select an Employee ID from the sidebar to view their risk profile.")

        # Show quick stats
        st.markdown("---")
        st.subheader("📊 Quick Employee Overview")
        col1, col2 = st.columns(2)
        with col1:
            if 'MonthlyIncome' in filtered_df.columns:
                fig = px.histogram(filtered_df, x='MonthlyIncome', color='Risk_Category',
                                   color_discrete_map=RISK_COLORS,
                                   title="Income Distribution by Risk Category")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'Age' in filtered_df.columns:
                fig = px.histogram(filtered_df, x='Age', color='Risk_Category',
                                   color_discrete_map=RISK_COLORS,
                                   title="Age Distribution by Risk Category")
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 3: DEPARTMENT VIEW
# ============================================================
with tab3:
    st.header("🏢 Department-Level Risk Analysis")

    if 'Department' in risk_df.columns:
        # --- Department Summary Table ---
        dept_summary = risk_df.groupby('Department').agg(
            Total=('EmployeeID', 'count'),
            Avg_Risk=('Risk_Percentage', 'mean'),
            Median_Risk=('Risk_Percentage', 'median'),
            Max_Risk=('Risk_Percentage', 'max'),
            High_Risk=('Risk_Category', lambda x: (x == 'High Risk').sum()),
            Medium_Risk=('Risk_Category', lambda x: (x == 'Medium Risk').sum()),
            Low_Risk=('Risk_Category', lambda x: (x == 'Low Risk').sum())
        ).round(2).reset_index()

        dept_summary['High_Risk_%'] = (dept_summary['High_Risk'] / dept_summary['Total'] * 100).round(1)

        st.subheader("📊 Department Summary")
        st.dataframe(
            dept_summary.style
            .background_gradient(subset=['Avg_Risk'], cmap='RdYlGn_r')
            .background_gradient(subset=['High_Risk'], cmap='Reds')
            .format({
                'Avg_Risk': '{:.1f}%',
                'Median_Risk': '{:.1f}%',
                'Max_Risk': '{:.1f}%',
                'High_Risk_%': '{:.1f}%'
            }),
            use_container_width=True
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(
                risk_df, x='Department', y='Risk_Percentage',
                title="Risk Score Distribution by Department",
                color='Department', points='outliers',
                labels={'Risk_Percentage': 'Risk Score (%)'}
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            dept_risk_counts = risk_df.groupby(
                ['Department', 'Risk_Category']
            ).size().reset_index(name='Count')
            fig = px.bar(
                dept_risk_counts, x='Department', y='Count',
                color='Risk_Category', barmode='stack',
                title="Risk Category Breakdown by Department",
                color_discrete_map=RISK_COLORS
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        # --- Role-level analysis ---
        if 'JobRole' in risk_df.columns:
            st.markdown("---")
            st.subheader("🗺️ Department → Role Risk Map")

            role_dept = risk_df.groupby(['Department', 'JobRole']).agg(
                Count=('EmployeeID', 'count'),
                Avg_Risk=('Risk_Percentage', 'mean'),
                High_Risk=('Risk_Category', lambda x: (x == 'High Risk').sum())
            ).round(2).reset_index()

            fig = px.treemap(
                role_dept, path=['Department', 'JobRole'],
                values='Count', color='Avg_Risk',
                color_continuous_scale='RdYlGn_r',
                title="Department → Role Risk Treemap (size=headcount, color=avg risk)"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Heatmap
            st.subheader("🌡️ Risk Heatmap: Department × Role")
            pivot = risk_df.pivot_table(
                values='Risk_Percentage', index='JobRole',
                columns='Department', aggfunc='mean'
            ).round(1)

            fig = px.imshow(
                pivot, text_auto='.1f', aspect='auto',
                color_continuous_scale='RdYlGn_r',
                title="Average Risk Score: Job Role × Department",
                labels=dict(color="Avg Risk %")
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Department column not found in the dataset.")


# ============================================================
# TAB 4: EXPLAINABILITY
# ============================================================
with tab4:
    st.header("🔬 Model Explainability")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Feature Importance (Top 15)")
        top15 = feature_importance.head(15).copy()
        fig = px.bar(
            top15, x='Importance', y='Feature', orientation='h',
            title="Top 15 Most Important Features for Prediction",
            color='Importance', color_continuous_scale='Viridis',
            labels={'Importance': 'Importance Score'}
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📋 Full Feature Ranking")
        st.dataframe(
            feature_importance.style.background_gradient(
                subset=['Importance'], cmap='YlOrRd'
            ).format({'Importance': '{:.4f}'}),
            use_container_width=True, height=500
        )

    # --- SHAP Images ---
    st.markdown("---")
    st.subheader("🎯 SHAP Analysis (Pre-computed)")

    shap_files = {
        'SHAP Summary Plot': 'shap_summary.png',
        'SHAP Bar Plot': 'shap_bar.png',
        'Individual Explanation': 'shap_individual.png'
    }

    shap_cols = st.columns(len(shap_files))
    for i, (title, filename) in enumerate(shap_files.items()):
        filepath = get_path(filename)
        with shap_cols[i]:
            if os.path.exists(filepath):
                st.image(filepath, caption=title, use_container_width=True)
            else:
                st.info(f"{title}: Not generated yet")

    # --- What-If Scenario Explorer ---
    st.markdown("---")
    st.subheader("🔮 What-If Scenario Explorer")
    st.write("Adjust parameters to explore how changes affect attrition risk:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        wi_age = st.slider("Age", 18, 65, 35, key="wi_age")
        wi_income = st.number_input("Monthly Income ($)", 1000, 25000, 5000, step=500, key="wi_income")
        wi_distance = st.slider("Distance from Home (mi)", 1, 30, 10, key="wi_dist")

    with col2:
        wi_job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3, key="wi_jsat")
        wi_env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 3, key="wi_esat")
        wi_wlb = st.slider("Work-Life Balance (1-4)", 1, 4, 3, key="wi_wlb")

    with col3:
        wi_overtime = st.selectbox("Overtime", ['No', 'Yes'], key="wi_ot")
        wi_years_company = st.slider("Years at Company", 0, 40, 5, key="wi_yc")
        wi_years_promo = st.slider("Years Since Promotion", 0, 15, 2, key="wi_yp")

    with col4:
        wi_total_years = st.slider("Total Working Years", 0, 40, 10, key="wi_tw")
        wi_num_companies = st.slider("Num Companies Worked", 0, 10, 2, key="wi_nc")
        wi_job_level = st.slider("Job Level (1-5)", 1, 5, 2, key="wi_jl")

    if st.button("🔮 Estimate Risk", type="primary"):
        # Heuristic-based estimation for the what-if tool
        risk_score = 15  # baseline

        if wi_overtime == 'Yes':
            risk_score += 18
        if wi_job_sat <= 1:
            risk_score += 15
        elif wi_job_sat <= 2:
            risk_score += 8
        if wi_env_sat <= 1:
            risk_score += 12
        elif wi_env_sat <= 2:
            risk_score += 6
        if wi_wlb <= 1:
            risk_score += 12
        elif wi_wlb <= 2:
            risk_score += 6
        if wi_years_promo >= 7:
            risk_score += 15
        elif wi_years_promo >= 5:
            risk_score += 8
        if wi_distance > 20:
            risk_score += 10
        elif wi_distance > 15:
            risk_score += 5
        if wi_income < 3000:
            risk_score += 12
        elif wi_income < 5000:
            risk_score += 5
        if wi_years_company <= 1:
            risk_score += 10
        elif wi_years_company <= 2:
            risk_score += 5
        if wi_num_companies > 5:
            risk_score += 8
        if wi_age < 25:
            risk_score += 5
        if wi_job_level <= 1:
            risk_score += 5

        # Cap score
        risk_score = min(max(risk_score, 5), 95)

        cat = get_risk_cat(risk_score, low_threshold, high_threshold)

        st.markdown("---")
        result_col1, result_col2 = st.columns([1, 2])
        with result_col1:
            st.metric("Estimated Risk", f"{risk_score}%")
            st.markdown(f"**Category:** {RISK_EMOJI.get(cat, '')} **{cat}**")

        with result_col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2c3e50"},
                    'steps': [
                        {'range': [0, low_threshold], 'color': '#2ecc71'},
                        {'range': [low_threshold, high_threshold], 'color': '#f39c12'},
                        {'range': [high_threshold, 100], 'color': '#e74c3c'}
                    ]
                }
            ))
            fig.update_layout(height=250, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Breakdown
        st.subheader("Contributing Factors:")
        factors = []
        if wi_overtime == 'Yes': factors.append("⬆️ Working overtime (+18%)")
        if wi_job_sat <= 2: factors.append(f"⬆️ Low job satisfaction: {wi_job_sat}/4")
        if wi_env_sat <= 2: factors.append(f"⬆️ Low environment satisfaction: {wi_env_sat}/4")
        if wi_wlb <= 2: factors.append(f"⬆️ Poor work-life balance: {wi_wlb}/4")
        if wi_years_promo >= 5: factors.append(f"⬆️ No promotion in {wi_years_promo} years")
        if wi_income < 5000: factors.append(f"⬆️ Low monthly income: ${wi_income}")
        if wi_distance > 15: factors.append(f"⬆️ Long commute: {wi_distance} miles")
        if wi_years_company <= 2: factors.append(f"⬆️ Short tenure: {wi_years_company} years")

        if wi_job_sat >= 4: factors.append("⬇️ High job satisfaction")
        if wi_income >= 10000: factors.append("⬇️ High compensation")
        if wi_years_company >= 10: factors.append("⬇️ Long tenure (loyal employee)")
        if wi_job_level >= 4: factors.append("⬇️ Senior position")

        for f in factors:
            st.write(f"  {f}")


# ============================================================
# TAB 5: MODEL PERFORMANCE
# ============================================================
with tab5:
    st.header("📈 Model Performance Comparison")

    # --- Metrics Table ---
    st.subheader("📊 Performance Metrics")
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    available_metrics = [m for m in metrics_cols if m in model_comparison.columns]

    st.dataframe(
        model_comparison[available_metrics].style
        .highlight_max(axis=0, color='#a8e6cf')
        .highlight_min(axis=0, color='#ffcdd2')
        .format("{:.4f}"),
        use_container_width=True
    )

    # Best model callout
    best = model_comparison['ROC-AUC'].idxmax()
    best_auc = model_comparison.loc[best, 'ROC-AUC']
    st.success(f"🏆 **Best Model:** {best} with ROC-AUC = {best_auc:.4f}")

    st.markdown("---")

    # --- Bar Chart ---
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        model_colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']
        for i, model_name in enumerate(model_comparison.index):
            fig.add_trace(go.Bar(
                name=model_name,
                x=available_metrics,
                y=[model_comparison.loc[model_name, m] for m in available_metrics],
                marker_color=model_colors[i % len(model_colors)]
            ))
        fig.update_layout(
            title="Model Performance Comparison",
            barmode='group', yaxis_title="Score",
            yaxis_range=[0, 1.1], height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Radar chart
        fig = go.Figure()
        for i, model_name in enumerate(model_comparison.index):
            values = [model_comparison.loc[model_name, m] for m in available_metrics]
            values.append(values[0])  # close the polygon
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics + [available_metrics[0]],
                fill='toself',
                name=model_name,
                opacity=0.6
            ))
        fig.update_layout(
            title="Model Performance Radar Chart",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Pre-computed Visuals ---
    st.markdown("---")
    st.subheader("📸 Detailed Evaluation Plots")

    eval_images = {
        'Model Comparison': 'model_comparison.png',
        'Confusion Matrices': 'confusion_matrices.png',
        'Feature Importance': 'feature_importance.png'
    }

    img_cols = st.columns(len(eval_images))
    for i, (title, filename) in enumerate(eval_images.items()):
        filepath = get_path(filename)
        with img_cols[i]:
            if os.path.exists(filepath):
                st.image(filepath, caption=title, use_container_width=True)
            else:
                st.info(f"{title}: Run analysis script first")

    # --- Recommendations ---
    st.markdown("---")
    st.subheader("🔑 Recommendations for HR Leadership")

    st.markdown("""
    | Priority | Action | Timeline |
    |----------|--------|----------|
    | 🔴 **Critical** | Conduct stay interviews with all High Risk employees | This week |
    | 🔴 **Critical** | Review compensation for underpaid high-risk employees | 2 weeks |
    | 🟡 **Important** | Implement targeted retention programs for Medium Risk | 1 month |
    | 🟡 **Important** | Address overtime policies and workload distribution | 1 month |
    | 🟢 **Strategic** | Establish quarterly risk assessment cadence | Quarterly |
    | 🟢 **Strategic** | Build manager accountability for team retention | Ongoing |
    | 🟢 **Strategic** | Create personalized career development plans | 3 months |
    """)

    # Download risk scores
    st.markdown("---")
    st.subheader("📥 Download Reports")
    col1, col2, col3 = st.columns(3)

    with col1:
        csv_risk = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Risk Scores (CSV)",
            csv_risk,
            file_name="employee_risk_scores.csv",
            mime="text/csv"
        )

    with col2:
        csv_fi = feature_importance.to_csv(index=False)
        st.download_button(
            "📥 Download Feature Importance (CSV)",
            csv_fi,
            file_name="feature_importance.csv",
            mime="text/csv"
        )

    with col3:
        high_risk_csv = risk_df[risk_df['Risk_Category'] == 'High Risk'].to_csv(index=False)
        st.download_button(
            "📥 Download High-Risk List (CSV)",
            high_risk_csv,
            file_name="high_risk_employees.csv",
            mime="text/csv"
        )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 1rem 0;'>
        <strong>Employee Attrition Risk Prediction System</strong><br>
        Palo Alto Networks — HR Analytics Division<br>
        <br>
        <small>© 2024 | Machine Learning–Based Predictive Intelligence</small>
    </div>
    """,
    unsafe_allow_html=True
)