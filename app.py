import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.talk_to_data.nl_to_sql import TalkToData
from src.ml.predict import RiskPredictor
from src.ml.rules import BusinessRulesEngine, RuleDeriver
from src.ui.eda_view import render_eda_dashboard

# Page Config (Must be the very first Streamlit command)
st.set_page_config(page_title="NeoStat Credit Risk Platform", page_icon="🏦", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00FFA3;
    }
    .risk-high { color: #EF4444 !important; }
    .risk-medium { color: #F59E0B !important; }
    .risk-low { color: #10B981 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Caching Heavy Resources
# ---------------------------------------------------------
@st.cache_resource
def load_predictor():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    return RiskPredictor(models_dir=models_dir)

@st.cache_resource
def load_chatbot():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_actual_api_key_here":
        return None
    db_path = os.path.join(os.path.dirname(__file__), 'credit_risk.db')
    model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-3.5-flash-lite")
    return TalkToData(api_key=api_key, model_name=model_name, db_path=db_path)

@st.cache_data
def get_base_applicant():
    """Reads a sample of the raw dataset to create a median 'base' profile for missing features."""
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'application_train.csv')
    try:
        df = pd.read_csv(data_path, nrows=5000)
        # Create a single dictionary with medians for numeric and mode for categorical
        base_profile = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                base_profile[col] = df[col].median()
            else:
                base_profile[col] = df[col].mode()[0] if not df[col].mode().empty else ""
        return base_profile
    except Exception as e:
        st.error(f"Failed to load base applicant profile: {e}")
        return {}

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("🏦 NeoStat")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["📊 Exploratory Data Analysis", "🔮 Risk Prediction", "⚙️ Business Rules", "💬 Talk to Data"])

# ---------------------------------------------------------
# Page 1: Exploratory Data Analysis
# ---------------------------------------------------------
if page == "📊 Exploratory Data Analysis":
    render_eda_dashboard()

# ---------------------------------------------------------
# Page 2: Risk Prediction
# ---------------------------------------------------------
elif page == "🔮 Risk Prediction":
    st.title("Risk Prediction & Explainability")
    st.markdown("Enter client details below to generate a real-time risk assessment powered by XGBoost & SHAP.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Applicant Profile")
        
        # Profile Preset Selector
        preset = st.selectbox(
            "Quick Profile Preset", 
            [
                "🟡 Average Profile (Default Bureau Scores ~0.50)",
                "🟢 Prime / Low Risk (High Bureau Scores ~0.80)",
                "🔴 Subprime / High Risk (Low Bureau Scores ~0.15)",
                "Custom"
            ]
        )
        
        if "Prime" in preset:
            init_income = 120000
            init_credit = 300000
            init_age = 42
            init_emp = 10.0
            init_edu = "Higher education"
            init_ext1, init_ext2, init_ext3 = 0.78, 0.82, 0.80
        elif "Subprime" in preset:
            init_income = 35000
            init_credit = 200000
            init_age = 28
            init_emp = 1.5
            init_edu = "Lower secondary"
            init_ext1, init_ext2, init_ext3 = 0.12, 0.15, 0.10
        else:
            init_income = 50000
            init_credit = 200000
            init_age = 35
            init_emp = 5.0
            init_edu = "Secondary / secondary special"
            init_ext1, init_ext2, init_ext3 = 0.51, 0.56, 0.53

        with st.form("prediction_form"):
            st.markdown("##### 💵 Financial & Demographic Details")
            income = st.number_input("Total Income ($)", min_value=0, value=init_income, step=5000)
            credit = st.number_input("Credit Amount Requested ($)", min_value=0, value=init_credit, step=10000)
            
            # Days are negative in dataset (days BEFORE application). UI takes years for UX.
            age_years = st.number_input("Age (Years)", min_value=18, max_value=100, value=init_age)
            emp_years = st.number_input("Years Employed", min_value=0.0, max_value=50.0, value=init_emp)
            
            edu_options = ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"]
            edu_index = edu_options.index(init_edu) if init_edu in edu_options else 0
            education = st.selectbox("Education Level", edu_options, index=edu_index)
            gender = st.selectbox("Gender", ["M", "F"])
            own_car = st.selectbox("Owns Car?", ["Y", "N"])
            own_realty = st.selectbox("Owns Real Estate?", ["Y", "N"])

            st.markdown("##### 📈 External Bureau Credit Scores (Key Risk Drivers)")
            st.caption("Normalized scores from 3 external credit rating agencies (0.00 = High Risk / Delinquent, 1.00 = Prime / Low Risk).")
            
            ext_source_1 = st.slider("Bureau Score 1 (EXT_SOURCE_1)", 0.00, 1.00, value=init_ext1, step=0.01, help="External rating agency 1 score")
            ext_source_2 = st.slider("Bureau Score 2 (EXT_SOURCE_2)", 0.00, 1.00, value=init_ext2, step=0.01, help="External rating agency 2 score")
            ext_source_3 = st.slider("Bureau Score 3 (EXT_SOURCE_3)", 0.00, 1.00, value=init_ext3, step=0.01, help="External rating agency 3 score")
            
            submitted = st.form_submit_button("Generate Prediction", use_container_width=True)
            
    with col2:
        if submitted:
            with st.spinner("Analyzing applicant profile..."):
                predictor = load_predictor()
                base_profile = get_base_applicant()
                
                # Override base profile with UI inputs
                applicant = base_profile.copy()
                applicant.update({
                    'AMT_INCOME_TOTAL': income,
                    'AMT_CREDIT': credit,
                    'DAYS_BIRTH': -(age_years * 365),
                    'DAYS_EMPLOYED': -(emp_years * 365) if emp_years > 0 else 365243, # 365243 is unemployed/pensioner flag
                    'NAME_EDUCATION_TYPE': education,
                    'CODE_GENDER': gender,
                    'FLAG_OWN_CAR': own_car,
                    'FLAG_OWN_REALTY': own_realty,
                    'EXT_SOURCE_1': ext_source_1,
                    'EXT_SOURCE_2': ext_source_2,
                    'EXT_SOURCE_3': ext_source_3
                })
                
                # Predict
                df_input = pd.DataFrame([applicant])
                try:
                    result = predictor.predict(df_input).iloc[0]
                    
                    # Display Top-Level Metrics
                    st.subheader("Assessment Results")
                    c1, c2, c3 = st.columns(3)
                    
                    risk_color_class = "risk-low" if result['Risk_Band'] == "Low" else "risk-medium" if result['Risk_Band'] == "Medium" else "risk-high"
                    norm_prob = result.get('Normalized_Default_Probability', result['Default_Probability'])
                    raw_prob = result.get('Raw_Default_Probability', result['Default_Probability'])
                    rel_risk = result.get('Relative_Risk', round(raw_prob / 0.0807, 1))
                    percentile = result.get('Portfolio_Percentile', 50.0)
                    decision = 'AUTO REJECT' if 'Rule Reject' in result['Risk_Band'] else 'DECLINE' if result['Risk_Band'] == 'High' else 'MANUAL REVIEW' if result['Risk_Band'] == 'Medium' else 'AUTO APPROVE'

                    c1.markdown(f"""<div class="metric-card">
                        <div>Normalized Default Risk</div>
                        <div class="metric-value">{norm_prob:.1%}</div>
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 5px;">
                            Raw Statistical PD: <b>{raw_prob:.1%}</b>
                        </div>
                        </div>""", unsafe_allow_html=True)
                        
                    c2.markdown(f"""<div class="metric-card">
                        <div>Calibrated Risk Score</div>
                        <div class="metric-value">{result['Risk_Score']}/1000</div>
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 5px;">
                            <b>{percentile}th</b> Portfolio Percentile
                        </div>
                        </div>""", unsafe_allow_html=True)
                        
                    c3.markdown(f"""<div class="metric-card">
                        <div>Risk Tier</div>
                        <div class="metric-value {risk_color_class}">{result['Risk_Band']}</div>
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 5px;">
                            Recommendation: <b>{decision}</b>
                        </div>
                        </div>""", unsafe_allow_html=True)
                        
                    st.info(f"📊 **Normalized Scoring Model Active**: The probability display is normalized across the portfolio distribution. The raw statistical rate (**{raw_prob:.1%}**) is scaled to **{norm_prob:.1%} Normalized Default Risk** (where 0% = safest portfolio tier, 100% = highest default risk), reflecting **{rel_risk}× higher risk** than the 8.07% population baseline.")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Explainability (SHAP)
                    st.subheader("Explainable AI Insights (SHAP)")
                    if "Rule Reject" in result['Risk_Band']:
                        st.error(f"🛑 **REJECTED BY BUSINESS RULE:** {result['Top_Risk_Drivers']}")
                    else:
                        st.markdown("#### 🚨 Top Risk Drivers (Pushed Risk UP)")
                        for driver in result['Top_Risk_Drivers'].split(' | '):
                            if driver: st.error(driver)
                            
                        st.markdown("#### ✅ Top Mitigating Factors (Pushed Risk DOWN)")
                        for mitigator in result['Top_Mitigating_Factors'].split(' | '):
                            if mitigator: st.success(mitigator)
                            
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.info("👈 Fill out the applicant profile and click 'Generate Prediction'.")

# ---------------------------------------------------------
# Page 3: Business Rules
# ---------------------------------------------------------
elif page == "⚙️ Business Rules":
    st.title("Business Rules & Underwriting Policy Engine")
    st.markdown("Combines **deterministic pre-screening policies** (hard rejections) with **automated empirical rule derivation** (interpretable surrogate decision trees).")

    tab_hard, tab_derived = st.tabs(["🛡️ Pre-Screening Hard Rules", "🌳 Automated Rule Derivation (Mined Rules)"])

    with tab_hard:
        st.markdown("#### 🛡️ Pre-Screening Policy Overrides")
        st.markdown("These hard rules intercept applicants *before* ML scoring. If an applicant violates any rule, they are automatically flagged as **High Risk (Rule Reject)** to safeguard capital.")
        
        rules_engine = BusinessRulesEngine()
        for rule_name, rule_logic in rules_engine.rules.items():
            st.warning(f"**{rule_name}**\n\n{rule_logic['description']}")
            
    with tab_derived:
        st.markdown("#### 🌳 Automated Rule Mining via Surrogate Decision Trees")
        st.markdown("""
        To satisfy regulatory interpretability and auditability, this module fits a constrained surrogate tree on portfolio risk drivers
        (`EXT_SOURCE_2`, `EXT_SOURCE_3`, `Age`, `Credit-to-Income`) to automatically derive **deterministic IF-THEN underwriting rules**.
        """)
        
        rule_deriver = RuleDeriver()
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Portfolio Baseline Default Rate", "8.07%", "Population benchmark")
        col_m2.metric("Mined Rule Segments", "5 Tiers", "High to Prime")
        col_m3.metric("Max Segment Lift", "2.95×", "23.8% default vs 8.1%")
        
        st.markdown("---")
        
        # Display each mined rule
        rules_list = rule_deriver.get_precomputed_rules()
        for rule in rules_list:
            is_high = "High" in rule['risk_tier']
            is_elev = "Elevated" in rule['risk_tier']
            is_mod = "Moderate" in rule['risk_tier']
            badge_color = "#EF4444" if is_high else "#F59E0B" if is_elev else "#3B82F6" if is_mod else "#10B981"
            
            with st.container():
                c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
                c1.markdown(f"##### `[{rule['rule_id']}]` <span style='color:{badge_color}; font-weight:bold;'>{rule['risk_tier']}</span>", unsafe_allow_html=True)
                c1.markdown(f"**Condition**: `{rule['condition']}`")
                c1.caption(rule['description'])
                
                c2.metric("Default Rate", f"{rule['default_rate']:.1%}")
                c3.metric("Risk Lift", f"{rule['lift']}×")
                c4.markdown(f"**Underwriting Action**:\n\n<span style='color:{badge_color}; font-weight:bold;'>{rule['action']}</span>", unsafe_allow_html=True)
                c4.caption(f"Support: {rule['support_count']:,} loans ({rule['support_pct']}%)")
                st.markdown("---")

# ---------------------------------------------------------
# Page 4: Talk to Data
# ---------------------------------------------------------
elif page == "💬 Talk to Data":
    chatbot = load_chatbot()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Talk to Data (Natural Language SQL)")
        st.markdown("Ask natural language questions about your SQLite database.")
    with col2:
        if chatbot:
            stats = chatbot.get_rate_limit_status()
            st.metric("Live Quota (RPM)", f"{stats['rpm_current']}/{stats['rpm_max']}", help="Requests per minute sliding window")

    with st.expander("🛡️ Active Safeguards & Guardrail Policies"):
        st.markdown("""
        * **Rate Limiting**: Sliding-window limiter strictly enforcing **15 RPM** & **500 RPD** with automatic cooldown.
        * **Input Guardrails**: Prompt injection detector & query length validator (3–600 characters).
        * **SQL Guardrails**: Read-only enforcement (`SELECT`/`WITH` CTE only); destructive commands (`DROP`, `DELETE`, `UPDATE`, etc.) and multi-statement injections are blocked.
        * **Database Protection**: Unconstrained queries automatically capped at `LIMIT 100` with a 5.0s query timeout.
        """)
    
    if chatbot is None:
        st.error("GEMINI_API_KEY is missing. Please set it in your `.env` file.")
    else:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sql"):
                    with st.expander("🔍 View Executed SQL Query"):
                        st.code(message["sql"], language="sql")

        # Quick Exploration Starter Chips
        st.markdown("##### 💡 Suggested Questions")
        chip_cols = st.columns(4)
        starter_queries = [
            ("💰 Defaulter Income", "What is the average income of clients who defaulted?"),
            ("📄 Contract Types", "What are the top 5 contract types with highest default rates?"),
            ("⚖️ Gender Risk", "Compare default rate between men and women"),
            ("🎓 Education Scores", "What is the average bureau score (EXT_SOURCE_2) by education level?")
        ]
        
        selected_prompt = None
        for i, (label, query_text) in enumerate(starter_queries):
            if chip_cols[i].button(label, key=f"chip_btn_{i}", use_container_width=True, help=query_text):
                selected_prompt = query_text

        # Accept user input from text box OR starter chip
        user_input = st.chat_input("E.g., What is the average income of clients who defaulted?")
        prompt = user_input or selected_prompt

        if prompt:
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Writing SQL and querying database..."):
                    res = chatbot.ask(prompt, return_dict=True)
                
                answer_text = res["answer"] if isinstance(res, dict) else str(res)
                sql_text = res.get("sql") if isinstance(res, dict) else None
                
                st.markdown(answer_text)
                if sql_text:
                    with st.expander("🔍 View Executed SQL Query"):
                        st.code(sql_text, language="sql")
                        
            # Add to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sql": sql_text
            })

