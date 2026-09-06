import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Custom Dark Plotly Template Palette
DARK_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "#0F172A",
    "plot_bgcolor": "#1E293B",
    "font": {"family": "Inter, system-ui, sans-serif", "color": "#E2E8F0"},
    "margin": dict(l=30, r=30, t=50, b=30),
}

COLOR_PAID = "#10B981"      # Emerald Green
COLOR_DEFAULT = "#EF4444"   # Crimson Red
COLOR_ACCENT = "#6366F1"    # Indigo
COLOR_AMBER = "#F59E0B"     # Amber
COLOR_CYAN = "#06B6D4"      # Cyan

@st.cache_data(show_spinner=False)
def load_eda_dataset(sample_size: int = 30000):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_parquet = os.path.join(base_dir, 'data', 'eda_sample.parquet')
    parquet_path = os.path.join(base_dir, 'data', 'merged_data.parquet')
    csv_path = os.path.join(base_dir, 'data', 'application_train.csv')
    
    if os.path.exists(sample_parquet):
        df = pd.read_parquet(sample_parquet)
    elif os.path.exists(parquet_path):
        # Read parquet directly for speed and merged feature richness
        df = pd.read_parquet(parquet_path)
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, nrows=sample_size)
    else:
        return None

    # Derive human-readable feature transformations for EDA
    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = (df['DAYS_BIRTH'].abs() / 365.25).round(1)
        df['AGE_GROUP'] = pd.cut(
            df['AGE_YEARS'], 
            bins=[18, 25, 35, 45, 55, 65, 100], 
            labels=['18-25', '25-35', '35-45', '45-55', '55-65', '65+'],
            right=False
        )
    
    if 'DAYS_EMPLOYED' in df.columns:
        df['EMPLOYED_YEARS'] = np.where(df['DAYS_EMPLOYED'] == 365243, np.nan, df['DAYS_EMPLOYED'].abs() / 365.25)
        df['IS_PENSIONER_ANOM'] = np.where(df['DAYS_EMPLOYED'] == 365243, 'Pensioner / Anom (365k)', 'Active Workforce')

    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_INCOME_RATIO'] = (df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)).clip(0, 1)
    
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['CREDIT_INCOME_RATIO'] = (df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)).clip(0, 20)

    if 'AMT_CREDIT' in df.columns and 'AMT_GOODS_PRICE' in df.columns:
        df['OVER_FINANCING_RATIO'] = (df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'].replace(0, np.nan)).clip(0.5, 2.5)

    df['TARGET_LABEL'] = df['TARGET'].map({0: 'Non-Default (Paid)', 1: 'Defaulted'})
    return df


def render_eda_dashboard():
    """Renders the complete interactive EDA dashboard in Streamlit."""
    st.title("📊 Exploratory Data Analysis & Portfolio Risk Profiling")
    st.markdown(
        "Deep exploration into **307,511 loan applications**, external credit bureau telemetry, "
        "demographic risk cohorts, and cross-table repayment signals."
    )

    with st.spinner("Loading analytical portfolio sample..."):
        df = load_eda_dataset(sample_size=35000)

    if df is None:
        st.error("Dataset not found in `data/merged_data.parquet` or `data/application_train.csv`.")
        return

    # ---------------------------------------------------------
    # 1. Executive KPI Summary Row
    # ---------------------------------------------------------
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    total_records = 307511
    sample_records = len(df)
    default_rate = df['TARGET'].mean() * 100
    median_credit = df['AMT_CREDIT'].median()
    median_income = df['AMT_INCOME_TOTAL'].median()
    median_dti = (df['ANNUITY_INCOME_RATIO'].median() * 100) if 'ANNUITY_INCOME_RATIO' in df.columns else 0.0

    kpi1.metric("📁 Total Portfolio", f"{total_records:,}", help="Full Home Credit training set size")
    kpi2.metric("⚠️ Baseline Default Rate", f"{default_rate:.2f}%", delta="-91.93% Repaid", delta_color="inverse")
    kpi3.metric("💳 Median Credit Limit", f"${median_credit:,.0f}")
    kpi4.metric("💵 Median Income", f"${median_income:,.0f}")
    kpi5.metric("⚖️ Median DTI Burden", f"{median_dti:.1f}%", help="Annual Loan Annuity / Total Annual Income")

    st.markdown("---")

    # ---------------------------------------------------------
    # 2. Six Interactive EDA Tabs
    # ---------------------------------------------------------
    tabs = st.tabs([
        "📊 Portfolio & Data Quality",
        "🎯 External Bureau Scores",
        "👥 Demographics & Socio-Economics",
        "💰 Financial Health & Ratios",
        "🏦 Multi-Table Repayment Behavior",
        "🔬 Correlation & Feature Explorer"
    ])

    # ---------------------------------------------------------
    # TAB 1: Portfolio Overview & Data Quality
    # ---------------------------------------------------------
    with tabs[0]:
        st.subheader("Portfolio Distribution & Data Quality Diagnostics")
        col_t1_a, col_t1_b = st.columns([1, 1])

        with col_t1_a:
            st.markdown("##### Target Class Imbalance (Default vs Paid)")
            target_counts = df['TARGET_LABEL'].value_counts().reset_index()
            target_counts.columns = ['Status', 'Count']
            
            fig_donut = px.pie(
                target_counts, 
                names='Status', 
                values='Count', 
                hole=0.55,
                color='Status',
                color_discrete_map={'Non-Default (Paid)': COLOR_PAID, 'Defaulted': COLOR_DEFAULT}
            )
            fig_donut.update_layout(**DARK_LAYOUT, height=360, showlegend=True)
            fig_donut.update_traces(textposition='outside', textinfo='percent+label+value')
            st.plotly_chart(fig_donut, use_container_width=True)
            st.caption("ℹ️ The ~8% default rate represents a severe class imbalance requiring class weighting (`scale_pos_weight`) and ROC-AUC / PR-AUC evaluation.")

        with col_t1_b:
            st.markdown("##### Loan Contract Type Distribution by Default")
            if 'NAME_CONTRACT_TYPE' in df.columns:
                contract_df = df.groupby(['NAME_CONTRACT_TYPE', 'TARGET_LABEL']).size().reset_index(name='Count')
                fig_contract = px.bar(
                    contract_df, 
                    x='NAME_CONTRACT_TYPE', 
                    y='Count', 
                    color='TARGET_LABEL',
                    barmode='group',
                    color_discrete_map={'Non-Default (Paid)': COLOR_PAID, 'Defaulted': COLOR_DEFAULT}
                )
                fig_contract.update_layout(**DARK_LAYOUT, height=360, xaxis_title="Contract Type", yaxis_title="Application Count")
                st.plotly_chart(fig_contract, use_container_width=True)
                st.caption("Cash loans comprise ~90% of applications, whereas Revolving loans represent short-term revolving credit lines.")

        st.markdown("##### Missing Value Profile & Imputation Thresholds")
        thresh = st.slider("Filter features by minimum missing percentage threshold (%):", 0, 80, 40, 5)
        
        missing_pct = (df.isnull().sum() / len(df) * 100)
        missing_filtered = missing_pct[missing_pct >= thresh].sort_values(ascending=True).tail(25)
        
        if len(missing_filtered) > 0:
            missing_df = pd.DataFrame({'Feature': missing_filtered.index, 'MissingPercentage': missing_filtered.values})
            fig_miss = px.bar(
                missing_df, 
                x='MissingPercentage', 
                y='Feature', 
                orientation='h',
                color='MissingPercentage',
                color_continuous_scale=['#3B82F6', '#EF4444'],
                labels={'MissingPercentage': 'Missing Data (%)', 'Feature': 'Dataset Column'}
            )
            fig_miss.update_layout(**DARK_LAYOUT, height=450, coloraxis_showscale=False)
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success(f"No features found with missing percentage above {thresh}%.")

    # ---------------------------------------------------------
    # TAB 2: External Credit Bureau Scores
    # ---------------------------------------------------------
    with tabs[1]:
        st.subheader("External Credit Bureau Scores (Strongest Risk Drivers)")
        st.markdown(
            "The Home Credit dataset includes 3 normalized external credit agency scores: "
            "`EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3`. Higher scores represent prime creditworthiness, "
            "while low scores strongly signal high default likelihood."
        )

        col_bureau_1, col_bureau_2 = st.columns([1, 1])

        with col_bureau_1:
            st.markdown("##### Score Distribution by Loan Outcome")
            score_col = st.radio("Select External Bureau Score to inspect:", ["EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1"], horizontal=True)
            
            if score_col in df.columns:
                fig_score = px.histogram(
                    df.dropna(subset=[score_col]), 
                    x=score_col, 
                    color='TARGET_LABEL', 
                    barmode='overlay',
                    marginal='box',
                    nbins=40,
                    opacity=0.65,
                    color_discrete_map={'Non-Default (Paid)': COLOR_PAID, 'Defaulted': COLOR_DEFAULT}
                )
                fig_score.update_layout(**DARK_LAYOUT, height=400, xaxis_title=f"{score_col} (Normalized 0.0 - 1.0)")
                st.plotly_chart(fig_score, use_container_width=True)

                # Statistical summary table
                s_paid = df.loc[df['TARGET'] == 0, score_col].dropna()
                s_def = df.loc[df['TARGET'] == 1, score_col].dropna()
                st.markdown(
                    f"**Statistical Separation**: Mean for Non-Default: `{s_paid.mean():.3f}` | "
                    f"Mean for Default: `{s_def.mean():.3f}` (*Δ = {(s_paid.mean() - s_def.mean()):.3f} point gap*)."
                )

        with col_bureau_2:
            st.markdown("##### Bivariate Interaction: EXT_SOURCE_2 vs EXT_SOURCE_3")
            if 'EXT_SOURCE_2' in df.columns and 'EXT_SOURCE_3' in df.columns:
                sub_ext = df.dropna(subset=['EXT_SOURCE_2', 'EXT_SOURCE_3']).sample(min(4000, len(df)), random_state=42)
                fig_scatter_ext = px.scatter(
                    sub_ext, 
                    x='EXT_SOURCE_2', 
                    y='EXT_SOURCE_3', 
                    color='TARGET_LABEL',
                    opacity=0.45,
                    color_discrete_map={'Non-Default (Paid)': '#10B981', 'Defaulted': '#EF4444'},
                    labels={'EXT_SOURCE_2': 'Bureau Score 2', 'EXT_SOURCE_3': 'Bureau Score 3'}
                )
                fig_scatter_ext.update_layout(**DARK_LAYOUT, height=400)
                st.plotly_chart(fig_scatter_ext, use_container_width=True)
                st.caption("Notice the dense cluster of Crimson Default points in the lower-left quadrant (both scores < 0.25).")

        # Correlation of bureau scores with TARGET
        ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
        if ext_cols:
            corrs = df[ext_cols + ['TARGET']].corr()['TARGET'].drop('TARGET').reset_index()
            corrs.columns = ['Feature', 'CorrelationWithTarget']
            fig_ext_corr = px.bar(
                corrs, 
                x='CorrelationWithTarget', 
                y='Feature', 
                orientation='h',
                color='CorrelationWithTarget',
                color_continuous_scale=['#EF4444', '#10B981'],
                title="Linear Correlation with Loan Default (TARGET)"
            )
            fig_ext_corr.update_layout(**DARK_LAYOUT, height=220, coloraxis_showscale=False)
            st.plotly_chart(fig_ext_corr, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: Demographics & Socio-Economic Risk
    # ---------------------------------------------------------
    with tabs[2]:
        st.subheader("Demographic & Socio-Economic Risk Cohorts")
        
        col_demo_1, col_demo_2 = st.columns(2)

        with col_demo_1:
            st.markdown("##### Age Cohort Default Rate (%)")
            if 'AGE_GROUP' in df.columns:
                age_summary = df.groupby('AGE_GROUP', observed=False).agg(
                    Total=('TARGET', 'count'),
                    Defaults=('TARGET', 'sum'),
                    DefaultRate=('TARGET', lambda x: x.mean() * 100)
                ).reset_index()
                
                fig_age = px.bar(
                    age_summary, 
                    x='AGE_GROUP', 
                    y='DefaultRate', 
                    text=age_summary['DefaultRate'].apply(lambda v: f"{v:.1f}%"),
                    color='DefaultRate',
                    color_continuous_scale=['#10B981', '#F59E0B', '#EF4444'],
                    labels={'AGE_GROUP': 'Age Bracket (Years)', 'DefaultRate': 'Default Rate (%)'}
                )
                fig_age.update_layout(**DARK_LAYOUT, height=380, coloraxis_showscale=False)
                fig_age.update_traces(textposition='outside')
                st.plotly_chart(fig_age, use_container_width=True)
                st.caption("Younger borrowers (18-25) default at **12.3%**, whereas seniors (65+) default at only **5.4%** — a monotonically decreasing risk profile.")

        with col_demo_2:
            st.markdown("##### Education Level vs Default Likelihood")
            if 'NAME_EDUCATION_TYPE' in df.columns:
                edu_summary = df.groupby('NAME_EDUCATION_TYPE').agg(
                    Total=('TARGET', 'count'),
                    DefaultRate=('TARGET', lambda x: x.mean() * 100)
                ).reset_index().sort_values(by='DefaultRate', ascending=False)

                fig_edu = px.bar(
                    edu_summary, 
                    x='DefaultRate', 
                    y='NAME_EDUCATION_TYPE', 
                    orientation='h',
                    text=edu_summary['DefaultRate'].apply(lambda v: f"{v:.1f}%"),
                    color='DefaultRate',
                    color_continuous_scale=['#10B981', '#EF4444'],
                    labels={'NAME_EDUCATION_TYPE': 'Education Level', 'DefaultRate': 'Default Rate (%)'}
                )
                fig_edu.update_layout(**DARK_LAYOUT, height=380, coloraxis_showscale=False)
                fig_edu.update_traces(textposition='outside')
                st.plotly_chart(fig_edu, use_container_width=True)
                st.caption("Applicants with Lower Secondary education default at >10.5%, compared to <2.0% for Academic Degree holders.")

        col_demo_3, col_demo_4 = st.columns(2)

        with col_demo_3:
            st.markdown("##### Gender Breakdown")
            if 'CODE_GENDER' in df.columns:
                gender_df = df[df['CODE_GENDER'].isin(['M', 'F'])].groupby('CODE_GENDER')['TARGET'].mean().reset_index()
                gender_df['DefaultRate'] = gender_df['TARGET'] * 100
                gender_df['Gender'] = gender_df['CODE_GENDER'].map({'M': 'Male (M)', 'F': 'Female (F)'})

                fig_gender = px.bar(
                    gender_df, 
                    x='Gender', 
                    y='DefaultRate', 
                    text=gender_df['DefaultRate'].apply(lambda v: f"{v:.2f}%"),
                    color='DefaultRate',
                    color_continuous_scale=['#10B981', '#EF4444'],
                    labels={'DefaultRate': 'Default Rate (%)'}
                )
                fig_gender.update_layout(**DARK_LAYOUT, height=320, coloraxis_showscale=False)
                fig_gender.update_traces(textposition='outside')
                st.plotly_chart(fig_gender, use_container_width=True)
                st.caption("Male applicants demonstrate an average default rate of ~10.1%, compared to ~7.0% for females.")

        with col_demo_4:
            st.markdown("##### Employment Outlier: The 365,243 Anomaly")
            if 'IS_PENSIONER_ANOM' in df.columns:
                anom_summary = df.groupby('IS_PENSIONER_ANOM').agg(
                    Total=('TARGET', 'count'),
                    DefaultRate=('TARGET', lambda x: x.mean() * 100)
                ).reset_index()

                fig_anom = px.bar(
                    anom_summary, 
                    x='IS_PENSIONER_ANOM', 
                    y='DefaultRate', 
                    text=anom_summary['DefaultRate'].apply(lambda v: f"{v:.2f}%"),
                    color='DefaultRate',
                    color_continuous_scale=['#10B981', '#EF4444'],
                    labels={'IS_PENSIONER_ANOM': 'Employment Category', 'DefaultRate': 'Default Rate (%)'}
                )
                fig_anom.update_layout(**DARK_LAYOUT, height=320, coloraxis_showscale=False)
                fig_anom.update_traces(textposition='outside')
                st.plotly_chart(fig_anom, use_container_width=True)
                st.caption("The anomalous value `DAYS_EMPLOYED = 365243` represents pensioners who default significantly less (5.4% vs 8.7%).")

    # ---------------------------------------------------------
    # TAB 4: Financial Health & Ratios
    # ---------------------------------------------------------
    with tabs[3]:
        st.subheader("Financial Health, Loan Economics & Credit Burden")
        
        col_fin_1, col_fin_2 = st.columns(2)

        with col_fin_1:
            st.markdown("##### Debt Service Burden (Annuity-to-Income Ratio)")
            if 'ANNUITY_INCOME_RATIO' in df.columns:
                fig_dti = px.box(
                    df[df['ANNUITY_INCOME_RATIO'] < 0.6], 
                    x='TARGET_LABEL', 
                    y='ANNUITY_INCOME_RATIO', 
                    color='TARGET_LABEL',
                    color_discrete_map={'Non-Default (Paid)': COLOR_PAID, 'Defaulted': COLOR_DEFAULT},
                    labels={'ANNUITY_INCOME_RATIO': 'Annuity / Income Ratio', 'TARGET_LABEL': 'Status'}
                )
                fig_dti.update_layout(**DARK_LAYOUT, height=380, showlegend=False)
                st.plotly_chart(fig_dti, use_container_width=True)
                st.caption("Applicants devoting a higher fraction of their monthly income to loan repayments exhibit elevated default risk.")

        with col_fin_2:
            st.markdown("##### Loan-to-Income Multiple (Credit / Total Income)")
            if 'CREDIT_INCOME_RATIO' in df.columns:
                fig_lti = px.box(
                    df[df['CREDIT_INCOME_RATIO'] < 10], 
                    x='TARGET_LABEL', 
                    y='CREDIT_INCOME_RATIO', 
                    color='TARGET_LABEL',
                    color_discrete_map={'Non-Default (Paid)': COLOR_PAID, 'Defaulted': COLOR_DEFAULT},
                    labels={'CREDIT_INCOME_RATIO': 'Credit / Income Multiple', 'TARGET_LABEL': 'Status'}
                )
                fig_lti.update_layout(**DARK_LAYOUT, height=380, showlegend=False)
                st.plotly_chart(fig_lti, use_container_width=True)
                st.caption("Credit amount multiples exceeding 5x annual income correlate with debt stress and payment delinquency.")

        st.markdown("##### Credit Requested vs Value of Goods Purchased (Over-Financing Risk)")
        if 'AMT_CREDIT' in df.columns and 'AMT_GOODS_PRICE' in df.columns:
            sub_fin = df.dropna(subset=['AMT_CREDIT', 'AMT_GOODS_PRICE']).sample(min(4000, len(df)), random_state=42)
            fig_goods = px.scatter(
                sub_fin, 
                x='AMT_GOODS_PRICE', 
                y='AMT_CREDIT', 
                color='TARGET_LABEL',
                opacity=0.45,
                color_discrete_map={'Non-Default (Paid)': COLOR_PAID, 'Defaulted': COLOR_DEFAULT},
                labels={'AMT_GOODS_PRICE': 'Goods Value ($)', 'AMT_CREDIT': 'Credit Requested ($)'}
            )
            # Add 1:1 reference line
            max_val = min(sub_fin['AMT_GOODS_PRICE'].quantile(0.99), sub_fin['AMT_CREDIT'].quantile(0.99))
            fig_goods.add_shape(
                type="line", line=dict(dash='dash', color='#94A3B8'),
                x0=0, y0=0, x1=max_val, y1=max_val
            )
            fig_goods.update_layout(**DARK_LAYOUT, height=420)
            st.plotly_chart(fig_goods, use_container_width=True)
            st.caption("Points above the dashed line represent over-financed loans (credit limit exceeding the purchase price of goods).")

    # ---------------------------------------------------------
    # TAB 5: Multi-Table Repayment Behavior
    # ---------------------------------------------------------
    with tabs[4]:
        st.subheader("Cross-Table Credit Bureau & Historical Repayment Signals")
        st.markdown(
            "Signals aggregated across **previous loan applications**, **Credit Bureau records**, "
            "and **installment repayment delays** provide critical behavioral patterns."
        )

        col_multi_1, col_multi_2 = st.columns(2)

        with col_multi_1:
            st.markdown("##### Previous Application History (Refused Loans)")
            prev_refused_cols = [c for c in df.columns if 'NAME_CONTRACT_STATUS_Refused' in c or 'PREV_NAME_CONTRACT_STATUS_Refused' in c]
            if prev_refused_cols:
                ref_col = prev_refused_cols[0]
                df['HAS_PREV_REFUSAL'] = np.where(df[ref_col] > 0, 'Had Previous Refusal', 'No Previous Refusals')
                ref_summary = df.groupby('HAS_PREV_REFUSAL')['TARGET'].agg(['count', 'mean']).reset_index()
                ref_summary['DefaultRate'] = ref_summary['mean'] * 100

                fig_ref = px.bar(
                    ref_summary, 
                    x='HAS_PREV_REFUSAL', 
                    y='DefaultRate', 
                    text=ref_summary['DefaultRate'].apply(lambda v: f"{v:.1f}%"),
                    color='DefaultRate',
                    color_continuous_scale=['#10B981', '#EF4444'],
                    labels={'HAS_PREV_REFUSAL': 'Previous Application Status', 'DefaultRate': 'Default Rate (%)'}
                )
                fig_ref.update_layout(**DARK_LAYOUT, height=360, coloraxis_showscale=False)
                fig_ref.update_traces(textposition='outside')
                st.plotly_chart(fig_ref, use_container_width=True)
                st.caption("Applicants with a prior loan refusal default at more than double the rate of applicants with clean approval histories.")
            elif 'PREV_SK_ID_PREV_count' in df.columns:
                fig_prev = px.box(
                    df, 
                    x='TARGET_LABEL', 
                    y='PREV_SK_ID_PREV_count', 
                    color='TARGET_LABEL',
                    color_discrete_map={'Non-Default (Paid)': COLOR_PAID, 'Defaulted': COLOR_DEFAULT},
                    labels={'PREV_SK_ID_PREV_count': 'Total Previous Applications', 'TARGET_LABEL': 'Status'}
                )
                fig_prev.update_layout(**DARK_LAYOUT, height=360, showlegend=False)
                st.plotly_chart(fig_prev, use_container_width=True)

        with col_multi_2:
            st.markdown("##### Installment Repayment Delays (Days Past Due)")
            delay_cols = [c for c in df.columns if 'PAYMENT_DELAY' in c and 'max' in c]
            if delay_cols:
                d_col = delay_cols[0]
                fig_delay = px.box(
                    df[df[d_col].between(0, 60)], 
                    x='TARGET_LABEL', 
                    y=d_col, 
                    color='TARGET_LABEL',
                    color_discrete_map={'Non-Default (Paid)': COLOR_PAID, 'Defaulted': COLOR_DEFAULT},
                    labels={d_col: 'Max Payment Delay (Days)', 'TARGET_LABEL': 'Status'}
                )
                fig_delay.update_layout(**DARK_LAYOUT, height=360, showlegend=False)
                st.plotly_chart(fig_delay, use_container_width=True)
                st.caption("A pattern of persistent payment delays (even 5-15 days) on past installment contracts is a leading indicator of current default.")
            else:
                st.info("Installment delay aggregations available in full merged parquet dataset.")

        # Bureau Overdue Debt
        overdue_cols = [c for c in df.columns if 'BUREAU_AMT_CREDIT_SUM_OVERDUE' in c and 'sum' in c]
        if overdue_cols:
            o_col = overdue_cols[0]
            st.markdown("##### External Bureau Overdue Debt Amount ($)")
            df['HAS_OVERDUE_DEBT'] = np.where(df[o_col] > 0, 'Has Overdue Bureau Debt', 'Zero Overdue Debt')
            over_summary = df.groupby('HAS_OVERDUE_DEBT')['TARGET'].agg(['count', 'mean']).reset_index()
            over_summary['DefaultRate'] = over_summary['mean'] * 100

            fig_over = px.bar(
                over_summary, 
                x='HAS_OVERDUE_DEBT', 
                y='DefaultRate', 
                text=over_summary['DefaultRate'].apply(lambda v: f"{v:.1f}%"),
                color='DefaultRate',
                color_continuous_scale=['#10B981', '#EF4444'],
                labels={'HAS_OVERDUE_DEBT': 'Credit Bureau Standing', 'DefaultRate': 'Default Rate (%)'}
            )
            fig_over.update_layout(**DARK_LAYOUT, height=300, coloraxis_showscale=False)
            fig_over.update_traces(textposition='outside')
            st.plotly_chart(fig_over, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 6: Correlation Matrix & Dynamic Feature Explorer
    # ---------------------------------------------------------
    with tabs[5]:
        st.subheader("Correlation Heatmap & Interactive Feature Explorer")
        
        # 1. Top Feature Correlations Heatmap
        st.markdown("##### Top Predictive Feature Correlations with Loan Default")
        numeric_df = df.select_dtypes(include=[np.number])
        if 'TARGET' in numeric_df.columns:
            corrs = numeric_df.corr()['TARGET'].drop('TARGET').dropna()
            top_pos = corrs.nlargest(7)
            top_neg = corrs.nsmallest(7)
            selected_corr_cols = list(top_pos.index) + list(top_neg.index) + ['TARGET']
            corr_matrix = numeric_df[selected_corr_cols].corr()

            fig_heat = px.imshow(
                corr_matrix, 
                color_continuous_scale='RdBu_r', 
                zmin=-0.3, 
                zmax=0.3,
                text_auto='.2f',
                aspect='auto'
            )
            fig_heat.update_layout(**DARK_LAYOUT, height=450)
            st.plotly_chart(fig_heat, use_container_width=True)

        # 2. Dynamic Feature Selector
        st.markdown("---")
        st.markdown("##### 🔍 Dynamic Univariate & Bivariate Feature Explorer")
        st.markdown("Pick any feature in the dataset to inspect its distribution split by loan outcome:")
        
        selectable_cols = [c for c in df.columns if c not in ['TARGET', 'TARGET_LABEL', 'SK_ID_CURR']]
        default_feat = 'EXT_SOURCE_3' if 'EXT_SOURCE_3' in selectable_cols else selectable_cols[0]
        chosen_col = st.selectbox("Select Feature to Explore:", selectable_cols, index=selectable_cols.index(default_feat))

        if chosen_col:
            is_num = pd.api.types.is_numeric_dtype(df[chosen_col])
            
            c_exp1, c_exp2 = st.columns([2, 1])
            with c_exp1:
                if is_num:
                    sub_clean = df.dropna(subset=[chosen_col])
                    # Trim extreme outliers for display if high skew
                    q99 = sub_clean[chosen_col].quantile(0.99)
                    q01 = sub_clean[chosen_col].quantile(0.01)
                    trimmed = sub_clean[(sub_clean[chosen_col] >= q01) & (sub_clean[chosen_col] <= q99)]

                    fig_dynamic = px.histogram(
                        trimmed, 
                        x=chosen_col, 
                        color='TARGET_LABEL', 
                        barmode='overlay',
                        marginal='box',
                        opacity=0.6,
                        color_discrete_map={'Non-Default (Paid)': COLOR_PAID, 'Defaulted': COLOR_DEFAULT}
                    )
                    fig_dynamic.update_layout(**DARK_LAYOUT, height=400, title=f"Distribution of {chosen_col} by Target")
                    st.plotly_chart(fig_dynamic, use_container_width=True)
                else:
                    cat_summary = df.groupby(chosen_col)['TARGET'].agg(
                        Count='count', 
                        DefaultRate=lambda x: x.mean() * 100
                    ).reset_index().sort_values(by='Count', ascending=False).head(15)

                    fig_dynamic = px.bar(
                        cat_summary, 
                        x=chosen_col, 
                        y='DefaultRate', 
                        text=cat_summary['DefaultRate'].apply(lambda v: f"{v:.1f}%"),
                        color='DefaultRate',
                        color_continuous_scale=['#10B981', '#EF4444'],
                        labels={'DefaultRate': 'Default Rate (%)'}
                    )
                    fig_dynamic.update_layout(**DARK_LAYOUT, height=400, title=f"Default Rate by {chosen_col}")
                    fig_dynamic.update_traces(textposition='outside')
                    st.plotly_chart(fig_dynamic, use_container_width=True)

            with c_exp2:
                st.markdown("###### Feature Summary Statistics")
                if is_num:
                    p_mean = df.loc[df['TARGET'] == 0, chosen_col].mean()
                    d_mean = df.loc[df['TARGET'] == 1, chosen_col].mean()
                    p_med = df.loc[df['TARGET'] == 0, chosen_col].median()
                    d_med = df.loc[df['TARGET'] == 1, chosen_col].median()
                    
                    st.metric("Paid (Mean)", f"{p_mean:.3f}" if abs(p_mean) < 100 else f"{p_mean:,.1f}")
                    st.metric("Default (Mean)", f"{d_mean:.3f}" if abs(d_mean) < 100 else f"{d_mean:,.1f}")
                    st.metric("Paid (Median)", f"{p_med:.3f}" if abs(p_med) < 100 else f"{p_med:,.1f}")
                    st.metric("Default (Median)", f"{d_med:.3f}" if abs(d_med) < 100 else f"{d_med:,.1f}")
                    st.metric("Missing Count", f"{df[chosen_col].isnull().sum():,}")
                else:
                    st.write(df[chosen_col].value_counts(normalize=True).head(8) * 100)
                    st.metric("Unique Categories", df[chosen_col].nunique())
                    st.metric("Missing Count", f"{df[chosen_col].isnull().sum():,}")

        # Raw Data Sample
        st.markdown("---")
        with st.expander("📄 View Underlying Micro-Dataset Sample (First 100 Rows)"):
            st.dataframe(df.head(100), use_container_width=True)
