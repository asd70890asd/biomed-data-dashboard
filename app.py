import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from data_utils import generate_synthetic_data, validate_uploaded_data
import base64
import markdown

# Streamlit Page Config
st.set_page_config(page_title="Biomed Data Dashboard", layout="wide", page_icon="🧬")

# Fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Avoid division by zero if variances are 0
    if var1 == 0 and var2 == 0:
        return 0.0

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

# Sidebar Controls
st.sidebar.title("🧬 Biomed Data Dashboard")
st.sidebar.markdown("**Global settings:**")
st.sidebar.info(f"Fixed Random Seed: **{RANDOM_SEED}**\n\nRuns are completely reproducible.")

data_source = st.sidebar.radio("Data Source", ["Built-in Synthetic Dataset", "Upload CSV"])

# Data Loading
df = None
target_col = None
numeric_cols = []

if data_source == "Built-in Synthetic Dataset":
    df = generate_synthetic_data(random_seed=RANDOM_SEED)
    target_col = "diagnosis"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.sidebar.success("Loaded 500 synthetic samples.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            temp_df = pd.read_csv(uploaded_file)
            is_valid, msg, t_col, n_cols = validate_uploaded_data(temp_df)
            if is_valid:
                df = temp_df
                target_col = t_col
                numeric_cols = n_cols
                st.sidebar.success(f"Loaded successfully. Target identified: '{target_col}'")
            else:
                st.sidebar.error(f"Validation Error: {msg}")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
    else:
        st.sidebar.warning("Please upload a CSV file to proceed.")

# Main Application logic
if df is not None:
    # Setup Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Explore", "Statistics", "ML Models", "Report"])

    # --- TAB 1: EXPLORE ---
    with tab1:
        st.header("Exploratory Data Analysis")
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Dataset Shape")
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")

            st.subheader("Missing Values")
            missing_df = df.isna().sum().reset_index()
            missing_df.columns = ["Feature", "Missing Count"]
            st.dataframe(missing_df, use_container_width=True)

        with col2:
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Distributions")
        dist_cols = st.columns(2)
        for i, col in enumerate(numeric_cols):
            fig = px.histogram(df, x=col, color=target_col, barmode="overlay",
                               title=f"Distribution of {col} by {target_col}",
                               opacity=0.7, color_discrete_sequence=px.colors.qualitative.Pastel)
            dist_cols[i % 2].plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Correlation Heatmap (Numeric Features)")
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)


    # --- TAB 2: STATISTICS ---
    with tab2:
        st.header("Hypothesis Testing")
        st.write(f"Comparing numeric features across groups defined by **{target_col}**.")

        clean_df = df.dropna(subset=[target_col] + numeric_cols)
        classes = clean_df[target_col].unique()

        if len(classes) != 2:
            st.error("Statistics tab requires exactly 2 classes in the target column.")
        else:
            group1 = clean_df[clean_df[target_col] == classes[0]]
            group2 = clean_df[clean_df[target_col] == classes[1]]

            stats_results = []

            for col in numeric_cols:
                g1_data = group1[col].values
                g2_data = group2[col].values

                # t-test
                t_stat, p_t = stats.ttest_ind(g1_data, g2_data, equal_var=False)
                # Mann-Whitney U
                u_stat, p_u = stats.mannwhitneyu(g1_data, g2_data)
                # ANOVA
                f_stat, p_f = stats.f_oneway(g1_data, g2_data)
                # Effect Size
                d = calculate_cohens_d(g1_data, g2_data)

                stats_results.append({
                    "Feature": col,
                    "t-test p-value": p_t,
                    "Mann-Whitney p-value": p_u,
                    "ANOVA p-value": p_f,
                    "Cohen's d": d,
                    "Abs(Cohen's d)": abs(d)
                })

            stats_df = pd.DataFrame(stats_results)
            st.dataframe(stats_df.drop(columns=["Abs(Cohen's d)"]).style.format({
                "t-test p-value": "{:.4e}",
                "Mann-Whitney p-value": "{:.4e}",
                "ANOVA p-value": "{:.4e}",
                "Cohen's d": "{:.4f}"
            }), use_container_width=True)

            st.subheader("Effect Size Visualization")
            fig_effect = px.bar(stats_df.sort_values("Abs(Cohen's d)", ascending=False),
                                x="Cohen's d", y="Feature", orientation='h',
                                title="Cohen's d Effect Size by Feature (Absolute magnitude sorting)",
                                color="Abs(Cohen's d)", color_continuous_scale="Viridis")
            st.plotly_chart(fig_effect, use_container_width=True)

    # --- TAB 3: ML MODELS ---
    with tab3:
        st.header("Machine Learning Model Comparison")
        st.write("Training Logistic Regression, Random Forest, and SVM on the dataset to predict the target label.")

        # Preprocessing
        ml_df = df.dropna(subset=[target_col] + numeric_cols)
        X = ml_df[numeric_cols]
        le = LabelEncoder()
        y = le.fit_transform(ml_df[target_col])

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Models
        models = {
            "Logistic Regression": LogisticRegression(random_state=RANDOM_SEED),
            "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100),
            "SVM (RBF Kernel)": SVC(probability=True, random_state=RANDOM_SEED)
        }

        results = []
        roc_data = {}
        rf_model = None

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            if name == "Random Forest":
                rf_model = model

            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "AUC": roc_auc
            })

        metrics_df = pd.DataFrame(results)

        col_m1, col_m2 = st.columns([1, 1])

        with col_m1:
            st.subheader("Classification Metrics")
            st.dataframe(metrics_df.style.format(precision=4), use_container_width=True)

            st.subheader("Feature Importance (Random Forest)")
            importances = rf_model.feature_importances_
            imp_df = pd.DataFrame({"Feature": numeric_cols, "Importance": importances}).sort_values("Importance", ascending=True)
            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h', title="RF Feature Importances")
            st.plotly_chart(fig_imp, use_container_width=True)

        with col_m2:
            st.subheader("ROC Curves")
            fig_roc = go.Figure()
            for name, data in roc_data.items():
                fig_roc.add_trace(go.Scatter(x=data["fpr"], y=data["tpr"], mode='lines',
                                             name=f"{name} (AUC = {data['auc']:.2f})"))

            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                         line=dict(dash='dash', color='gray'), showlegend=False))

            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                  title="Receiver Operating Characteristic (ROC)",
                                  legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99))
            st.plotly_chart(fig_roc, use_container_width=True)


    # --- TAB 4: REPORT ---
    with tab4:
        st.header("Analysis Report Generation")
        st.write("Generate a summary of the current session's findings.")

        # Build Markdown String
        md_report = f"""
# Biomedical Data Analysis Report

## 1. Dataset Overview
- **Total Samples:** {df.shape[0]}
- **Total Features:** {df.shape[1]}
- **Target Variable:** `{target_col}`
- **Numeric Features:** {', '.join(numeric_cols)}
- **Missing Values:** {df.isna().sum().sum()} total missing cells.

## 2. Statistical Highlights
The following are the top features differentiating the target classes based on Cohen's d effect size:
"""
        # Add top 3 stats
        if 'stats_df' in locals():
            top_stats = stats_df.sort_values("Abs(Cohen's d)", ascending=False).head(3)
            for _, row in top_stats.iterrows():
                feat_name = row["Feature"]
                d_val = row["Cohen's d"]
                p_val = row["ANOVA p-value"]
                md_report += f"- **{feat_name}**: Cohen's d = {d_val:.4f} (ANOVA p = {p_val:.4e})\n"

        md_report += """
## 3. Machine Learning Performance
Models were evaluated using an 80/20 train-test split (Seed 42).

| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
"""
        if 'metrics_df' in locals():
            for _, row in metrics_df.iterrows():
                md_report += f"| {row['Model']} | {row['Accuracy']:.4f} | {row['F1 Score']:.4f} | {row['AUC']:.4f} |\n"

        st.markdown(md_report)

        # HTML conversion
        html_content = markdown.markdown(md_report, extensions=['tables'])
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Biomedical Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                h1, h2 {{ color: #2c3e50; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        b64_html = base64.b64encode(full_html.encode('utf-8')).decode()
        href = f'<a href="data:text/html;base64,{b64_html}" download="biomed_report.html"><button style="padding:10px; background-color:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer;">Download Report as HTML</button></a>'
        st.markdown(href, unsafe_allow_html=True)

else:
    st.info("👈 Please select a data source from the sidebar to begin analysis.")
