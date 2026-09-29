# Biomed Data Dashboard

An interactive biomedical data analysis dashboard built with Python, Streamlit, scikit-learn, and Plotly.

Note on Origin: This project is an interactive re-creation of the author's research-assistant data-analysis workflow; not the original research code.

Disclaimer: The built-in dataset is purely synthetic. It mimics the statistical properties of extracted audio frequencies, image texture contrasts, and cellular density distributions but contains no real patient data.

## Features

Data Intake: Utilize the built-in 500-sample synthetic dataset or upload your own CSV. Uploads are strictly validated for numeric features and a binary label target.

Exploratory Data Analysis (EDA): Interactive feature distribution histograms, automated missing value tracking, and correlation heatmaps.

One-Click Hypothesis Testing: Automated execution of t-tests, Mann-Whitney U, and ANOVA tests across label groups, complete with Cohen's d effect size calculations and visualizations.

Machine Learning Comparison: Side-by-side evaluation of Logistic Regression, Random Forest, and SVM models using a fixed seed. Features dynamic ROC curve plotting and classification metric tables (Accuracy, Precision, Recall, F1).

Feature Importance Tracking: Automatic extraction and visualization of Random Forest feature importances.

Automated Reporting: Generate a comprehensive session report combining dataset shape, statistical highlights, and ML metrics, exportable as an HTML document.

## How to Run

Ensure you have Python 3.9+ installed, then execute the following commands in your terminal:

```bash
# Install the required dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py
```

The application will open automatically in your default web browser (typically at http://localhost:8501).
