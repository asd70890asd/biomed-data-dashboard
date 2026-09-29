import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=500, random_seed=42):
    """
    Generates a synthetic biomedical dataset simulating feature extraction from
    audio and imaging modalities, alongside cell density counts.
    """
    np.random.seed(random_seed)

    # Define classes
    half = n_samples // 2
    labels = np.array(['healthy'] * half + ['diseased'] * (n_samples - half))
    np.random.shuffle(labels)

    data = []
    for label in labels:
        if label == 'healthy':
            # Healthy distribution parameters
            mf = np.random.normal(loc=250.0, scale=30.0)
            se = np.random.normal(loc=4.5, scale=0.5)
            itc = np.random.normal(loc=12.0, scale=2.0)
            cd = np.random.normal(loc=85.0, scale=10.0)
        else:
            # Diseased distribution parameters (shifted to simulate pathology)
            mf = np.random.normal(loc=210.0, scale=35.0)
            se = np.random.normal(loc=3.8, scale=0.6)
            itc = np.random.normal(loc=15.0, scale=2.5)
            cd = np.random.normal(loc=65.0, scale=12.0)

        data.append([mf, se, itc, cd, label])

    df = pd.DataFrame(data, columns=[
        'mean_frequency',
        'spectral_entropy',
        'image_texture_contrast',
        'cell_density',
        'diagnosis'
    ])

    # Introduce ~2% missing data randomly in numeric columns to test robustness
    for col in df.columns[:-1]:
        mask = np.random.rand(n_samples) < 0.02
        df.loc[mask, col] = np.nan

    return df

def validate_uploaded_data(df):
    """
    Validates an uploaded CSV dataframe for the dashboard workflow.
    Ensures numeric feature columns exist and identifies one binary label column.

    Returns: (is_valid: bool, error_message: str, target_col: str, numeric_cols: list)
    """
    if df.empty:
        return False, "The uploaded dataset is empty.", None, None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    target_col = None

    # Look for a binary target column
    # Priority 1: Categorical columns with exactly 2 unique values
    binary_cats = [col for col in categorical_cols if df[col].nunique(dropna=True) == 2]

    # Priority 2: Numeric columns with exactly 2 unique values (e.g., 0 and 1)
    binary_nums = [col for col in numeric_cols if df[col].nunique(dropna=True) == 2]

    if binary_cats:
        target_col = binary_cats[-1]  # Take the last valid categorical as target
    elif binary_nums:
        target_col = binary_nums[-1]  # Take the last valid binary numeric as target
        numeric_cols.remove(target_col)
    else:
        return False, "Could not identify a valid binary label column (needs exactly 2 unique classes for classification).", None, None

    if len(numeric_cols) < 1:
        return False, "The dataset requires at least one continuous numeric feature column.", None, None

    return True, "Data is valid.", target_col, numeric_cols
