import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def columns(df):
        # ✅ Se codifican con LabelEncoder manualmente (se convierten a 0/1 antes del pipeline)
    label_encoding_cols = [
        "gender", "under_30", "senior_citizen", "married", "dependents",
        "referred_a_friend", "phone_service", "multiple_lines", "online_security",
        "online_backup", "device_protection_plan", "premium_tech_support",
        "streaming_tv", "streaming_movies", "streaming_music", "unlimited_data",
        "paperless_billing", "churn_label"
    ]

    # ✅ Se codifican con OneHotEncoder
    onehot_encoding_cols = [
        "city", "offer", "internet_service", "internet_type",
        "contract", "payment_method", "customer_status",
        "churn_category", "churn_reason"
    ]

    # ✅ Se escalan con StandardScaler
    numeric_cols = [
        "age", "number_of_dependents", "population", "number_of_referrals",
        "tenure_in_months", "avg_monthly_long_distance_charges",
        "avg_monthly_gb_download", "monthly_charge", "total_charges",
        "total_refunds", "total_extra_data_charges", "total_long_distance_charges",
        "total_revenue", "satisfaction_score", "churn_score", "cltv"
    ]
    return label_encoding_cols, onehot_encoding_cols, numeric_cols

def pipeline_encoding(label_encoding_cols, onehot_encoding_cols, numeric_cols):
    # Pipeline de label cols
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}

    for col in label_encoding_cols:
        if df[col].nunique() == 2:
            df[col] = df[col].map(binary_map)
    onehot_pipeline = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Escalado numérico (sin imputación)
    numeric_pipeline = Pipeline(steps=[
        ("scaler", RobustScaler())
    ])

    # ColumnTransformer completo
    preprocessor = ColumnTransformer(transformers=[
        ("onehot", onehot_pipeline, onehot_encoding_cols),
        ("numeric", numeric_pipeline, numeric_cols)
    ], remainder="passthrough")  # Mantiene las columnas con label encoding manual

    


def data_processing(df):

    label_encoding_cols, onehot_encoding_cols, numeric_cols = columns(df)

    pipeline_encoding(label_encoding_cols, onehot_encoding_cols, numeric_cols)
    

    return df




data_processing(pd.read_csv("data/processed/telco_cleaned.csv"))