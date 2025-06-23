import pandas as pd


def seleccionar_columnas_binarias(df):
    columnas_binarias = []
    for columna in df.columns:
        valores_unicos = df[columna].dropna().unique()
        if set(valores_unicos).issubset({"Yes", "No"}):
            columnas_binarias.append(columna)
    return df[columnas_binarias]


def data_processing(df):

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print("🧾 [NUMERICAL] Columnas numéricas detectadas:", numerical_cols)

    binary_cols_manual = ["under_30","senior_citizen","married","dependents","referred_a_friend","phone_service","multiple_lines",
                   "internet_service","online_security","online_backup","device_protection_plan","premium_tech_support",
                   "streaming_tv","streaming_movies","streaming_music","unlimited_data","paperless_billing, gender"]
    
    binary_cols = [col for col in binary_cols_manual if col in df.columns.tolist()]
    print("🧾 [BINARY] Columnas binarias detectadas:", binary_cols)

    ordinal_cols



data_processing(pd.read_csv("data/processed/telco_cleaned.csv"))