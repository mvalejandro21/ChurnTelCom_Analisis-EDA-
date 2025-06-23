import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib




def data_cleaning(df):

    """
    Perform data cleaning on the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """

    print("🧾 [INFO] Iniciando limpieza de datos...")

    

    df = standardize_column_names(df)
    df = eliminar_duplicados(df)  # Eliminar duplicados
    df = imputar_valores_nulos(df)  # Imputar valores nulos
    df = drop_irrelevant_columns(df)  # Eliminar columnas irrelevantes
    df = save_cleaned_data(df)  # Guardar datos limpios


    return df

def eliminar_duplicados(df):
    """
    Elimina duplicados del DataFrame.
    
    Parameters:
    df (pd.DataFrame): El DataFrame original.
    
    Returns:
    pd.DataFrame: El DataFrame sin duplicados.
    """
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)  # Reset index after dropping rows
    print(f"✅ Duplicados eliminados. Nuevo tamaño: {df.shape}")
    return df   

def imputar_valores_nulos(df):
    # Imputa valores nulos en el DataFrame.
    """
    Para imputar los valores vamos a identificar la causa ya sabemos que los nulos son offer, internet_type, churn reason y chur category
    estos dos ultimos se deben a que no son filas que tengan churn por lo tanto no hat motivo ni categoria
    Las otras dos son por no tener oferta activa o no tener contratado el interent asi que se imputan con "No tiene oferta" y "No tiene internet contratado" 
    """ 
    print("🧾 [INFO] Imputando valores nulos...")
    df["churn_reason"] = df["churn_reason"].fillna("no_churn")
    df["churn_category"] = df["churn_category"].fillna("no_churn")
    df["offer"] = df["offer"].fillna("no_offer")
    df["internet_type"] = df["internet_type"].fillna("no_internet_service")
    print("✅ Imputación de valores nulos completada.")
    print(f"🧾 [INFO] Valores nulos restantes: {df.isnull().sum().sum()}")
    return df


def drop_irrelevant_columns(df) :
    """
    Elimina columnas innecesarias:
    - Completamente vacías
    - Constantes (un único valor)
    - Indicadas manualmente
    Se respeta una lista de columnas protegidas que no deben eliminarse.
    """
    protected_cols =  []
    manual_drop =  ['zip_code','latitude','longitude','churn_score']

    # Detectar columnas vacías
    empty_cols = df.columns[df.isnull().all()].tolist()

    # Detectar columnas constantes
    constant_cols = df.columns[df.nunique() <= 1].tolist()

    # Unir todas las columnas a eliminar
    all_to_drop = list(set(empty_cols + constant_cols + manual_drop))

    # Excluir columnas protegidas
    final_to_drop = [col for col in all_to_drop if col in df.columns and col not in protected_cols]

    print("📌 Columnas a eliminar:")
    print(final_to_drop)

    # Eliminar columnas seleccionadas
    df = df.drop(columns=final_to_drop, errors='ignore')
    print("✅ Limpieza de columnas irrelevantes completada.")
    print(f"🧾 [INFO] Columnas eliminadas: {final_to_drop}")
    return df


def standardize_column_names(df):
   
    df.columns = (
        df.columns
        .str.strip()               # elimina espacios al inicio/fin
        .str.lower()               # convierte a minúsculas
        .str.replace(" ", "_")     # reemplaza espacios por guiones bajos
        .str.replace(r"[^\w_]", "", regex=True)  # quita símbolos raros
    )

    print("🧾 [COLUMNS] Nombres de columnas estandarizados.")
    print("🧾 [INFO] Columnas actuales en el dataset:"
          f"\n{df.columns.tolist()}")
    

    return df


def save_cleaned_data(df):
    """
    Guarda el DataFrame limpio en un archivo CSV.
    
    Parameters:
    df (pd.DataFrame): El DataFrame a guardar.
    filename (str): Ruta del archivo donde se guardará el DataFrame.
    """
    df.to_csv("data/processed/telco_cleaned.csv", index=False)
    print(f"✅ Datos limpios guardados en: data/processed/telco_cleaned.csv")


data_cleaning(pd.read_csv("data/raw/telco.csv"))

