import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Optional

# ================================
# 🧾 FUNCIONES UTILITARIAS
# ================================

def resumen_dataset(df: pd.DataFrame) -> None:
    """
    Muestra un resumen general del dataset: dimensiones, tipos de datos, nulos y estadísticas descriptivas.
    """
    print(f"🔎 Dimensiones: {df.shape}")
    print(f"\n📋 Tipos de datos:\n{df.dtypes}")
    print(f"\n📉 Valores nulos:\n{df.isnull().sum()}")
    print(f"\n📊 Descripción:\n{df.describe(include='all').T}")


# =====================================================
# 🚫 ELIMINAR COLUMNAS IRRELEVANTES AUTOMÁTICAMENTE
# =====================================================

def drop_irrelevant_columns(
    df: pd.DataFrame,
    protected_cols: Optional[List[str]] = None,
    manual_drop: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Elimina columnas innecesarias:
    - Completamente vacías
    - Constantes (un único valor)
    - Indicadas manualmente
    Se respeta una lista de columnas protegidas que no deben eliminarse.
    """
    protected_cols = protected_cols or []
    manual_drop = manual_drop or []

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
    return df


# ================================
# 🧽 ELIMINAR DUPLICADOS
# ================================

def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina registros duplicados del DataFrame y muestra cuántos se eliminaron.
    """
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"🧹 Duplicados eliminados: {before - after}")
    return df


# ====================================================
# ⚙️ IMPUTACIÓN AUTOMÁTICA DE VALORES NULOS
# ====================================================

def imputar_valores_nulos(
    df: pd.DataFrame,
    protected_cols: Optional[List[str]] = None,
    threshold: float = 0.3
) -> pd.DataFrame:
    """
    Imputa valores nulos automáticamente:
    - Numéricos: por la media si hay pocos nulos.
    - Categóricos: por la moda.
    Las columnas con demasiados nulos se omiten (según el umbral).
    También se respeta una lista de columnas protegidas.
    """
    protected_cols = protected_cols or []

    for col in df.columns:
        if col in protected_cols:
            continue

        missing_ratio = df[col].isnull().mean()

        if missing_ratio == 0:
            continue
        elif missing_ratio > threshold:
            print(f"⚠️ Saltando imputación en {col} (más del {threshold*100:.1f}% de nulos).")
            continue

        # Imputación según tipo de dato
        if df[col].dtype in [np.float64, np.int64]:
            valor = df[col].mean()
            df[col] = df[col].fillna(valor)
            print(f"🔧 Imputado {col} con media: {valor:.2f}")
        else:
            moda = df[col].mode().iloc[0]
            df[col] = df[col].fillna(moda)
            print(f"🔧 Imputado {col} con moda: {moda}")

    return df


# ====================================================
# 🔢 ENCODING INTELIGENTE
# ====================================================

def codificar_variables(
    df: pd.DataFrame,
    protected_cols: Optional[List[str]] = None,
    max_onehot: int = 10
) -> pd.DataFrame:
    """
    Codifica variables categóricas automáticamente:
    - Binarias: Label Encoding
    - Categóricas con pocos valores únicos: One-hot Encoding
    - Alta cardinalidad: elimina la columna
    Se omiten las columnas protegidas.
    """
    protected_cols = protected_cols or []
    df_encoded = df.copy()
    label_encoder = LabelEncoder()

    for col in df_encoded.select_dtypes(include="object").columns:
        if col in protected_cols:
            continue

        unique_vals = df_encoded[col].nunique()

        if unique_vals == 2:
            # Codificación para variables binarias
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
            print(f"🔒 Label encoded: {col}")

        elif unique_vals <= max_onehot:
            # One-hot encoding para pocas categorías
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            print(f"🔒 One-hot encoded: {col}")

        else:
            # Demasiados valores únicos: se elimina por alta cardinalidad
            df_encoded.drop(columns=[col], inplace=True)
            print(f"⚠️ Eliminado {col} por alta cardinalidad ({unique_vals})")

    return df_encoded


# ===============================================
# 📏 ESCALADO DE VARIABLES NUMÉRICAS
# ===============================================

def escalar_columnas_numericas(
    df: pd.DataFrame,
    columnas: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Escala columnas numéricas utilizando StandardScaler:
    - Si no se especifican columnas, escala todas las numéricas.
    """
    scaler = StandardScaler()
    columnas = columnas or df.select_dtypes(include=np.number).columns.tolist()

    df[columnas] = scaler.fit_transform(df[columnas])
    print(f"📐 Escaladas columnas: {columnas}")
    return df


# ===============================================
# ✂️ DIVISIÓN EN TRAIN Y TEST
# ===============================================

def dividir_dataset(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Divide el dataset en conjuntos de entrenamiento y prueba,
    estratificando por la variable objetivo.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
