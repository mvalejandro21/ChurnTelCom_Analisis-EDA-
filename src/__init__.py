# main.py

import pandas as pd
from data_cleaning import (
    resumen_dataset,
    drop_irrelevant_columns,
    eliminar_duplicados,
    imputar_valores_nulos,
    codificar_variables,
    escalar_columnas_numericas
)
from data_prepping import dividir_dataset


# ===============================
# 📥 CARGA Y PREPROCESAMIENTO
# ===============================

# Cambia esta ruta por la tuya
df = pd.read_csv("ruta/a/tu_dataset.csv")

# Paso 1: Resumen inicial
resumen_dataset(df)

# Paso 2: Eliminar columnas vacías, constantes o manuales
df = drop_irrelevant_columns(df, protected_cols=["target_col"], manual_drop=["col_inutil"])

# Paso 3: Eliminar duplicados
df = eliminar_duplicados(df)

# Paso 4: Imputar valores nulos
df = imputar_valores_nulos(df, protected_cols=["target_col"])

# Paso 5: Codificar variables categóricas
df = codificar_variables(df, protected_cols=["target_col"])

# Paso 6: Escalar variables numéricas
df = escalar_columnas_numericas(df)

# ===============================
# 🤖 MACHINE LEARNING FLOW
# ===============================

# Cambia esto por el nombre real de tu variable objetivo
TARGET = "target_col"

# Paso 7: Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = dividir_dataset(df, target=TARGET)

# Verifica que todo ha ido bien
print("✅ Datos preparados:")
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")