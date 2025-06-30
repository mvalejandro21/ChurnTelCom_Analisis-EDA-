# main.py
import pandas as pd

from data_cleaning import data_cleaning
from data_encoding import data_processing
from data_prepping import main_prepping
from data_prepping import comparar_modelos


# ===============================
# 📥 CARGA Y PREPROCESAMIENTO
# ===============================

# Cambia esta ruta por la tuya
df = pd.read_csv("../data/raw/telco.csv")
# Limpieza de datos
data_cleaning(df)
# Preprocesamiento de datos
df_cleaned = pd.read_csv("../data/processed/telco_cleaned.csv")
X, y, preprocessor = data_processing(df_cleaned)

comparar_modelos(preprocessor, X, y)
main_prepping(X, y, preprocessor)


# ===============================

# Cambia esto por el nombre real de tu variable objetivo
