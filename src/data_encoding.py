import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report



def data_processing(df):
    # Eliminar columnas que filtran el target
    columns_to_remove = ['customer_id', 'city', 'churn_reason', 'customer_status', 'churn_category','population']
    df_reduced = df.drop(columns=columns_to_remove)

    # Separar target y features
    X = df_reduced.drop(columns=['churn_label'])
    y = df_reduced['churn_label'].map({'Yes': 1, 'No': 0})

    # Columnas categóricas y numéricas
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Pipelines de transformación
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    return X, y, preprocessor
