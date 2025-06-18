# ml_utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def main_prepping():

    df = pd.read_csv("data/processed/telco_cleaned.csv")
   
    # Resumen inicial del dataset
    X_train, X_test, y_train, y_test = dividir_dataset(df, target='churn')
    entrenar_y_evaluar_modelo(
        modelo=RandomForestClassifier(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    evaluar_modelos_basicos(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    mostrar_confusion_roc(
        modelo=RandomForestClassifier(),
        X_test=X_test,
        y_test=y_test
    )


# ===============================
# ✂️ DIVISIÓN EN TRAIN Y TEST
# ===============================

def dividir_dataset(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """Divide el dataset en train y test manteniendo la proporción de clases."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# ==========================================
# ⚖️ ENTRENAMIENTO Y EVALUACIÓN GENERAL
# ==========================================

def entrenar_y_evaluar_modelo(modelo, X_train, y_train, X_test, y_test):
    """Entrena un modelo y muestra métricas básicas."""
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
    print("\n🧾 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    return modelo


# =====================================================
# 🔁 EVALUAR VARIOS MODELOS DE CLASIFICACIÓN
# =====================================================

def evaluar_modelos_basicos(X_train, y_train, X_test, y_test, cv: int = 5):
    """Evalúa múltiples modelos de clasificación comunes con cross-validation."""
    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier()
    }

    resultados = []

    for nombre, modelo in modelos.items():
        print(f"\n🔍 Evaluando: {nombre}")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores = cross_val_score(modelo, X_train, y_train, cv=cv)

        resultados.append({
            "Modelo": nombre,
            "Accuracy": acc,
            "CV Mean": scores.mean(),
            "CV Std": scores.std()
        })

    df_resultados = pd.DataFrame(resultados).sort_values(by="CV Mean", ascending=False)
    print("\n📊 Resultados Comparativos:")
    print(df_resultados)
    return df_resultados


# =====================================================
# 📉 MATRIZ DE CONFUSIÓN Y CURVA ROC
# =====================================================

def mostrar_confusion_roc(modelo, X_test, y_test):
    """Muestra matriz de confusión y curva ROC si es binaria."""
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("📊 Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # Curva ROC si es binario y se puede
    if y_proba is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("📈 Curva ROC")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("⚠️ ROC no disponible (modelo no binario o sin predict_proba)")


main_prepping()