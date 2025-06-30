import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline




def main_prepping(X, y, preprocessor):
    # -------------------------
    # Dividir en train/test
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------------------------
    # Crear pipeline con preprocessor y modelo
    # -------------------------
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # -------------------------
    # Entrenar modelo
    # -------------------------
    model_pipeline.fit(X_train, y_train)

    # -------------------------
    # Evaluar modelo
    # -------------------------
    y_pred = model_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # -------------------------
    # Guardar modelo completo
    # -------------------------
    joblib.dump(model_pipeline, "modelo_completo.joblib")

    # Si prefieres guardar el preprocessor y el modelo por separado
    joblib.dump(preprocessor, "preprocessor.joblib")
    joblib.dump(model_pipeline.named_steps['classifier'], "modelo.joblib")

    print("✅ Modelo y preprocessor guardados correctamente.")





def comparar_modelos(preprocessor, X, y):
    """
    Compara varios modelos, devuelve tabla de métricas,
    y muestra matriz de confusión y ROC solo para Random Forest.
    """
    # Separar en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Modelos a comparar
    modelos = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    # Guardar métricas
    metricas = []

    for nombre, modelo in modelos.items():
        # Pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', modelo)
        ])

        # Entrenar
        pipeline.fit(X_train, y_train)

        # Predicciones
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # Guardar
        metricas.append({
            "Modelo": nombre,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "AUC": roc_auc
        })

        # Mostrar gráficos solo para Random Forest
        if nombre == "Random Forest":
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No churn', 'Churn'])
            disp.plot(cmap='Blues')
            plt.title(f"Matriz de Confusión - {nombre}")
            plt.show()

            # Curva ROC
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Curva ROC - {nombre}")
            plt.legend(loc="lower right")
            plt.grid()
            plt.show()

    # Crear dataframe resumen
    df_metricas = pd.DataFrame(metricas)
    print("Resumen comparativo de modelos:")
    print(df_metricas)


