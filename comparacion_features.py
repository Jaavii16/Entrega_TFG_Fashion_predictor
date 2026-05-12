import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- CONFIGURACIÓN ----------------
base_dir = os.path.dirname(os.path.abspath(__file__))
FILE_DATASET = os.path.join(base_dir, "dataset_entrenamiento.csv")
FECHA_CORTE = "2025-01-01" 
TARGET_COL = "TARGET_NEXT_3M_AVG" 

def ejecutar_experimento(nombre, lista_features, X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train[lista_features], y_train)
    preds = model.predict(X_test[lista_features])
    
    return {
        "Escenario": nombre,
        "MAE": round(mean_absolute_error(y_test, preds), 2),
        "R2": round(r2_score(y_test, preds), 4)
    }

def comparar_escenarios():
    if not os.path.exists(FILE_DATASET): return
    df = pd.read_csv(FILE_DATASET)
    df['fecha_dt'] = pd.to_datetime(df['mes'])
    
    # Split
    train = df[df['fecha_dt'] < pd.to_datetime(FECHA_CORTE)]
    test = df[df['fecha_dt'] >= pd.to_datetime(FECHA_CORTE)]
    
    # Definición de grupos de variables
    features_trends = [c for c in df.columns if 'trends_lag' in c or 'trends_mean' in c]
    # Ahora incluimos las nuevas variables de crecimiento y diferencia
    features_social = [c for c in df.columns if 'score_lag' in c or 'social_mean' in c or 'impact_score' in c or 'social_growth' in c or 'social_diff' in c]
    features_extra = [c for c in df.columns if 'sin_mes' in c or 'cos_mes' in c or 'cat_' in c]
    
    resultados = []

    # Escenario 1: Solo Trends (El benchmark estándar)
    resultados.append(ejecutar_experimento("Solo Google Trends", features_trends, train, train[TARGET_COL], test, test[TARGET_COL]))

    # Escenario 2: Solo Social (El valor de tus Agentes IA)
    # Añadimos las extras (categoría/mes) para que tenga algo de contexto
    resultados.append(ejecutar_experimento("Solo Redes Sociales + Cat", features_social + features_extra, train, train[TARGET_COL], test, test[TARGET_COL]))

    # Escenario 3: Modelo Híbrido (Tu propuesta final)
    resultados.append(ejecutar_experimento("Modelo Híbrido (Completo)", features_trends + features_social + features_extra, train, train[TARGET_COL], test, test[TARGET_COL]))

    # Mostrar tabla final
    df_res = pd.DataFrame(resultados)
    print("\n=== COMPARATIVA DE ESCENARIOS PARA EL TFG ===")
    print(df_res.to_string(index=False))
    
    # Conclusión automática para la memoria
    mejor_r2 = df_res.loc[df_res['R2'].idxmax(), 'Escenario']
    print(f"\nConclusión: El escenario '{mejor_r2}' ofrece el mejor rendimiento.")

if __name__ == "__main__":
    comparar_escenarios()