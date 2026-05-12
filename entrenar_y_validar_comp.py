import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- CONFIGURACIÓN ----------------
base_dir = os.path.dirname(os.path.abspath(__file__))
FILE_DATASET = os.path.join(base_dir, "dataset_entrenamiento.csv")

# FECHA DE CORTE PARA VALIDACIÓN
# Entrenamos con el pasado y probamos si acierta el 2025
FECHA_CORTE = "2025-01-01" 

TARGET_COL = "TARGET_NEXT_3M_AVG" 
# TARGET_COL = "TARGET_NEXT_6M_AVG"

# ---------------- FUNCIONES AUXILIARES ----------------

def graficar_termino(df_completo, term):
    # Filtramos datos
    subset = df_completo[df_completo['termino'] == term].sort_values('fecha_dt')
    # Nota: Asegúrate de que 'df_completo' tenga la columna 'Prediccion' unida
    subset_test = subset.dropna(subset=['Prediccion']) 

    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    plt.plot(subset['fecha_dt'], subset[TARGET_COL], label='Realidad (Target)', color='gray', alpha=0.4, linewidth=2)
    plt.plot(subset_test['fecha_dt'], subset_test['Prediccion'], label='Predicción IA', color='red', marker='x', markersize=6, linewidth=2)

    plt.axvline(x=pd.to_datetime(FECHA_CORTE), color='black', linestyle='--', label='Corte Predicción')
    plt.title(f"Análisis de Tendencia: {term.upper()}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def mostrar_top_tendencias(df_test, n=10):
    """Muestra las top n tendencias predichas para el próximo trimestre, ordenadas por la predicción de interés.
    """
    # 1. Cogemos solo el último mes de datos que tenemos (el futuro más cercano)
    ultimo_mes = df_test['fecha_dt'].max()
    df_reciente = df_test[df_test['fecha_dt'] == ultimo_mes]
    
    # 2. Ordenamos por el valor que ha predicho la IA
    top_terminos = df_reciente.sort_values(by='Prediccion', ascending=False).head(n)
    
    print(f"\n--- TOP {n} TENDENCIAS PREDICHAS PARA EL PRÓXIMO TRIMESTRE ---")
    print(top_terminos[['termino', 'Prediccion']].to_string(index=False))
    
    # 3. Graficamos esos términos
    for term in top_terminos['termino']:
        graficar_termino(df_test, term) # Usaremos la función de graficar que definimos antes

def mostrar_mejores_predicciones(df_test, n=5):
    # 1. Calculamos el error absoluto por cada fila
    df_test['Error'] = np.abs(df_test[TARGET_COL] - df_test['Prediccion'])
    
    # 2. Calculamos el error medio por término
    error_por_termino = df_test.groupby('termino')['Error'].mean().reset_index()
    
    # 3. Los 5 con menor error
    mejores = error_por_termino.sort_values(by='Error', ascending=True).head(n)
    
    print(f"\n--- TOP {n} PREDICCIONES MÁS PRECISAS (MENOR ERROR) ---")
    print(mejores.to_string(index=False))
    
    for term in mejores['termino']:
        graficar_termino(df_test, term)

# ---------------- LÓGICA ----------------

def entrenar_modelo():
    print(f"--- ENTRENANDO MODELOS COMPARATIVOS ({TARGET_COL}) ---")
    
    if not os.path.exists(FILE_DATASET):
        print("ERROR: No encuentro el dataset de entrenamiento.")
        return

    df = pd.read_csv(FILE_DATASET)
    
    # 1. PREPARACIÓN DE DATOS
    
    # NUEVO: Calculamos la variable a predecir para el Modelo Delta (Crecimiento)
    df['TARGET_DELTA'] = df['TARGET_NEXT_3M_AVG'] - df['trends_lag_0']
    
    cols_metadata = ['mes', 'periodo', 'termino', 'interes', 'TARGET_NEXT_3M_AVG', 'TARGET_NEXT_6M_AVG', 'TARGET_DELTA']
    #cols_metadata = ['mes', 'periodo', 'termino', 'interes', 'TARGET_NEXT_3M_AVG', 'TARGET_NEXT_6M_AVG']
    
    # --- LA MAGIA DEL EXPERIMENTO ---
    # Features Modelo A (Completo)
    features_completas = [c for c in df.columns if c not in cols_metadata]
    
    # Features Modelo B (Ciego/Social): Quitamos el pasado de Google Trends
    #CAMBIAR PARA HACER PRUEBAS DE QUÉ PASA SI LE DEJAMOS ALGUNA PISTA DE GOOGLE (POR EJEMPLO, EL LAG_0 QUE ES EL MES ACTUAL)
    # ANTES: Le quitábamos todo el historial de Google
    #columnas_trampa = ['trends_lag_0', 'trends_lag_1', 'trends_lag_2', 'trends_mean_last_3m']

    # AHORA: Le dejamos 'trends_lag_0' (Dónde estamos hoy) para que tenga una base,
    # pero le quitamos el resto para obligarle a predecir el futuro usando TUS datos.
    columnas_trampa = ['trends_lag_1', 'trends_lag_2', 'trends_mean_last_3m']
    features_sociales = [c for c in features_completas if c not in columnas_trampa]

    # 2. DIVISIÓN ENTRENAMIENTO Y VALIDACIÓN
    df['fecha_dt'] = pd.to_datetime(df['mes'])
    fecha_split = pd.to_datetime(FECHA_CORTE)

    train = df[df['fecha_dt'] < fecha_split].copy()
    test = df[df['fecha_dt'] >= fecha_split].copy()

    y_train = train[TARGET_COL]
    y_test = test[TARGET_COL]

    # 3. ENTRENAMIENTO DE LOS DOS MODELOS
    print("\nEntrenando Modelo A (Completo)...")
    model_A = RandomForestRegressor(n_estimators=300, max_depth=8, max_features='sqrt', min_samples_leaf=3, random_state=42)
    model_A.fit(train[features_completas], y_train)

    print("Entrenando Modelo B (Solo Revistas y Redes)...")
    model_B = RandomForestRegressor(n_estimators=300, max_depth=8, max_features='sqrt', min_samples_leaf=3, random_state=42)
    model_B.fit(train[features_sociales], y_train)

    # Entrenamos el Modelo C (Prediciendo el Delta con datos sociales)
    print("Entrenando Modelo C (Buscador de Tendencias / Delta)...")
    model_C = RandomForestRegressor(n_estimators=300, max_depth=8, max_features='sqrt', min_samples_leaf=3, random_state=42)
    # Fíjate que le pasamos y_train_delta, no y_train
    y_train_delta = train['TARGET_DELTA'] 
    y_test_delta = test['TARGET_DELTA']
    
    model_C.fit(train[features_sociales], y_train_delta)
    
    # Feature Importance del Modelo C
    importancias_C = pd.DataFrame({
        'Feature': features_sociales,
        'Importancia': model_C.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    print("\nTOP 5 VARIABLES PARA PREDECIR CRECIMIENTO (Modelo Delta):")
    print(importancias_C.head(5).to_string(index=False))

    # 4. PREDICCIÓN Y EVALUACIÓN COMPARATIVA
    pred_A = model_A.predict(test[features_completas])
    pred_B = model_B.predict(test[features_sociales])
    
    # Predicción del Modelo C (Nos da el Delta)
    pred_C_delta = model_C.predict(test[features_sociales])
    
    # RECONSTRUCCIÓN: Para comparar de tú a tú con A y B, 
    # sumamos el Delta predicho al valor actual (trends_lag_0)
    pred_C_absoluta = pred_C_delta + test['trends_lag_0']

    mae_A = mean_absolute_error(y_test, pred_A)
    r2_A = r2_score(y_test, pred_A)
    
    mae_B = mean_absolute_error(y_test, pred_B)
    r2_B = r2_score(y_test, pred_B)
    
    mae_C = mean_absolute_error(y_test, pred_C_absoluta)
    r2_C = r2_score(y_test, pred_C_absoluta)
    
    print("\n--- RESULTADOS DEL ESTUDIO COMPARATIVO ---")
    print(f"{'Métrica':<10} | {'Mod. A (Completo)':<18} | {'Mod. B (Híbrido)':<18} | {'Mod. C (Delta)':<18}")
    print("-" * 75)
    print(f"{'MAE':<10} | {mae_A:<18.2f} | {mae_B:<18.2f} | {mae_C:<18.2f}")
    print(f"{'R²':<10} | {r2_A:<18.4f} | {r2_B:<18.4f} | {r2_C:<18.4f}")
    print("-" * 75)

    # 5. FEATURE IMPORTANCE DEL MODELO B (El que de verdad nos importa)
    importancias_B = pd.DataFrame({
        'Feature': features_sociales,
        'Importancia': model_B.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    print("\nTOP 5 VARIABLES DEL MODELO B (Sin trampas de Google):")
    print(importancias_B.head(5).to_string(index=False))

    # Guardamos las predicciones del modelo B en el test para las gráficas
    test['Prediccion'] = pred_B

    # 6. VISUALIZACIÓN DE EJEMPLOS (Mostrando el modelo B)
    terminos_test = test['termino'].unique()
    if len(terminos_test) > 0:
        ejemplos = np.random.choice(terminos_test, min(3, len(terminos_test)), replace=False)
        for term in ejemplos:
            subset = df[df['termino'] == term].sort_values('fecha_dt') 
            subset_test = test[test['termino'] == term].sort_values('fecha_dt')
            
            # Cogemos las predicciones de ambos modelos para compararlas visualmente
            subset_test_A = pred_A[test['termino'] == term]
            subset_test_B = pred_B[test['termino'] == term]
            
            plt.figure(figsize=(14, 6))
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)) 
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)

            plt.plot(subset['fecha_dt'], subset[TARGET_COL], label='Realidad (Target)', color='gray', alpha=0.4, linewidth=2)
            plt.plot(subset_test['fecha_dt'], subset_test_A, label='Modelo A (Con Histórico)', color='orange', linestyle=':', linewidth=2)
            plt.plot(subset_test['fecha_dt'], subset_test_B, label='Modelo B (Solo Revistas/Redes)', color='red', marker='x', markersize=4, linewidth=2)

            plt.axvline(x=fecha_split, color='black', linestyle='--', label='Corte Predicción')
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.title(f"Comparativa de IA: {term.upper()}")
            plt.ylabel("Interés / Score")
            plt.legend()
            plt.tight_layout() 
            plt.show()

if __name__ == "__main__":
    entrenar_modelo()