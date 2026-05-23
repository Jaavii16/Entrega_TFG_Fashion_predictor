import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

# ---------------- CONFIGURACIÓN ----------------
base_dir = os.path.dirname(os.path.abspath(__file__))
FILE_DATASET = os.path.join(base_dir, "dataset_entrenamiento.csv")

# FECHA DE CORTE PARA VALIDACIÓN
# Para leer la fecha de corte desde Streamlit
FECHA_CORTE = os.environ.get("FECHA_CORTE", "2025-09-01") 

# TARGET_COL = "TARGET_NEXT_3M_AVG" 
# TARGET_COL = "TARGET_NEXT_6M_AVG"
# AHORA:
TARGET_COL = os.environ.get("TARGET_COL", "TARGET_NEXT_3M_AVG")

# ---------------- FUNCIONES AUXILIARES ----------------

def graficar_termino(df_completo, term):
    # Filtramos datos
    subset = df_completo[df_completo['termino'] == term].sort_values('fecha_dt')
    # Nos aseguramos de quedarnos solo con las filas que tienen el futuro
    subset_test = subset.dropna(subset=['Prediccion']) 

    plt.figure(figsize=(16, 6))
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.xticks(rotation=90, fontsize=9)

    plt.plot(subset['fecha_dt'], subset[TARGET_COL], label='Realidad (Target)', color='gray', alpha=0.4, linewidth=2)
    plt.plot(subset_test['fecha_dt'], subset_test['Prediccion'], label='Predicción IA', color='red', marker='x', markersize=6, linewidth=2)

    plt.axvline(x=pd.to_datetime(FECHA_CORTE), color='black', linestyle='--', label='Corte Predicción')
    plt.title(f"Análisis de Tendencia: {term.upper()}", fontsize=14, fontweight='bold')
    plt.ylabel('Índice de Interés (0-100)', fontsize=10) 
    plt.xlabel('Fecha', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def mostrar_top_tendencias(df_test, n=10):
    """Muestra las top n tendencias predichas para el próximo trimestre, ordenadas por la predicción de interés."""
    ultimo_mes = df_test['fecha_dt'].max()
    df_reciente = df_test[df_test['fecha_dt'] == ultimo_mes]
    
    top_terminos = df_reciente.sort_values(by='Prediccion', ascending=False).head(n)
    
    print(f"\n--- TOP {n} TENDENCIAS PREDICHAS PARA EL PRÓXIMO TRIMESTRE ---")
    print(top_terminos[['termino', 'Prediccion']].to_string(index=False))
    
    for term in top_terminos['termino']:
        graficar_termino(df_test, term)

def mostrar_mejores_predicciones(df_test, n=5):
    df_test['Error'] = np.abs(df_test[TARGET_COL] - df_test['Prediccion'])
    error_por_termino = df_test.groupby('termino')['Error'].mean().reset_index()
    mejores = error_por_termino.sort_values(by='Error', ascending=True).head(n)
    
    print(f"\n--- TOP {n} PREDICCIONES MÁS PRECISAS (MENOR ERROR) ---")
    print(mejores.to_string(index=False))
    
    for term in mejores['termino']:
        graficar_termino(df_test, term)

# ---------------- LÓGICA ----------------

def entrenar_modelo():
    print(f"--- ENTRENANDO MODELOS ({TARGET_COL}) ---")
    
    if not os.path.exists(FILE_DATASET):
        print("ERROR: No se encuentra el dataset de entrenamiento.")
        return

    df = pd.read_csv(FILE_DATASET)
    
    # -----------------------------------------------------------------------------------------------

    # 1. PREPARACIÓN DE DATOS
    df['TARGET_DELTA'] = df[TARGET_COL] - df['trends_lag_0'] 
    
    cols_metadata = ['mes', 'periodo', 'termino', 'interes', 'TARGET_NEXT_3M_AVG', 'TARGET_NEXT_6M_AVG', 'TARGET_DELTA']
    
    features_completas = [c for c in df.columns if c not in cols_metadata]
    columnas_trampa = ['trends_lag_1', 'trends_lag_2', 'trends_mean_last_3m']
    features_sociales = [c for c in features_completas if c not in columnas_trampa]

    # -----------------------------------------------------------------------------------------------

    # 2. DIVISIÓN: ENTRENAMIENTO Y VALIDACIÓN
    df['fecha_dt'] = pd.to_datetime(df['mes'])
    fecha_split = pd.to_datetime(FECHA_CORTE)

    train = df[df['fecha_dt'] < fecha_split].copy()
    test = df[df['fecha_dt'] >= fecha_split].copy()

    y_train = train[TARGET_COL]
    y_test = test[TARGET_COL]

    # ESCALADO DE DATOS
    print("Escalando datos ...")
    scaler = StandardScaler()
    
    X_train_social_scaled = scaler.fit_transform(train[features_sociales])
    X_test_social_scaled = scaler.transform(test[features_sociales])

    # -----------------------------------------------------------------------------------------------

    # 3. ENTRENAMIENTO DE LOS MODELOS
    print("\nEntrenando Modelo A (Completo)...")
    model_A = RandomForestRegressor(n_estimators=300, max_depth=8, max_features='sqrt', min_samples_leaf=3, random_state=42)
    model_A.fit(train[features_completas], y_train)

    print("Entrenando Modelo B (Solo Revistas y Redes)...")
    model_B = RandomForestRegressor(n_estimators=300, max_depth=8, max_features='sqrt', min_samples_leaf=3, random_state=42)
    model_B.fit(train[features_sociales], y_train)

    print("Entrenando Modelo C (Calculando el crecimiento)...")
    model_C = RandomForestRegressor(n_estimators=300, max_depth=8, max_features='sqrt', min_samples_leaf=3, random_state=42)
    y_train_delta = train['TARGET_DELTA'] 
    model_C.fit(train[features_sociales], y_train_delta)

    print("Entrenando Modelo D (SVR con datos escalados)...")
    model_D_svr = SVR(kernel='rbf', C=100, gamma='scale') 
    model_D_svr.fit(X_train_social_scaled, y_train_delta)

    print("Entrenando Modelo E (XGBoost prediciendo el crecimiento)...")
    model_E_xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model_E_xgb.fit(train[features_sociales], y_train_delta)
    
    importancias_E = pd.DataFrame({
        'Feature': features_sociales,
        'Importancia': model_E_xgb.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    print("\nTOP 5 VARIABLES PARA PREDECIR CRECIMIENTO (XGBoost - Modelo E):")
    print(importancias_E.to_string(index=False))

    # 4. PREDICCIÓN Y EVALUACIÓN COMPARATIVA
    pred_A = model_A.predict(test[features_completas])
    pred_B = model_B.predict(test[features_sociales])
    
    pred_C_delta = model_C.predict(test[features_sociales])
    pred_D_delta = model_D_svr.predict(X_test_social_scaled)
    pred_E_delta = model_E_xgb.predict(test[features_sociales])
    
    pred_C_absoluta = pred_C_delta + test['trends_lag_0']
    pred_D_absoluta = pred_D_delta + test['trends_lag_0']
    pred_E_absoluta = pred_E_delta + test['trends_lag_0']

    mae_A = mean_absolute_error(y_test, pred_A)
    r2_A = r2_score(y_test, pred_A)
    rmse_A = np.sqrt(mean_squared_error(y_test, pred_A))
    
    mae_B = mean_absolute_error(y_test, pred_B)
    r2_B = r2_score(y_test, pred_B)
    rmse_B = np.sqrt(mean_squared_error(y_test, pred_B))
    
    mae_C = mean_absolute_error(y_test, pred_C_absoluta)
    r2_C = r2_score(y_test, pred_C_absoluta)
    rmse_C = np.sqrt(mean_squared_error(y_test, pred_C_absoluta))

    mae_D = mean_absolute_error(y_test, pred_D_absoluta)
    r2_D = r2_score(y_test, pred_D_absoluta)
    rmse_D = np.sqrt(mean_squared_error(y_test, pred_D_absoluta))

    mae_E = mean_absolute_error(y_test, pred_E_absoluta)
    r2_E = r2_score(y_test, pred_E_absoluta)
    rmse_E = np.sqrt(mean_squared_error(y_test, pred_E_absoluta))
    
    print("\n--- RESULTADOS DEL ESTUDIO COMPARATIVO ---")
    print(f"{'Métrica':<10} | {'Mod. A (Completo)':<18} | {'Mod. B (Híbrido)':<18} | {'Mod. C (RF Delta)':<18} | {'Mod. D (SVR Delta)':<18} | {'Mod. E (XGB Delta)':<18}")
    print("-" * 115)
    print(f"{'MAE':<10} | {mae_A:<18.2f} | {mae_B:<18.2f} | {mae_C:<18.2f} | {mae_D:<18.2f} | {mae_E:<18.2f}")
    print(f"{'RMSE':<10} | {rmse_A:<18.2f} | {rmse_B:<18.2f} | {rmse_C:<18.2f} | {rmse_D:<18.2f} | {rmse_E:<18.2f}")
    print(f"{'R²':<10} | {r2_A:<18.4f} | {r2_B:<18.4f} | {r2_C:<18.4f} | {r2_D:<18.4f} | {r2_E:<18.4f}")
    print("-" * 115)

    
    # 5. INNER JOIN
    test['Pred_C_delta'] = pred_C_delta
    test['Pred_D_delta'] = pred_D_delta
    test['Pred_E_delta'] = pred_E_delta
    
    ultimo_mes = test['fecha_dt'].min()
    df_reciente = test[test['fecha_dt'] == ultimo_mes].copy()
    
    N = 15
    top_C = set(df_reciente.sort_values(by='Pred_C_delta', ascending=False).head(N)['termino'])
    top_D = set(df_reciente.sort_values(by='Pred_D_delta', ascending=False).head(N)['termino'])
    top_E = set(df_reciente.sort_values(by='Pred_E_delta', ascending=False).head(N)['termino'])
    
    consenso = top_C.intersection(top_D).intersection(top_E)

    print(f"\n--- TOP TENDENCIAS (INNER JOIN DEL TOP {N} DE CADA MODELO) ---")
    terminos_consenso = []
    if consenso:
        df_consenso = df_reciente[df_reciente['termino'].isin(consenso)].copy()
        df_consenso['crecimiento_medio'] = (df_consenso['Pred_C_delta'] + df_consenso['Pred_D_delta'] + df_consenso['Pred_E_delta']) / 3
        df_final = df_consenso.sort_values(by='crecimiento_medio', ascending=False)
        df_final = df_final.rename(columns={'TARGET_DELTA': 'Crecimiento_REAL'})
        print(df_final[['termino', 'crecimiento_medio', 'Crecimiento_REAL']].to_string(index=False))
        terminos_consenso = list(df_final['termino'])
    else:
        print(f"No hay términos que coincidan simultáneamente en el Top {N} de los tres modelos.")

    
    # Guardamos todas las predicciones en el dataframe de test para graficarlas
    test['Pred_A'] = pred_A
    test['Pred_B'] = pred_B
    test['Pred_C'] = pred_C_absoluta
    test['Pred_D'] = pred_D_absoluta
    test['Pred_E'] = pred_E_absoluta

    # 6. VISUALIZACIÓN INDIVIDUAL PARA LA MEMORIA
    # Unificamos los términos que queremos forzar (casos de estudio) con los que el modelo predice como top
    terminos_forzados = ['boots', 'silk']
    terminos_a_graficar = list(set(terminos_forzados + terminos_consenso))
    terminos_a_graficar = [t for t in terminos_a_graficar if t in test['termino'].unique()]
    
    for term in terminos_a_graficar:
        subset = df[df['termino'] == term].sort_values('fecha_dt') 
        subset_test = test[test['termino'] == term].sort_values('fecha_dt')
        
        modelos_a_graficar = {
            'Modelo A (Histórico)': ('orange', ':', subset_test['Pred_A']),
            'Modelo B (Híbrido)': ('purple', ':', subset_test['Pred_B']),
            'Modelo C (RF Delta)': ('red', '-', subset_test['Pred_C']),
            'Modelo D (SVR Delta)': ('blue', '--', subset_test['Pred_D']),
            'Modelo E (XGBoost Delta)': ('green', '-.', subset_test['Pred_E'])
        }

        for nombre_modelo, (color, estilo, prediccion) in modelos_a_graficar.items():
            plt.figure(figsize=(16, 6))
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=8)
            plt.margins(x=0.01)
            
            plt.plot(subset['fecha_dt'], subset[TARGET_COL], label='Realidad (Google Trends)', color='black', alpha=0.3, linewidth=5)
            plt.plot(subset_test['fecha_dt'], prediccion, label=nombre_modelo, color=color, linestyle=estilo, marker='o', markersize=5)
            
            plt.axvline(x=fecha_split, color='black', linestyle='--')
            
            plt.title(f"Predicción: {term.upper()} ({nombre_modelo})", fontsize=14, fontweight='bold')
            plt.ylabel('Índice de Interés (0-100)', fontsize=10)
            plt.xlabel('Fecha', fontsize=10)
            
            plt.legend()
            plt.tight_layout()
            
            carpeta_graficas = os.path.join(base_dir, "resultados_graficas")
            os.makedirs(carpeta_graficas, exist_ok=True)
            nombre_archivo = f"{term}_{nombre_modelo.split(' ')[1].replace('(', '').replace(')', '')}.png"
            plt.savefig(os.path.join(carpeta_graficas, nombre_archivo))
            plt.close()

if __name__ == "__main__":
    entrenar_modelo()