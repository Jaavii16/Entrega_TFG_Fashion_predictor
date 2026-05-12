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
    print(f"--- ENTRENANDO MODELO ({TARGET_COL}) ---")
    
    if not os.path.exists(FILE_DATASET):
        print("ERROR: No encuentro el dataset de entrenamiento.")
        return

    df = pd.read_csv(FILE_DATASET)
    
    # 1. PREPARACIÓN DE DATOS
    # Separamos las columnas que no son features
    cols_metadata = ['mes', 'periodo', 'termino', 'interes', 'TARGET_NEXT_3M_AVG', 'TARGET_NEXT_6M_AVG']
    
    # Definimos X 
    # Todas las columnas menos los metadatos y los targets
    features = [c for c in df.columns if c not in cols_metadata]
    
    print(f"Features utilizadas ({len(features)}):")
    print(features)

    # 2. DIVISIÖN ENTRENAMIENTO Y VALIDACIÓN (Train vs Test)
    df['fecha_dt'] = pd.to_datetime(df['mes'])
    fecha_split = pd.to_datetime(FECHA_CORTE)

    train = df[df['fecha_dt'] < fecha_split].copy()
    test = df[df['fecha_dt'] >= fecha_split].copy()

    print(f"\nConjunto Train: {len(train)} muestras")
    print(f"Conjunto Test: {len(test)} muestras")

    if len(test) == 0:
        print("ERROR: El conjunto de Test está vacío")
        return

    # Definimos X e Y
    X_train = train[features]
    y_train = train[TARGET_COL]
    
    X_test = test[features]
    y_test = test[TARGET_COL]

    # 3. ENTRENAMIENTO DEL MODELO (Random Forest de momento, luego probar varios y comparar)
    # n_estimators=100 (100 árboles), random_state=42 (semilla fija para que sea repetible)
    print("\nEntrenando Random Forest...")
    model = RandomForestRegressor(
        n_estimators=300, 
        max_depth=8,         # Limitamos profundidad para que no memorice demasiado el entrenamiento
        max_features='sqrt', # Obligamos a elegir entre un subconjunto de variables en cada división, para que no dependa demasiado de unas pocas
        min_samples_leaf=3,  # Evita árboles demasiado específicos
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4. PREDICCIÓN
    print(f"Realizando predicciones sobre {FECHA_CORTE}...")
    predicciones = model.predict(X_test)

    # 5. EVALUACIÓN DEL MODELO
    # Miramos el error medio absoluto y el R²
    mae = mean_absolute_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)
    
    print("\nRESULTADOS DEL MODELO:")
    print(f"------------------------------------------------")
    print(f"MAE (Error Medio Absoluto): {mae:.2f}")
    print(f"R² (Capacidad de Explicación): {r2:.4f}")
    print(f"------------------------------------------------")
    # Cuanto más bajo el MAE y más alto el R², mejor es el modelo. Un R² cercano a 1 indica que el modelo explica bien la 
    # variabilidad de los datos, mientras que un MAE bajo indica que las predicciones son cercanas a los valores reales

    # 6. VISUALIZACIÓN DE UN EJEMPLO
    import matplotlib.dates as mdates # Asegúrate de tener esta importación al principio
    
    test['Prediccion'] = predicciones
    terminos_test = test['termino'].unique()

    if len(terminos_test) > 0:
        ejemplos = np.random.choice(terminos_test, min(3, len(terminos_test)), replace=False)
        for term in ejemplos:
            # Filtramos el histórico completo para la línea gris
            subset = df[df['termino'] == term].sort_values('fecha_dt') 
            
            # Esto es para la parte de test donde comparamos realidad vs predicción
            subset_test = test[test['termino'] == term].sort_values('fecha_dt')
            
            plt.figure(figsize=(14, 6))

            # Configuración de ejes
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)) 
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)

            # Dibujamos las líneas
            plt.plot(subset['fecha_dt'], subset[TARGET_COL], label='Realidad (Target)', color='gray', alpha=0.4, linewidth=2)
            plt.plot(subset_test['fecha_dt'], subset_test['Prediccion'], label='Predicción IA', color='red', marker='x', markersize=4, linewidth=2)

            plt.axvline(x=fecha_split, color='black', linestyle='--', label='Corte Predicción')
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.title(f"Predicción: {term.upper()}")
            plt.ylabel("Interés / Score")
            plt.legend()
            plt.tight_layout() # Para que no se corten las etiquetas de los meses
            plt.show()

    mostrar_mejores_predicciones(test, n=5)
    mostrar_top_tendencias(test, n=10)
    

    # 7. FEATURE IMPORTANCE
    # Esto sirve para ver qué variables fueron las que más influyeron en las predicciones del modelo
    importancias = pd.DataFrame({
        'Feature': features,
        'Importancia': model.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    print("\nTOP 5 VARIABLES MÁS IMPORTANTES:")
    print(importancias.head(5))

if __name__ == "__main__":
    entrenar_modelo()