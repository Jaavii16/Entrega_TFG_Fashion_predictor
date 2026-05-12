import pandas as pd
import numpy as np
import os

# ---------------- CONFIGURACIÓN ----------------
base_dir = os.path.dirname(os.path.abspath(__file__))

# Archivos de entrada
FILE_MAESTRO = os.path.join(base_dir, "dataset_maestro.csv")       
FILE_TRENDS = os.path.join(base_dir, "google_trends_moda.csv")    

# Salida
FILE_TRAIN = os.path.join(base_dir, "dataset_entrenamiento.csv")

# ---------------- LÓGICA ----------------

def preparar_dataset_ventana():
    print("PREPARANDO DATASET PARA ENTRENAMIENTO")

    if not os.path.exists(FILE_MAESTRO) or not os.path.exists(FILE_TRENDS):
        print("ERROR: Faltan archivos de entrada.")
        return

    df_x = pd.read_csv(FILE_MAESTRO) 
    df_y = pd.read_csv(FILE_TRENDS)  

    # Limpieza
    df_x['termino'] = df_x['termino'].str.strip().str.lower()
    df_y['termino'] = df_y['termino'].str.strip().str.lower()

    # Creamos columna de periodo para facilitar cálculos temporales (mes y año juntos) (aunque en principio ya viene en este formato
    # porque lo hemos normalizado antes, pero por si acaso, para que no se rompa nada si el formato cambia un poco)
    df_x['periodo'] = pd.to_datetime(df_x['mes']).dt.to_period('M')
    df_y['periodo'] = pd.to_datetime(df_y['mes']).dt.to_period('M')

    # 1. MERGE (Unimos Inputs y Datos Reales actuales)
    print("Cruzando datos...")
    # Inner join para tener solo filas donde tengamos datos de ambos datasets (inputs y targets)
    df_full = pd.merge(df_x, df_y, on=['periodo', 'termino'], how='inner')
    # Ordenamos para que los cálculos temporales funcionen
    df_full = df_full.sort_values(by=['termino', 'periodo'])
    
    print(f"Filas base tras cruce: {len(df_full)}")

    # ---------------- INPUTS ----------------
    # Variables que el modelo usará para aprender (score de redes sociales de los meses anteriores, estacionalidad, categoría, etc.)
    
    print("Generando variables del pasado...")
    # Vemos cuanto se habló de cada término en redes sociales en los meses anteriores
    for i in [1, 2, 3]:
        df_full[f'score_lag_{i}'] = df_full.groupby('termino')['impact_score'].shift(i)

    # Promedio del score en redes de los últimos 3 meses
    df_full['score_mean_last_3m'] = df_full.groupby('termino')['impact_score'].transform(
        lambda x: x.rolling(window=3).mean().shift(1)
    )

    # Crecimiento (diferencia mes a mes)
    # Esto indica si el ruido en redes está explotando o muriendo
    df_full['social_diff'] = df_full.groupby('termino')['impact_score'].diff().shift(1)
    
    # Crecimiento porcentual (evitando infinitos por ceros)
    df_full['social_growth'] = df_full.groupby('termino')['impact_score'].pct_change().shift(1).replace([np.inf, -np.inf], 0).fillna(0)

    #Google Trends
    # El modelo necesita saber cuánto interés había el mes pasado para proyectar el futuro
    for i in [0, 1, 2]: # 0 es el mes actual, 1 es hace un mes, etc
        # Si predecimos en el momento T, conocemos el interés de Google Trends hasta T incluido
        df_full[f'trends_lag_{i}'] = df_full.groupby('termino')['interes'].shift(i)

    # Tendencia de Trends en los últimos 3 meses para saber si el interés está subiendo o bajando
    df_full['trends_mean_last_3m'] = df_full.groupby('termino')['interes'].transform(
        lambda x: x.rolling(window=3).mean().shift(0) # Shift 0 porque conocemos el pasado hasta hoy
    )

    # Estacionalidad (Seno/Coseno)
    # Usamos la fecha para crear variables de estacionalidad. Esto ayuda al modelo a entender patrones cíclicos
    # Guardamos seno y coseno del mes y luego se usarán como features en el modelo. 
    # Esto se suele usar para que el modelo pueda aprender patrones estacionales como que en verano sube el interés por bikinis, etc
    df_full['mes_num'] = df_full['periodo'].dt.month
    df_full['sin_mes'] = np.sin(2 * np.pi * df_full['mes_num'] / 12)
    df_full['cos_mes'] = np.cos(2 * np.pi * df_full['mes_num'] / 12)

    # One-Hot Encoding de Categorías
    # Creamos variables binarias para cada categoría de término
    if 'category' in df_full.columns:
        dummies = pd.get_dummies(df_full['category'], prefix='cat', dtype=int)
        df_full = pd.concat([df_full, dummies], axis=1)
        # Borramos la columna de texto original para que no rompa el entrenamiento
        df_full.drop(columns=['category'], inplace=True)

    # ---------------- TARGETS ----------------
    
    print("Calculando ventanas futuras...")

    # Función auxiliar para calcular media futura
    # Shift negativo (-1) mira hacia delante.
    
    # --- VENTANA 3 MESES ---
    # Cogemos los valores de trends de T+1, T+2 y T+3
    grouped = df_full.groupby('termino')['interes']
    t1 = grouped.shift(-1)
    t2 = grouped.shift(-2)
    t3 = grouped.shift(-3)
    
    # Hacemos la media.
    df_full['TARGET_NEXT_3M_AVG'] = (t1 + t2 + t3) / 3

    # --- VENTANA 6 MESES ---
    futures = []
    for i in range(1, 7):
        futures.append(grouped.shift(-i))
    
    # Concatenamos y hacemos media por columnas
    df_temp_6m = pd.concat(futures, axis=1)
    df_full['TARGET_NEXT_6M_AVG'] = df_temp_6m.mean(axis=1)

    # ---------------- LIMPIEZA FINAL ----------------
    
    # Eliminamos filas que tengan NaN en los inputs o en los targets y columnas que no aporten información (que han podido quedar tras el merge)
    cols_basura = ['mes_x', 'mes_y'] 
    df_full.drop(columns=[c for c in cols_basura if c in df_full.columns], inplace=True)
    df_final = df_full.dropna().copy()

    # Convertimos el objeto Periodo a string para que el CSV sea compatible con cualquier lector
    df_final['mes'] = df_final['periodo'].astype(str)

    # Guardado
    df_final.to_csv(FILE_TRAIN, index=False)
    
    print("\n--- RESUMEN DEL DATASET ---")
    print(f"Archivo generado: {FILE_TRAIN}")
    print(f"Muestras válidas para entrenar: {len(df_final)}")


if __name__ == "__main__":
    preparar_dataset_ventana()