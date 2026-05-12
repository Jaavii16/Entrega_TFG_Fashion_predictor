import pandas as pd
import numpy as np
import os
import sys

# ------------------ CONFIGURACION DE PESOS ------------------
W_SOURCE_REVISTAS = 2.0
W_SOURCE_INSTAGRAM = 1.0
W_SOURCE_PINTEREST = 0.3

W_AUTH_VIP = 1.5      
W_AUTH_EVENT = 1.2    
W_AUTH_NORMAL = 1.0   

LISTA_VIPS = ["haileybieber", "kendalljenner", "bellahadid", "asaprocky", "lilbieber", "lilyachty"]
LISTA_EVENTOS = ["pfw", "nyfw", "mfw", "cphfw", "lfw"]

# ----------------- RUTAS ------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

# Archivos de entrada
FILE_DIC = os.path.join(base_dir, "diccionario_moda.csv")
FILE_REVISTAS = os.path.join(base_dir, "dataset_revistas", "dataset_final_normalizado.csv") 
FILE_INSTA = os.path.join(base_dir, "dataset_instagram", "instagram_posts.csv")
FILE_PINT = os.path.join(base_dir, "dataset_pinterest", "pinterest_dataset_completo.csv")

# Archivo de salida
FILE_OUTPUT = os.path.join(base_dir, "dataset_maestro.csv")

# ----------------- FUNCIONES AUXILIARES ------------------

def cargar_diccionario(ruta):
    if not os.path.exists(ruta):
        print(f"[ERROR] No encuentro diccionario en {ruta}")
        sys.exit()
    try:
        df = pd.read_csv(ruta)
        
        # Esto lo hacemos por si el diccionario tiene varias columnas
        if 'category' in df.columns:
            df = df[df['category'] != 'parte_prenda']
        
        # Seleccion de columna segura
        if 'term_en' not in df.columns:
            col = df.columns[0]
        else:
            col = 'term_en'
            
        return df[col].dropna().str.lower().str.strip().unique().tolist()
    except Exception as e:
        print(f"[ERROR] Leyendo diccionario: {e}")
        sys.exit()

def obtener_autoridad(target):
    target = str(target).lower()
    if any(vip in target for vip in LISTA_VIPS): return W_AUTH_VIP
    if any(evt in target for evt in LISTA_EVENTOS): return W_AUTH_EVENT
    return W_AUTH_NORMAL

def procesar_texto_visual(df, col_texto, col_visual):
    """Combina columnas de texto y etiquetas visuales."""
    t = df[col_texto].fillna("").astype(str).str.lower()
    v = ""
    if col_visual in df.columns:
        v = df[col_visual].fillna("").astype(str).str.lower()
    return t + " " + v

# ----------------- PROCESAMIENTO POR FUENTE ----------------

def procesar_instagram(ruta, terminos):
    print(f"--- Procesando Instagram (Peso: {W_SOURCE_INSTAGRAM}) ---")
    if not os.path.exists(ruta): 
        print("   [AVISO] Archivo no encontrado.")
        return pd.DataFrame()
    
    df = pd.read_csv(ruta)
    print(f"   Registros cargados: {len(df)}")
    
    # 1. Fechas y Mes
    df['fecha_dt'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['fecha_dt'])
    df['mes'] = df['fecha_dt'].dt.to_period('M').astype(str)
    
    # 2. Viralidad y Autoridad 
    # Convertimos nulos a 0 y forzamos que no haya negativos (-1 pasa a 0)
    df['Likes'] = df['Likes'].fillna(0).astype(int)
    df['Likes'] = df['Likes'].clip(lower=0) 
    df['W_viral'] = 1 + np.log10(df['Likes'] + 1)
    
    df['W_auth'] = df['Target_Busqueda'].apply(obtener_autoridad)
    
    # 3. Texto Unificado
    df['full_text'] = procesar_texto_visual(df, 'Texto', 'Etiquetas_Visuales')
    
    resultados = []
    total_terminos = len(terminos)
    
    print("   Buscando terminos...")
    for i, term in enumerate(terminos):
        if i % 50 == 0: 
            print(f"   [{i}/{total_terminos}]", end="\r")
            sys.stdout.flush()
        
        # Conteo vectorizado
        counts = df['full_text'].str.count(term)
        mask = counts > 0
        
        if mask.any():
            subset = df[mask].copy()
            # Calculamos score y lo metemos en el dataframe en una columna temporal para evitar FutureWarning de pandas
            subset['temp_score'] = counts[mask] * W_SOURCE_INSTAGRAM * subset['W_auth'] * subset['W_viral']
            
            # Agrupar por mes y sumar directamente la columna
            grouped = subset.groupby('mes')['temp_score'].sum().reset_index(name='score_fila')
            grouped['termino'] = term
            resultados.append(grouped)

    if resultados:
        return pd.concat(resultados, ignore_index=True)
    return pd.DataFrame()

def procesar_pinterest(ruta, terminos):
    print(f"\n--- Procesando Pinterest (Peso: {W_SOURCE_PINTEREST}) ---")
    if not os.path.exists(ruta): 
        print("   [AVISO] Archivo no encontrado.")
        return pd.DataFrame()
    
    df = pd.read_csv(ruta)
    print(f"   Registros cargados: {len(df)}")
    
    df['fecha_dt'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['fecha_dt'])
    df['mes'] = df['fecha_dt'].dt.to_period('M').astype(str)
    
    df['full_text'] = procesar_texto_visual(df, 'Texto', 'Etiquetas_Visuales')
    
    resultados = []
    total_terminos = len(terminos)
    
    print("   Buscando terminos...")
    for i, term in enumerate(terminos):
        if i % 50 == 0: 
            print(f"   [{i}/{total_terminos}]", end="\r")
            sys.stdout.flush()
        
        counts = df['full_text'].str.count(term)
        mask = counts > 0
        
        if mask.any():
            subset = df[mask].copy()
            # Calculamos score en columna temporal
            subset['temp_score'] = counts[mask] * W_SOURCE_PINTEREST
            
            # Agrupar y sumar
            grouped = subset.groupby('mes')['temp_score'].sum().reset_index(name='score_fila')
            grouped['termino'] = term
            resultados.append(grouped)

    if resultados:
        return pd.concat(resultados, ignore_index=True)
    return pd.DataFrame()

def procesar_revistas(ruta, terminos):
    print(f"\n--- Procesando Revistas (Peso: {W_SOURCE_REVISTAS}) ---")
    if not os.path.exists(ruta): 
        print("   [AVISO] Archivo no encontrado.")
        return pd.DataFrame()
    
    df = pd.read_csv(ruta)
    print(f"   Registros cargados: {len(df)}")
    
    if 'Fecha' in df.columns:
        df['fecha_dt'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['fecha_dt'])
        df['mes'] = df['fecha_dt'].dt.to_period('M').astype(str)
    else:
        print("   [ERROR] No hay columna 'Fecha' en revistas.")
        return pd.DataFrame()
    
    # Asumimos que la columna de texto normalizada se llama 'Contenido' porque es la que generamos en el extractor
    col_texto = 'Contenido' if 'Contenido' in df.columns else 'Texto'
    df['full_text'] = df[col_texto].fillna("").astype(str).str.lower()
    
    resultados = []
    total_terminos = len(terminos)
    
    print("   Buscando terminos...")
    for i, term in enumerate(terminos):
        if i % 50 == 0: 
            print(f"   [{i}/{total_terminos}]", end="\r")
            sys.stdout.flush()
        
        counts = df['full_text'].str.count(term)
        mask = counts > 0 
        if mask.any():
            subset = df[mask].copy()
            # Calculamos score en columna temporal
            subset['temp_score'] = counts[mask] * W_SOURCE_REVISTAS
            
            # Agrupar y sumar
            grouped = subset.groupby('mes')['temp_score'].sum().reset_index(name='score_fila')
            grouped['termino'] = term
            resultados.append(grouped)
            
    if resultados:
        return pd.concat(resultados, ignore_index=True)
    return pd.DataFrame()

# ------------------ EJECUCION PRINCIPAL ----------------

if __name__ == "__main__":
    print("=== INICIANDO UNIFICADOR DE DATASETS ===")
    
    print("\n1. Cargando Diccionario...")
    terminos = cargar_diccionario(FILE_DIC)
    print(f"   Total terminos a rastrear: {len(terminos)}")
    
    # Procesar fuentes
    df_scores_insta = procesar_instagram(FILE_INSTA, terminos)
    df_scores_pint = procesar_pinterest(FILE_PINT, terminos)
    df_scores_revi = procesar_revistas(FILE_REVISTAS, terminos)
    
    print("\n\n2. Unificando Datos...")
    frames = []
    if not df_scores_insta.empty: frames.append(df_scores_insta)
    if not df_scores_pint.empty: frames.append(df_scores_pint)
    if not df_scores_revi.empty: frames.append(df_scores_revi)
    
    if not frames:
        print("[ERROR] No se genero ningun dato de ninguna fuente.")
        sys.exit()

    all_data = pd.concat(frames, ignore_index=True)
    
    print("   Agrupando scores finales...")
    df_master = all_data.groupby(['mes', 'termino'])['score_fila'].sum().reset_index()
    df_master.rename(columns={'score_fila': 'impact_score'}, inplace=True)
    
    # Ordenar
    df_master = df_master.sort_values(by=['mes', 'termino'])
    
    # Guardar
    df_master.to_csv(FILE_OUTPUT, index=False)
    
    print("\n-----------------------------------------------")
    print(f"DATASET MAESTRO GENERADO: {FILE_OUTPUT}")
    print(f"   Total filas: {len(df_master)}")
    print(f"   Meses cubiertos: {df_master['mes'].nunique()}")
    print("-----------------------------------------------")