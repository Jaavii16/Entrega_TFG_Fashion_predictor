import pandas as pd
import os
import re
import random
from datetime import datetime, timedelta

# ----------------- CONFIGURACIÓN -----------------
base_dir = os.path.dirname(os.path.abspath(__file__))
FILE_INSTA = os.path.join(base_dir, "dataset_instagram", "instagram_posts.csv")
FILE_SALIDA = os.path.join(base_dir, "dataset_instagram", "instagram_posts_CORREGIDO.csv")

# ----------------- LÓGICA DE REPARACIÓN -----------------

def obtener_anio_de_target(target):
    """Extrae 2020, 2021... o 20, 21... del hashtag"""
    target = str(target).lower()
    
    # Busca 4 dígitos (2020-2026)
    match_long = re.search(r'(202[0-6])', target)
    if match_long:
        return int(match_long.group(1))
    
    # Busca 2 dígitos al final (pfw22, streetstyle21, etc)
    match_short = re.search(r'([2][0-6])$', target) # Busca que termine en 20, 21... 26
    if match_short:
        return 2000 + int(match_short.group(1))
        
    return None

def generar_fecha_inteligente(anio_target, texto_target):
    """Genera una fecha coherente con la estación mencionada en el hashtag"""
    texto_target = str(texto_target).lower()
    
    # Rango de meses por defecto (Enero a Diciembre)
    mes_inicio, mes_fin = 1, 12
    
    # 1. Tratamos de detectar la estación
    if any(x in texto_target for x in ['ss', 'spring', 'summer']):
        mes_inicio, mes_fin = 4, 8 # Abril - Agosto
        
    elif any(x in texto_target for x in ['fw', 'fall', 'winter']):
        mes_inicio, mes_fin = 9, 12 # Septiembre - Diciembre (Simplificado)
        
    # 2. Detectar Fashion Weeks (Suelen ser Feb y Sept)
    elif any(x in texto_target for x in ['fw', 'fashionweek', 'pfw', 'nyfw', 'mfw', 'cphfw', 'lfw']):
        # Elegimos al azar entre la edición de Febrero o Septiembre porque no podemos saber cuál es exactamente
        mes_inicio = random.choice([2, 9])
        mes_fin = mes_inicio # Solo ese mes, para ser más precisos
        
    # Generar día aleatorio
    try:
        start_date = datetime(anio_target, mes_inicio, 1)
        if mes_fin == 12:
            end_date = datetime(anio_target, 12, 31)
        else:
            if mes_fin == 2: # Febrero es distinto asi que hay que hacerlo separado
                 end_date = datetime(anio_target, 2, 28)
            else:
                 end_date = datetime(anio_target, mes_fin, 30)
                 
        dias_totales = (end_date - start_date).days
        dias_random = random.randint(0, max(0, dias_totales))
        
        fecha_final = start_date + timedelta(days=dias_random)
        return fecha_final.strftime("%Y-%m-%d")
        
    except:
        # Fallback si falla algo raro con las fechas
        return f"{anio_target}-06-15" # Fecha genérica a mitad de año

def reparar_csv(ruta_entrada, ruta_salida):
    if not os.path.exists(ruta_entrada):
        print("No encuentro el archivo de entrada.")
        return

    print(f"Leyendo {ruta_entrada}...")
    df = pd.read_csv(ruta_entrada)
    
    corregidos = 0
    total = len(df)
    
    # Aseguramos formato fecha
    df['fecha_dt'] = pd.to_datetime(df['Fecha'], errors='coerce')
    
    for i, row in df.iterrows():
        target = row['Target_Busqueda']
        fecha_actual = row['fecha_dt']
        
        anio_objetivo = obtener_anio_de_target(target)
        
        if anio_objetivo:
            # Si la fecha ya es del año correcto, no hacemos nada. Si no, la corregimos.
            if pd.notna(fecha_actual) and fecha_actual.year == anio_objetivo:
                # Todo correcto, la foto se subió el mismo año del hashtag
                continue
            else:
                # La fecha no coincide con el año del hashtag, la corregimos
                nueva_fecha = generar_fecha_inteligente(anio_objetivo, target)
                df.at[i, 'Fecha'] = nueva_fecha
                corregidos += 1
    
    # Limpieza columna temporal
    if 'fecha_dt' in df.columns:
        del df['fecha_dt']
        
    df.to_csv(ruta_salida, index=False)
    print(f"\nReparación completada.")
    print(f"   - Total filas: {total}")
    print(f"   - Fechas corregidas: {corregidos}")
    print(f"   - Archivo guardado en: {ruta_salida}")

if __name__ == "__main__":
    reparar_csv(FILE_INSTA, FILE_SALIDA)