import pandas as pd
from pytrends.request import TrendReq
import os
import time
import random

# ----------------- CONFIGURACIÓN -----------------
START_DATE = "2026-01-01" 
END_DATE = "2026-03-31"

TIMEFRAMES = [
    # ("2020-01-01", "2021-12-31"),
    # ("2022-01-01", "2023-12-31"),
    # ("2024-01-01", "2025-12-31"),
    ("2026-01-01", "2026-03-31")
]

TERMINOS_POR_DIA = 12
ANCHOR_TERM = "jeans"  # TÉRMINO ANCLA

base_dir = os.path.dirname(os.path.abspath(__file__))
DICCIONARIO_FILE = os.path.join(base_dir, "diccionario_moda.csv")
OUTPUT_FILE = os.path.join(base_dir, "google_trends_moda.csv")

# ----------------- FUNCIONES AUXILIARES -----------------

def guardar_incremental(df_nuevo, ruta_archivo):
    """
    Guarda los datos nuevos en el CSV. Si ya existe, los añade sin duplicar.
    """
    if df_nuevo is None or df_nuevo.empty:
        return

    if os.path.exists(ruta_archivo):
        try:
            # Leemos lo que ya había
            df_exist = pd.read_csv(ruta_archivo)
            # Concatenamos
            df_final = pd.concat([df_exist, df_nuevo])
            # Eliminamos duplicados (Mismo término, mismo mes)
            df_final = df_final.drop_duplicates(subset=['termino', 'mes'])
        except Exception as e:
            print(f"   [ERROR CRÍTICO AL LEER CSV] {e}")
            # Si falla la lectura, guardamos en un archivo de rescate para no perder nada
            ruta_rescate = ruta_archivo.replace(".csv", "_RESCATE.csv")
            df_nuevo.to_csv(ruta_rescate, index=False, mode='a', header=not os.path.exists(ruta_rescate))
            print(f"   [SALVADO] Datos guardados en {ruta_rescate}")
            return
    else:
        df_final = df_nuevo
        
    # Guardamos
    df_final.to_csv(ruta_archivo, index=False)
    print(f"[DISCO] Datos guardados correctamente.")

def obtener_datos_normalizados(pytrends, lista_objetivo, anchor):
    # La lista de petición es el Ancla + los 4 términos que vamos a consultar
    lista_peticion = [anchor] + lista_objetivo
    resultados = []

    max_intentos = 2
    intentos = 0
    ESPERA_429 = 180  # segundos (3 minutos)

    for inicio, fin in TIMEFRAMES:
        print(f"      ↳ Timeframe {inicio} → {fin}")

        intento = 0
        while intento <= max_intentos:
            try:
                time.sleep(random.uniform(2.0, 5.0))

                pytrends.build_payload(
                    lista_peticion,
                    timeframe=f"{inicio} {fin}"
                )

                time.sleep(random.uniform(1.5, 3.0))
                df = pytrends.interest_over_time()

                if df is None or df.empty:
                    print("        [INFO] Sin datos en este tramo")
                    break

                if 'isPartial' in df.columns:
                    df = df.drop(columns=['isPartial'])

                df = df.resample('ME').mean().reset_index()

                df_melt = df.melt(
                    id_vars=['date'],
                    var_name='termino',
                    value_name='interes'
                )

                df_final = df_melt[df_melt['termino'] != anchor].copy()
                df_final.rename(columns={'date': 'mes'}, inplace=True)
                df_final['mes'] = df_final['mes'].dt.strftime('%Y-%m')

                resultados.append(df_final)
                break  # éxito → salimos del while

            except Exception as e:
                if "429" in str(e):
                    if intento < max_intentos:
                        print(f"        [429] Bloqueo. Esperando {ESPERA_429//60} min y reintentando...")
                        time.sleep(ESPERA_429)
                        intento += 1
                        pytrends = TrendReq(hl='es-ES', tz=360, timeout=30)
                    else:
                        print("        [429] Segundo fallo. Abortando bloque.")
                        return None
                else:
                    print(f"        [ERROR] {e}")
                    return None

    if not resultados:
        return None

    df_total = pd.concat(resultados)
    df_total = df_total.drop_duplicates(subset=['termino', 'mes'])
    
    return df_total

# ----------------- PROCESO PRINCIPAL -----------------

def main():
    print("--- INICIANDO RECOLECTOR DE TRENDS ---")
    
    print("DEBUG: Iniciando Pytrends con timeout extendido (15, 30)...")
    pytrends = TrendReq(hl='es-ES', tz=360, timeout=30)
    # 1. Cargar Diccionario
    if not os.path.exists(DICCIONARIO_FILE):
        print(f"ERROR: No encuentro {DICCIONARIO_FILE}")
        return

    df_dic = pd.read_csv(DICCIONARIO_FILE)

    # Preparar términos
    terminos = pd.melt(df_dic, id_vars=['category'], value_vars=['term_en', 'term_es'], 
                       var_name='idioma', value_name='termino')
    terminos = terminos.dropna(subset=['termino']).drop_duplicates(subset=['termino'])
    terminos['termino'] = terminos['termino'].str.strip().str.lower()

    # Excluimos el ancla de objetivos
    terminos = terminos[terminos['termino'] != ANCHOR_TERM]

    # 2. Cargar estado actual para ver qué falta
    if os.path.exists(OUTPUT_FILE):
        df_exist = pd.read_csv(OUTPUT_FILE)
        if not df_exist.empty:
            df_exist['termino'] = df_exist['termino'].astype(str).str.strip().str.lower()
            
            # ¡NUEVO!: Filtramos para ver qué términos tienen ya datos de 2026
            # La columna 'mes' tiene formato 'YYYY-MM', así que buscamos los mayores a '2026-01'
            df_nuevo_periodo = df_exist[df_exist['mes'] >= '2026-01']
            
            # Ahora los "procesados" son solo los que ya tienen datos en 2026
            terminos_procesados = set(df_nuevo_periodo['termino'].unique())
        else:
            terminos_procesados = set()
    else:
        terminos_procesados = set()

    # Lista de pendientes
    pendientes = [t for t in terminos['termino'].tolist() if t not in terminos_procesados]

    print(f"Total términos en diccionario: {len(terminos)}")
    print(f"Ya procesados: {len(terminos_procesados)}")
    print(f"Pendientes: {len(pendientes)}")

    if not pendientes:
        print("Dataset completo! No hay nada pendiente")
        return

    # Lote de hoy
    lote_hoy = pendientes[:TERMINOS_POR_DIA]
    
    # Bloques de 4
    bloques = [lote_hoy[i:i+4] for i in range(0, len(lote_hoy), 4)]
    print(f"Procesando {len(bloques)} bloques ({len(lote_hoy)} términos) hoy...\n")

    # ----------------- BUCLE PRINCIPAL DE RECOLECCIÓN -----------------
    try:
        for i, bloque in enumerate(bloques):
            print(f"[{i+1}/{len(bloques)}] Consultando: {bloque} + [Ancla: {ANCHOR_TERM}]")
            
            df_bloque = obtener_datos_normalizados(pytrends, bloque, ANCHOR_TERM)
            
            if df_bloque is not None and not df_bloque.empty:
                # Añadir metadatos (categoría)
                df_bloque['termino'] = df_bloque['termino'].str.strip().str.lower()
                df_bloque = df_bloque.merge(terminos[['termino', 'category']], on='termino', how='left')
                
                # ----------------- GUARDADO INCREMENTAL DE SEGURIDAD -----------------
                guardar_incremental(df_bloque, OUTPUT_FILE)
            else:
                print("   [FAIL] No se obtuvieron datos de este bloque.")

            # Pausa aleatoria para evitar bloqueos
            if i < len(bloques) - 1: # No esperar en el último
                t = random.randint(45, 80)
                print(f"   Enfriando {t}s...")
                time.sleep(t)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPCIÓN USUARIO] Detectado Ctrl+C.")
        print("   Los datos anteriores ya se han guardado en el CSV.")
        print("   Cerrando el script de forma segura...")

    print(f"\n--- Fin de la ejecución. Archivo: {OUTPUT_FILE} ---")

if __name__ == "__main__":
    main()