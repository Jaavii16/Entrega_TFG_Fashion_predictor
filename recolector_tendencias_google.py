import pandas as pd
from pytrends.request import TrendReq
import os
import time
import random

# ----------------- CONFIGURACIÓN DEL ENTORNO Y EXTRACCIÓN -----------------
# Variables globales para escalabilidad: Solo se necesita modificar estas fechas
# para recolectar nuevos horizontes temporales en el futuro.
PERIOD_START = "2026-01-01" 
PERIOD_END = "2026-05-31"

TIMEFRAMES = [(PERIOD_START, PERIOD_END)]

TERMINOS_POR_DIA = 12
ANCHOR_TERM = "jeans"

base_dir = os.path.dirname(os.path.abspath(__file__))
DICCIONARIO_FILE = os.path.join(base_dir, "diccionario_moda.csv")
OUTPUT_FILE = os.path.join(base_dir, "google_trends_moda.csv")

# ----------------- FUNCIONES AUXILIARES -----------------

def guardar_incremental(df_nuevo: pd.DataFrame, ruta_archivo: str):
    """
    Persiste los datos extraídos en el archivo de salida de forma incremental,
    asegurando la eliminación de duplicados por término y fecha.
    """
    if df_nuevo is None or df_nuevo.empty:
        return

    if os.path.exists(ruta_archivo):
        try:
            df_exist = pd.read_csv(ruta_archivo)
            df_final = pd.concat([df_exist, df_nuevo])
            df_final = df_final.drop_duplicates(subset=['termino', 'mes'])
        except Exception as e:
            print(f"   [CRITICAL ERROR] Fallo de lectura/escritura: {e}")
            ruta_rescate = ruta_archivo.replace(".csv", "_RESCATE.csv")
            df_nuevo.to_csv(ruta_rescate, index=False, mode='a', header=not os.path.exists(ruta_rescate))
            print(f"   [RECUPERACIÓN] Datos volcados en archivo de seguridad: {ruta_rescate}")
            return
    else:
        df_final = df_nuevo
        
    df_final.to_csv(ruta_archivo, index=False)
    print(f"[INFO] Completado. Datos guardados en disco.")

def obtener_datos_normalizados(pytrends: TrendReq, lista_objetivo: list, anchor: str) -> pd.DataFrame:
    """
    Realiza la petición a la API de Google Trends normalizando los resultados
    respecto a un término ancla para mantener la coherencia transversal de las métricas.
    """
    lista_peticion = [anchor] + lista_objetivo
    resultados = []

    max_intentos = 2
    ESPERA_429 = 180  

    for inicio, fin in TIMEFRAMES:
        print(f"Extrayendo ventana temporal: {inicio} → {fin}")

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
                    print("        [WARNING] No se reportan datos para esta ventana.")
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
                break  

            except Exception as e:
                if "429" in str(e):
                    if intento < max_intentos:
                        print(f"        [RATE LIMIT 429] Bloqueo detectado. Suspendiendo ejecución {ESPERA_429//60} min...")
                        time.sleep(ESPERA_429)
                        intento += 1
                        pytrends = TrendReq(hl='es-ES', tz=360, timeout=30)
                    else:
                        print("        [ERROR] Máximo número de reintentos alcanzado por Rate Limiting.")
                        return None
                else:
                    print(f"        [ERROR] Excepción no controlada: {e}")
                    return None

    if not resultados:
        return None

    df_total = pd.concat(resultados)
    df_total = df_total.drop_duplicates(subset=['termino', 'mes'])
    
    return df_total

# ----------------- FLUJO DE EJECUCIÓN PRINCIPAL -----------------

def main():
    print("--- INICIANDO PIPELINE DE GOOGLE TRENDS ---")
    
    pytrends = TrendReq(hl='es-ES', tz=360, timeout=30)
    
    if not os.path.exists(DICCIONARIO_FILE):
        print(f"[ERROR] Dependencia ausente: {DICCIONARIO_FILE}")
        return

    df_dic = pd.read_csv(DICCIONARIO_FILE)

    terminos = pd.melt(df_dic, id_vars=['category'], value_vars=['term_en', 'term_es'], 
                       var_name='idioma', value_name='termino')
    terminos = terminos.dropna(subset=['termino']).drop_duplicates(subset=['termino'])
    terminos['termino'] = terminos['termino'].str.strip().str.lower()

    terminos = terminos[terminos['termino'] != ANCHOR_TERM]

    # Dinamización del horizonte temporal: Extraemos "YYYY-MM" directamente de PERIOD_START
    start_month_str = PERIOD_START[:7]

    if os.path.exists(OUTPUT_FILE):
        df_exist = pd.read_csv(OUTPUT_FILE)
        if not df_exist.empty:
            df_exist['termino'] = df_exist['termino'].astype(str).str.strip().str.lower()
            
            # Filtro dinámico basado en la configuración global
            df_nuevo_periodo = df_exist[df_exist['mes'] >= start_month_str]
            terminos_procesados = set(df_nuevo_periodo['termino'].unique())
        else:
            terminos_procesados = set()
    else:
        terminos_procesados = set()

    pendientes = [t for t in terminos['termino'].tolist() if t not in terminos_procesados]

    print(f"[INFO] Terminos en diccionario: {len(terminos)}")
    print(f"[INFO] Terminos procesadas ({start_month_str} en adelante): {len(terminos_procesados)}")
    print(f"[INFO] Terminos en cola: {len(pendientes)}")

    if not pendientes:
        print("[INFO] Dataset actualizado. No se requieren nuevas peticiones.")
        return

    lote_ejecucion = pendientes[:TERMINOS_POR_DIA]
    
    # Segmentación en bloques de 4 términos (+1 ancla = 5 términos máximos por petición)
    bloques = [lote_ejecucion[i:i+4] for i in range(0, len(lote_ejecucion), 4)]
    print(f"[INFO] Configuración de lote: {len(bloques)} bloques ({len(lote_ejecucion)} entidades).\n")

    try:
        for i, bloque in enumerate(bloques):
            print(f"[{i+1}/{len(bloques)}] Evaluando clúster: {bloque} | Ancla: '{ANCHOR_TERM}'")
            
            df_bloque = obtener_datos_normalizados(pytrends, bloque, ANCHOR_TERM)
            
            if df_bloque is not None and not df_bloque.empty:
                df_bloque['termino'] = df_bloque['termino'].str.strip().str.lower()
                df_bloque = df_bloque.merge(terminos[['termino', 'category']], on='termino', how='left')
                
                guardar_incremental(df_bloque, OUTPUT_FILE)
            else:
                print("   [ALERTA] Omisión de bloque por falta de datos o error de red.")

            # Rutina de evasión de bloqueos
            if i < len(bloques) - 1: 
                t = random.randint(45, 80)
                print(f"   [PAUSA] Suspendiendo hilo {t}s para mitigar Rate Limiting...")
                time.sleep(t)

    except KeyboardInterrupt:
        print("\n\n[INFO] Señal de interrupción (SIGINT) recibida.")
        print("   Finalizando volcado en disco de transacciones pendientes...")
        print("   Terminación segura completada.")

    print(f"\n--- Ejecución finalizada. Archivo resultante: {OUTPUT_FILE} ---")

if __name__ == "__main__":
    main()