"""
Interfaz de usuario (Ejecución Local Real) - Predictor de Tendencias de Moda.
Proporciona un panel de control interactivo para la ejecución real del pipeline
de extracción, procesamiento multimodal y entrenamiento de modelos predictivos.
"""

import streamlit as st
import subprocess
import os
import sys
import datetime
import glob 

# ----------------- CONFIGURACIÓN DEL ENTORNO -----------------
st.set_page_config(page_title="Panel de Control - Predictor de Tendencias", layout="wide", initial_sidebar_state="expanded")

st.title("Predictor de Tendencias de Moda Basado en Redes Sociales - TFG de Javier")
st.markdown("Bienvenido al panel de control del TFG. Siga el orden de ejecución para procesar los datos y entrenar la Inteligencia Artificial.")
st.markdown("---")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------- PANEL LATERAL (CONFIGURACIÓN) -----------------
st.sidebar.header("Configuración Global")
modo_ejecucion = st.sidebar.radio(
    "Modo de recolección:", 
    ("HISTORICO", "DIARIO"),
    help="Histórico recopila datos desde 2020. Diario solo recopila lo actual."
)

ventana_prediccion = st.sidebar.selectbox(
    "Ventana de Predicción (Target):",
    ("TARGET_NEXT_3M_AVG", "TARGET_NEXT_6M_AVG"),
    help="Selecciona qué variable quieres que el modelo prediga: la popularidad promedio de los próximos 3 o 6 meses."
)

fecha_corte = st.sidebar.date_input(
    "Fecha de partición (Train/Test Split):",
    datetime.date(2025, 1, 1),
    help="Los datos estríctamente anteriores a esta fecha se usarán para entrenar. Los posteriores se aíslan para validar el modelo."
)

# ----------------- NÚCLEO DE EJECUCIÓN REAL (BACKEND) -----------------
def ejecutar_script(nombre_script: str):
    """Ejecuta los scripts reales inyectando las variables de la UI."""
    ruta_script = os.path.join(BASE_DIR, nombre_script)
    if not os.path.exists(ruta_script):
        st.error(f"[ERROR] No se encuentra el script {nombre_script}")
        return False, ""
    
    entorno_actual = os.environ.copy()
    entorno_actual["MODO_EJECUCION"] = modo_ejecucion
    entorno_actual["TARGET_COL"] = ventana_prediccion
    entorno_actual["FECHA_CORTE"] = fecha_corte.strftime("%Y-%m-%d")
    
    with st.spinner(f"Ejecutando módulo {nombre_script} en backend..."):
        try:
            proceso = subprocess.run(
                [sys.executable, ruta_script], 
                capture_output=True, text=True, check=True, env=entorno_actual
            )
            st.success(f"[OK] Ejecución completada.")
            with st.expander("Inspeccionar logs de ejecución"):
                st.code(proceso.stdout)
            return True, proceso.stdout 
        except subprocess.CalledProcessError as e:
            st.error(f"[ERROR] Fallo en la ejecución de {nombre_script}.")
            with st.expander("Inspeccionar traza de error"):
                st.code(e.stderr)
            return False, ""

# =========================================================================
# FASE 1: EXTRACCIÓN SOCIAL
# =========================================================================
st.header("Fase 1: Extracción en Redes Sociales")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pinterest")
    if st.button("1a. Ejecutar Scraper de Pinterest"):
        ejecutar_script("pinterest_scraper_2.py")
            
    if st.button("2. Descarga de Imágenes (Pinterest)"):
        ejecutar_script("pinterest_image_downloader.py")

with col2:
    st.subheader("Instagram")
    if st.button("1b. Ejecutar Scraper de Instagram"):
        ejecutar_script("apify_hashtags_profiles.py")

st.markdown("---")

# =========================================================================
# FASE 2: VISIÓN ARTIFICIAL
# =========================================================================
st.header("Fase 2: Inferencia Visual (Zero-Shot Learning)")
st.write("Procesamiento semántico del corpus de imágenes recolectado mediante modelos transformadores.")

if st.button("3. Ejecutar Análisis de Imágenes (CLIP)"):
    ejecutar_script("analisis_imagenes.py")

st.markdown("---")

# =========================================================================
# FASE 3: NLP Y METADATOS
# =========================================================================
st.header("Fase 3: Procesamiento de Revistas Digitales")

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("4. Recolector de URLs (Revistas Digitales)"):
        ejecutar_script("recolector_urls.py")

with c2:
    if st.button("5. Extractor Híbrido"):
        ejecutar_script("extractor_hibrido.py")

with c3:
    if st.button("6. Procesador de Fechas para normalizar formatos"):
        ejecutar_script("procesador_fechas.py")

st.markdown("---")

# =========================================================================
# FASE 4: ENTRENAMIENTO Y EVALUACIÓN
# =========================================================================
st.header("Fase 4: Ingeniería de Datos y Modelado")

if st.button("7. Unificación de Datasets"):
    ejecutar_script("unificar_datasets.py")

if st.button("8. Generación de dataset de entrenamiento"):
    ejecutar_script("crear_dataset_entrenamiento.py")

st.markdown("### Evaluación de Modelos Predictivos")
if st.button("9. ENTRENAR Y EVALUAR ALGORITMOS (RF, SVR, XGBoost)", type="primary"):
    
    # Limpiamos gráficas antiguas antes de ejecutar el modelo real
    carpeta_graficas = os.path.join(BASE_DIR, "resultados_graficas")
    if os.path.exists(carpeta_graficas):
        for file in glob.glob(os.path.join(carpeta_graficas, "*.png")):
            os.remove(file)
            
    exito, consola = ejecutar_script("entrenamiento_3modelos.py")
    
    if exito:
        st.subheader("Métricas de Precisión Absoluta")
        # Buscamos la tabla en la consola impresa
        if "--- RESULTADOS DEL ESTUDIO COMPARATIVO ---" in consola:
            tabla_texto = consola.split("--- RESULTADOS DEL ESTUDIO COMPARATIVO ---")[1].split("TOP 5 VARIABLES")[0]
            st.code(tabla_texto.strip(), language="plaintext")
        else:
            st.warning("[ALERTA] No se pudo extraer la tabla, revisa la consola completa arriba.")

        st.subheader("Proyección Gráfica de Tendencias")
        if os.path.exists(carpeta_graficas):
            imagenes = glob.glob(os.path.join(carpeta_graficas, "*.png"))
            if imagenes:
                cols = st.columns(2)
                for i, img_path in enumerate(imagenes):
                    cols[i % 2].image(img_path, use_container_width=True)
            else:
                st.warning("[ALERTA] No se encontraron gráficas en el directorio local tras la ejecución.")
        else:
            st.error("[ERROR] Directorio de salida 'resultados_graficas' no encontrado.")