"""
Interfaz de usuario (Demo Cloud) - Predictor de Tendencias de Moda.
Proporciona un panel de control interactivo simulando el pipeline de extracción,
procesamiento multimodal y entrenamiento de modelos predictivos.
"""

import streamlit as st
import os
import time
import pandas as pd
import datetime
import glob

# ----------------- CONFIGURACIÓN DEL ENTORNO -----------------
st.set_page_config(page_title="Panel de Control - Predictor de Tendencias", layout="wide", initial_sidebar_state="expanded")

st.info("[INFO] **ENTORNO DEMOSTRATIVO (CLOUD):** Debido a limitaciones de cómputo en la nube, esta instancia opera sobre un conjunto de resultados pre-calculados procedentes de la última ejecución estable local, garantizando así la fluidez de la interfaz. NOTA: Los volúmenes de datos (cantidad de imágenes, URLs, etc.) mostrados en los logs de esta simulación son puramente ilustrativos para la demo; la volumetría masiva real del proyecto se detalla íntegramente en la memoria escrita.")
st.title("Sistema Híbrido Predictor de Tendencias - TFG")
st.markdown("Panel de control del pipeline de datos. Ejecute los módulos secuencialmente para procesar la información y generar las predicciones.")
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
    help="Punto temporal de división para evitar filtración de datos durante el entrenamiento."
)

# ----------------- FUNCIÓN DE EJECUCIÓN (MOCK) -----------------
def simular_script(nombre_script: str, tiempo_espera: int = 2, mensaje_exito: str = "Ejecución completada."):
    """Simula la ejecución en consola de los scripts del backend."""
    with st.spinner(f"Ejecutando módulo {nombre_script} en backend..."):
        time.sleep(tiempo_espera)
        st.success(f"{mensaje_exito}")
        
        with st.expander("Inspeccionar logs de ejecución"):
            st.code(f"Iniciando {nombre_script}...\nCargando variables de entorno:\n- MODO_RECOLECCIÓN: {modo_ejecucion}\n- HORIZONTE_PREDICTIVO: {ventana_prediccion}\n- FECHA_SPLIT: {fecha_corte.strftime('%Y-%m-%d')}\n[OK] Pipeline finalizado con código de salida 0.", language="shell")

# =========================================================================
# FASE 1: EXTRACCIÓN SOCIAL
# =========================================================================
st.header("Fase 1: Extracción en Redes Sociales")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pinterest")
    if st.button("1a. Inicializar Crawler (Pinterest)"):
        simular_script("pinterest_scraper_2.py", 2, "[INFO] 245 URLs de tableros recolectadas.")
            
    if st.button("2. Descarga de Archivos Visuales"):
        simular_script("pinterest_image_downloader.py", 3, "[INFO] 1200 imágenes descargadas y encoladas.")

with col2:
    st.subheader("Instagram")
    if st.button("1b. Ejecutar Extracción API (Instagram)"):
        simular_script("apify_hashtags_profiles.py", 2, "[INFO] Metadatos de perfiles e interacciones extraídos correctamente.")

st.markdown("---")

# =========================================================================
# FASE 2: VISIÓN ARTIFICIAL
# =========================================================================
st.header("Fase 2: Inferencia Visual (Zero-Shot Learning)")
st.write("Procesamiento semántico del corpus de imágenes recolectado mediante modelos transformadores.")

if st.button("3. Ejecutar Inferencia Multimodal (CLIP)"):
    simular_script("analisis_imagenes.py", 4, "[INFO] Modelo CLIP instanciado. 1500 imágenes categorizadas.")

st.markdown("---")

# =========================================================================
# FASE 3: NLP Y METADATOS
# =========================================================================
st.header("Fase 3: Procesamiento de Revistas Digitales")

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("4. Recolección de Endpoints"):
        simular_script("recolector_urls.py", 2, "[INFO] Mapa de URLs objetivo generado.")

with c2:
    if st.button("5. Parseo HTML (Extractor Híbrido)"):
        simular_script("extractor_hibrido.py", 3, "[INFO] Entidades (Texto, Autor, Tags) estructuradas.")

with c3:
    if st.button("6. Normalización Temporal"):
        simular_script("procesador_fechas.py", 1, "[INFO] Formatos relativos estandarizados a Datetime.")

st.markdown("---")

# =========================================================================
# FASE 4: ENTRENAMIENTO Y EVALUACIÓN
# =========================================================================
st.header("Fase 4: Ingeniería de Datos y Modelado")

if st.button("7. Consolidación de Datasets (Master Table)"):
    simular_script("unificar_datasets.py", 2, "[INFO] Integración completada. Matriz final: (2600, 45).")

if st.button("8. Generación de Variables Rezagadas (Lags)"):
    simular_script("crear_dataset_entrenamiento.py", 2, "[INFO] Features temporales y variables Delta generadas.")

st.markdown("### Evaluación de Modelos Predictivos")
if st.button("9. COMPILAR Y EVALUAR ALGORITMOS (RF, SVR, XGBoost)", type="primary"):
    with st.spinner("Optimizando hiperparámetros y calculando métricas de error..."):
        time.sleep(3) 
        st.balloons()
        st.success("Evaluación de rendimiento completada. Visualizando resultados.")
        
        st.subheader("Métricas de Precisión Absoluta")
        # Tabla con los valores exactos mostrados en la memoria
        resultados = pd.DataFrame({
            "Métrica": ["MAE", "RMSE", "R²"],
            "Modelo A (Histórico)": ["3.13", "5.55", "0.9501"],
            "Modelo B (Híbrido)": ["5.72", "8.38", "0.8864"],
            "Modelo C (RF Delta)": ["2.43", "4.45", "0.9680"],
            "Modelo D (SVR Delta)": ["2.38", "4.34", "0.9695"],
            "Modelo E (XGB Delta)": ["2.31", "4.15", "0.9721"]
        })
        st.table(resultados.set_index("Métrica"))
        
        st.subheader("Proyección Gráfica de Tendencias")
        carpeta_graficas = os.path.join(BASE_DIR, "resultados_graficas")
        
        if os.path.exists(carpeta_graficas):
            imagenes = glob.glob(os.path.join(carpeta_graficas, "*.png"))
            if imagenes:
                cols = st.columns(2)
                for i, img_path in enumerate(imagenes):
                    cols[i % 2].image(img_path, use_container_width=True)
            else:
                st.warning("[ALERTA] No se encontraron gráficas en el directorio local.")
        else:
            st.error("[ERROR] Directorio de salida 'resultados_graficas' no encontrado.")