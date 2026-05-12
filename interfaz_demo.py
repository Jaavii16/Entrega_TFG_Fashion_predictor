import streamlit as st
import os
import time
import pandas as pd
import datetime # NUEVO: Para la fecha de corte
import glob     # NUEVO: Para leer las imágenes

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Demo TFG Javier", layout="wide", initial_sidebar_state="expanded")

# AVISO IMPORTANTE PARA EL TRIBUNAL / TUTORA
st.info("ℹ **MODO DEMOSTRACIÓN (CLOUD):** Esta interfaz está alojada en un servidor gratuito en la nube. Por limitaciones de hardware y tiempo de ejecución, los botones simulan la ejecución de los algoritmos de scraping y Deep Learning, mostrando resultados pre-calculados del entorno de desarrollo local.")

st.title("Predictor de Tendencias de Moda Basado en Redes Sociales - TFG de Javier")
st.markdown("Bienvenido al panel de control del TFG. Siga el orden de ejecución para procesar los datos y entrenar la Inteligencia Artificial.")
st.markdown("---")

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CONFIGURACIÓN GLOBAL (BARRA LATERAL) ---
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

# NUEVO: Selector de fecha de corte
fecha_corte = st.sidebar.date_input(
    "Fecha de corte (Train/Test Split):",
    datetime.date(2025, 1, 1),
    help="Los datos estrictamente anteriores a esta fecha se usarán para entrenar. Los posteriores se aíslan para validar el modelo."
)

# --- FUNCIÓN DE SIMULACIÓN PARA LA DEMO ---
def simular_script(nombre_script, tiempo_espera=2, mensaje="Ejecución completada."):
    with st.spinner(f"Simulando ejecución de {nombre_script} en la nube..."):
        # Pausa artificial para simular procesamiento
        time.sleep(tiempo_espera)
        st.success(f"{mensaje}")
        
        # Simulamos una salida de consola básica
        with st.expander("Ver log de consola simulado"):
            st.code(f"Iniciando {nombre_script}...\nCargando variables de entorno:\n- MODO: {modo_ejecucion}\n- TARGET: {ventana_prediccion}\n- FECHA_CORTE: {fecha_corte.strftime('%Y-%m-%d')}\nProcesamiento finalizado con éxito.", language="shell")

# =========================================================================
# 1. PINTEREST E INSTAGRAM
# =========================================================================
st.header("Fase 1: Redes Sociales (Pinterest e Instagram)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pinterest")
    if st.button("1a. Ejecutar Scraper de Pinterest"):
        simular_script("pinterest_scraper_2.py", 2, "245 URLs de tableros recolectadas.")
            
    if st.button("2. Descargar Imágenes (Pinterest)"):
        simular_script("pinterest_image_downloader.py", 3, "1200 imágenes descargadas en directorio local.")

with col2:
    st.subheader("Instagram")
    if st.button("1b. Ejecutar Scraper de Instagram"):
        simular_script("apify_hashtags_profiles.py", 2, "Datos de perfiles de influencers extraídos mediante API.")

st.markdown("---")

# =========================================================================
# 2. ANÁLISIS DE IMÁGENES
# =========================================================================
st.header("Fase 2: Análisis de Imágenes por IA")
st.write("Este proceso analizará todas las imágenes descargadas en la fase anterior (Pinterest e Insta).")

if st.button("3. Analizar todas las imágenes (CLIP)"):
    simular_script("analisis_imagenes.py", 4, "Modelo CLIP cargado. 1500 imágenes clasificadas por prenda, color y estilo.")

st.markdown("---")

# =========================================================================
# 3. REVISTAS DE MODA
# =========================================================================
st.header("Fase 3: Scrapping de Revistas")
st.write("Extracción de artículos y metadatos de revistas especializadas.")

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("4. Recolectar URLs Revistas"):
        simular_script("recolector_urls.py", 2, "URLs de artículos de moda extraídas con éxito.")

with c2:
    if st.button("5. Extractor Híbrido"):
        simular_script("extractor_hibrido.py", 3, "Texto, autor y temáticas extraídas del HTML.")

with c3:
    if st.button("6. Procesador de Fechas para normalizar formatos"):
        simular_script("procesador_fechas.py", 1, "Fechas relativas convertidas a formato Datetime (YYYY-MM-DD).")

st.markdown("---")

# =========================================================================
# 4. PREPARACIÓN Y MACHINE LEARNING
# =========================================================================
st.header("Fase 4: Consolidación y Entrenamiento Predictivo")
st.write("Fusión de todas las fuentes de datos (Redes + Revistas + Google Trends) y evaluación de los algoritmos.")

if st.button("7. Unificar Datasets Completos"):
    simular_script("unificar_datasets.py", 2, "Tablas unidas por 'termino' y 'fecha_dt'. Dimensiones: (2600, 45).")

if st.button("8. Generar Dataset de Entrenamiento"):
    simular_script("crear_dataset_entrenamiento.py", 2, "Variables rezagadas (lags) y variables Target generadas correctamente.")

st.markdown("### Competición de Algoritmos (Predictor de Tendencias)")
if st.button("9. ENTRENAR Y EVALUAR MODELOS (Random Forest, SVR, XGBoost)", type="primary"):
    with st.spinner("Entrenando modelos matemáticos..."):
        time.sleep(3) # Pausita dramática para que parezca que entrena
        st.balloons()
        st.success("¡Entrenamiento finalizado con éxito! Mostrando métricas y gráficas definitivas.")
        
        # 1. MOSTRAMOS LA TABLA REAL (Con los últimos datos que me pasaste)
        st.subheader("📊 Tabla de Métricas de Error")
        resultados = pd.DataFrame({
            "Métrica": ["MAE", "RMSE", "R²"],
            "Mod. A (Completo)": ["3.34", "5.80", "0.9505"],
            "Mod. B (Híbrido)": ["6.15", "8.98", "0.8816"],
            "Mod. C (RF Delta)": ["2.50", "4.32", "0.9725"],
            "Mod. D (SVR Delta)": ["2.50", "4.32", "0.9726"],
            "Mod. E (XGB Delta)": ["2.48", "4.21", "0.9740"]
        })
        # Ocultamos el índice numérico para que se vea como en la consola
        st.table(resultados.set_index("Métrica"))
        
        # 2. MOSTRAMOS LAS GRÁFICAS PRE-CALCULADAS
        st.subheader("📈 Predicciones del Comportamiento a Futuro")
        carpeta_graficas = os.path.join(BASE_DIR, "resultados_graficas")
        
        if os.path.exists(carpeta_graficas):
            imagenes = glob.glob(os.path.join(carpeta_graficas, "*.png"))
            if imagenes:
                cols = st.columns(2)
                for i, img_path in enumerate(imagenes):
                    cols[i % 2].image(img_path, use_container_width=True)
            else:
                st.info("Debes subir 3 o 4 gráficas PNG a la carpeta 'resultados_graficas' del servidor para que se muestren aquí.")
        else:
            st.info("Debes crear una carpeta llamada 'resultados_graficas' y mete algunas gráficas PNG para que la demo las pinte al terminar.")