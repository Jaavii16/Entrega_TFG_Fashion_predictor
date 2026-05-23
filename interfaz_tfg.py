import streamlit as st
import subprocess
import os
import sys
import datetime
import glob # Para buscar las imagenes guardadas

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Prototipo TFG Javier", layout="wide", initial_sidebar_state="expanded")
st.title("Predictor de Tendencias de Moda Basado en Redes Sociales - TFG de Javier")
st.markdown("Bienvenido al panel de control del TFG. Siga el orden de ejecución para procesar los datos y entrenar la Inteligencia Artificial.")
st.markdown("---")

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.sidebar.header("Configuración Global")
modo_ejecucion = st.sidebar.radio(
    "Modo de recolección:", 
    ("HISTORICO", "DIARIO"),
    help="Histórico recopila datos desde 2020. Diario solo recopila lo actual."
)

ventana_prediccion = st.sidebar.selectbox(
    "Ventana de Predicción (Target):",
    ("TARGET_NEXT_3M_AVG", "TARGET_NEXT_6M_AVG"),
    help="Selecciona qué variable quieres que el modelo prediga: la popularidad promedio de los próximos 3 meses o de los próximos 6 meses."
)

# Selector de fecha de corte en el panel lateral
fecha_corte = st.sidebar.date_input(
    "Fecha de corte (Train/Test Split):",
    datetime.date(2025, 1, 1),
    help="Los datos estrictamente anteriores a esta fecha se usarán para entrenar. Los posteriores se aíslan para validar el modelo."
)


def ejecutar_script(nombre_script):
    ruta_script = os.path.join(BASE_DIR, nombre_script)
    if not os.path.exists(ruta_script):
        st.error(f"Error: No se encuentra {nombre_script}")
        return False, ""
    
    # Preparamos el entorno virtual pasándole nuestras variables de la interfaz
    entorno_actual = os.environ.copy()
    entorno_actual["MODO_EJECUCION"] = modo_ejecucion
    entorno_actual["TARGET_COL"] = ventana_prediccion
    entorno_actual["FECHA_CORTE"] = fecha_corte.strftime("%Y-%m-%d")
    
    with st.spinner(f"Ejecutando {nombre_script}..."):
        try:
            proceso = subprocess.run(
                [sys.executable, ruta_script], 
                capture_output=True, text=True, check=True, env=entorno_actual
            )
            st.success(f"Ejecución completada.")
            with st.expander("Ver consola completa"):
                st.code(proceso.stdout)
            return True, proceso.stdout # Devolvemos el texto impreso
        except subprocess.CalledProcessError as e:
            st.error("Error en la ejecución.")
            with st.expander("Ver error"):
                st.code(e.stderr)
            return False, ""

# =========================================================================
# 1. PINTEREST E INSTAGRAM (Recolección e Imágenes)
# =========================================================================
st.header("Fase 1: Redes Sociales (Pinterest e Instagram)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pinterest")
    if st.button("1a. Ejecutar Scraper de Pinterest"):
        ejecutar_script("pinterest_scraper_2.py")
            
    if st.button("2. Descargar Imágenes (Pinterest)"):
        ejecutar_script("pinterest_image_downloader.py")

with col2:
    st.subheader("Instagram")
    if st.button("1b. Ejecutar Scraper de Instagram"):
        ejecutar_script("apify_hashtags_profiles.py")

st.markdown("---")

# =========================================================================
# 2. ANÁLISIS DE IMÁGENES
# =========================================================================
st.header("Fase 2: Análisis de Imágenes por IA")
st.write("Este proceso analizará todas las imágenes descargadas en la fase anterior (Pinterest e Insta).")

if st.button("3. Analizar todas las imágenes (CLIP)"):
    ejecutar_script("analisis_imagenes.py")

st.markdown("---")

# =========================================================================
# 3. REVISTAS DE MODA
# =========================================================================
st.header("Fase 3: Scrapping de Revistas")
st.write("Extracción de artículos y metadatos de revistas especializadas.")

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("4. Recolectar URLs Revistas"):
        ejecutar_script("recolector_urls.py")

with c2:
    if st.button("5. Extractor Híbrido"):
        ejecutar_script("extractor_hibrido.py")

with c3:
    if st.button("6. Procesador de Fechas para normalizar formatos"):
        ejecutar_script("procesador_fechas.py")

st.markdown("---")

# =========================================================================
# 4. PREPARACIÓN Y MACHINE LEARNING
# =========================================================================
st.header("Fase 4: Consolidación y Entrenamiento Predictivo")
st.write("Fusión de todas las fuentes de datos (Redes + Revistas + Google Trends) y evaluación de los algoritmos.")

if st.button("7. Unificar Datasets Completos"):
    ejecutar_script("unificar_datasets.py")

if st.button("8. Generar Dataset de Entrenamiento"):
    ejecutar_script("crear_dataset_entrenamiento.py")

st.markdown("### Competición de Algoritmos (Predictor de Tendencias)")
if st.button("9. ENTRENAR Y EVALUAR MODELOS (Random Forest, SVR, XGBoost)", type="primary"):
    
    # Limpiamos gráficas antiguas antes de ejecutar
    carpeta_graficas = os.path.join(BASE_DIR, "resultados_graficas")
    if os.path.exists(carpeta_graficas):
        for file in glob.glob(os.path.join(carpeta_graficas, "*.png")):
            os.remove(file)
            
    exito, consola = ejecutar_script("entrenamiento_3modelos.py")
    
    if exito:
        
        # 1. Mostrar la tabla de resultados extraída de la consola
        st.subheader("Tabla de Métricas de Error")
        # Buscamos la línea donde empieza la tabla y la mostramos
        if "--- RESULTADOS DEL ESTUDIO COMPARATIVO ---" in consola:
            tabla_texto = consola.split("--- RESULTADOS DEL ESTUDIO COMPARATIVO ---")[1].split("TOP 5 VARIABLES")[0]
            st.code(tabla_texto.strip(), language="plaintext")
        else:
            st.warning("No se pudo extraer la tabla, revisa la consola completa arriba.")

        # 2. Mostrar las gráficas generadas
        st.subheader("Predicciones del Comportamiento a Futuro")
        if os.path.exists(carpeta_graficas):
            imagenes = glob.glob(os.path.join(carpeta_graficas, "*.png"))
            if imagenes:
                cols = st.columns(2)
                for i, img_path in enumerate(imagenes):
                    cols[i % 2].image(img_path, use_container_width=True)
            else:
                st.info("No se generaron gráficas para mostrar.")