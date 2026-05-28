# Predictor de Tendencias de Moda Basado en Redes Sociales

Este repositorio contiene el código fuente y los pipelines de datos desarrollados para el Trabajo de Fin de Grado (TFG). El proyecto consiste en un sistema híbrido capaz de anticipar el éxito comercial de prendas de ropa y micro-tendencias mediante la recolección y análisis de datos no estructurados.

El sistema integra técnicas de web scraping, visión artificial (Zero-Shot Learning con CLIP), procesamiento de lenguaje natural (NLP) y algoritmos de Machine Learning (XGBoost, Random Forest, SVR) para analizar el "eco social" en redes y predecir el interés de búsqueda futuro en Google Trends.

## Demo Interactiva (Recomendado)
Para visualizar el flujo de trabajo del proyecto y explorar las métricas resultantes sin necesidad de configuración local ni ejecución de algoritmos pesados, se ha desplegado una versión demostrativa en la nube:

https://entregatfgfashionpredictor-lhrsyqd5unx2uncfn6zrke.streamlit.app/

---

## Arquitectura del Pipeline
El sistema se divide en cuatro fases modulares:
1. **Extracción Social:** Recolección de imágenes y metadatos desde Pinterest e Instagram.
2. **Inferencia Visual:** Clasificación semántica masiva de prendas mediante el modelo CLIP de OpenAI.
3. **Procesamiento de Revistas:** Extracción y normalización de artículos de portales de moda.
4. **Ingeniería de Datos y Modelado:** Consolidación de fuentes, cálculo de métricas de crecimiento neto temporal (Delta) y entrenamiento de algoritmos predictivos.

---

## Ejecución Local (Entorno de Desarrollo)

Si desea ejecutar el pipeline completo en su máquina local, siga estos pasos:

### 1. Requisitos Previos
* Python 3.9 o superior.
* Clonar este repositorio.

### 2. Instalación de Dependencias
Abra una terminal en la raíz del proyecto y ejecute:
```bash
pip install -r requirements.txt
```

### 3. Configuración de Variables de Entorno (IMPORTANTE)
Por motivos de seguridad, los tokens de acceso a APIs de terceros no se incluyen en este repositorio. Para ejecutar la Fase 1 (Instagram), es necesario crear un archivo .env en la raíz del proyecto con sus credenciales privadas:

### 4. Lanzar la Interfaz de Usuario
El proyecto cuenta con un panel de control interactivo desarrollado en Streamlit que orquesta la ejecución secuencial de todos los scripts. Para iniciarlo, asegúrese de estar en la raíz del proyecto y ejecute en su terminal:
```bash
streamlit run interfaz_tfg.py
```

Una vez ejecutado, se abrirá automáticamente una ventana en su navegador web (por defecto en http://localhost:8501) desde donde podrá operar el sistema completo.
