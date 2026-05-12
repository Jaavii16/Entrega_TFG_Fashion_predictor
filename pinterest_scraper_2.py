import json
import time
import pandas as pd
import os
import random
import re
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager



# ----------------- CONFIGURACIÓN AVANZADA DE BÚSQUEDA -----------------

#MODO_EJECUCION = "HISTORICO" 
# Para la aplicacion
MODO_EJECUCION = os.environ.get("MODO_EJECUCION", "DIARIO")

# Lista de Influencers que suelen generar tendencias
# En Pinterest funcionan mejor con nombre y apellido + "street style" u "outfits"
INFLUENCERS = [
    "hailey bieber outfits",
    "lil yachty outfits",
    "kendall jenner outfits",
    "asap rocky outfits",
    "justin bieber outfits",
    "bella hadid outfits",
]

def generar_queries_historicas():
    """Genera búsquedas neutras, globales y consistentes."""
    lista = []
    years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    
    # Ciudades de las fashion weeks más importantes 
    fw = ["paris", "new york", "milan", "london", "copenhagen"]
    
    for year in years:
        # 1. GENERALES
        lista.append(f"street style {year} trends")
        lista.append(f"fashion trends {year}") 
        
        # 2. ESTACIONES
        lista.append(f"spring summer {year} fashion")
        lista.append(f"fall winter {year} street style")
        
        # 3. FASHION WEEKS 
        for city in fw:
            lista.append(f"{city} fashion week {year}")

    return lista

# ----------------- GENERACIÓN DE LA LISTA DE QUERIES -----------------

if MODO_EJECUCION == "DIARIO":
    # 1. Lo general de este año
    queries = [
        "latest street style trends",
        "viral fashion trends",
        "trending outfits",
        "popular fashion",
        "top fashion trends",
    ]
    # 2. Añadimos los influencers
    queries += [f"{inf}" for inf in INFLUENCERS]

elif MODO_EJECUCION == "HISTORICO":
    queries = generar_queries_historicas()


# ----------------- CONFIGURACIÓN TÉCNICA -----------------
NUM_PINS_A_ANALIZAR = 15
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_pinterest")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "pinterest_dataset_completo.csv")

print(f"--- MODO {MODO_EJECUCION} ACTIVADO ---")
print(f"Se procesarán {len(queries)} términos de búsqueda distintos.")


# ------------------ FUNCIONES AUXILIARES DE FECHA ------------------
def obtener_anio_de_target(target):
    """Extrae el año del texto de búsqueda."""
    target = str(target).lower()
    match = re.search(r'(202[0-6])', target)
    if match: return int(match.group(1))
    return None

def generar_fecha_inteligente(anio_target, texto_target):
    # ... (Exactamente la misma función que en el script de Instagram) ...
    # Copiar y pegar la implementación completa aquí para que funcione.
    texto_target = str(texto_target).lower()
    mes_inicio, mes_fin = 1, 12
    if any(x in texto_target for x in ['ss', 'spring', 'summer']):
        mes_inicio, mes_fin = 4, 8
    elif any(x in texto_target for x in ['fw', 'fall', 'winter']):
        mes_inicio, mes_fin = 9, 12
    elif any(x in texto_target for x in ['fw', 'fashionweek', 'pfw', 'nyfw', 'mfw', 'cphfw']):
        mes_inicio = random.choice([2, 9])
        mes_fin = mes_inicio
        
    try:
        start_date = datetime(anio_target, mes_inicio, 1)
        if mes_fin == 12: end_date = datetime(anio_target, 12, 31)
        elif mes_fin == 2: end_date = datetime(anio_target, 2, 28)
        else: end_date = datetime(anio_target, mes_fin, 30)
        dias_random = random.randint(0, max(0, (end_date - start_date).days))
        return (start_date + timedelta(days=dias_random)).strftime("%Y-%m-%d")
    except: return f"{anio_target}-06-15"

# Para cada query, obtener URLs de pines
def obtener_urls_de_feed(driver, query, num_max):
    print(f"--- Buscando: {query} ---")
    query_url = query.replace(" ", "+")
    url = f"https://www.pinterest.com/search/pins/?q={query_url}"
    driver.get(url)
    time.sleep(5)

    pin_urls = set()
    scrolls = 0
    max_scrolls = 20
    
    while len(pin_urls) < num_max and scrolls < max_scrolls:
        # Buscamos elementos 'a' que lleven a un pin (/pin/...)
        elementos = driver.find_elements(By.CSS_SELECTOR, "a[href*='/pin/']")
        for a in elementos:
            try:
                href = a.get_attribute("href")
                # Filtramos para asegurarnos que es un pin real y no basura
                if href and "/pin/" in href and str(href).count("/") >= 4:
                    pin_urls.add(href)
                    if len(pin_urls) >= num_max:
                        break
            except StaleElementReferenceException:
                continue
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        scrolls += 1
        print(f"   URLs recolectadas: {len(pin_urls)}")

    return list(pin_urls)

def extraer_detalles_pin(driver, url):
    """Entra al pin y busca el JSON oculto con la fecha"""
    try:
        driver.get(url)
        time.sleep(random.uniform(2, 4)) # Pausa humana
        
        # 1. Buscamos script JSON-LD (Donde Pinterest guarda los datos para Google)
        scripts = driver.find_elements(By.CSS_SELECTOR, 'script[type="application/ld+json"]')
        
        fecha = None
        likes = 0
        texto = ""
        imagen = ""
        
        for script in scripts:
            try:
                data = json.loads(script.get_attribute('innerHTML'))
                
                # A veces es una lista, a veces un dict
                if isinstance(data, list): data = data[0]
                
                # Buscamos la fecha
                if 'datePublished' in data:
                    fecha = data['datePublished'] # Formato suele ser YYYY-MM-DD
                
                # Buscamos interacciones (UserInteraction)
                if 'interactionStatistic' in data:
                    interactions = data['interactionStatistic']
                    if isinstance(interactions, list):
                        for i in interactions:
                            if i.get('interactionType') == 'http://schema.org/InteractAction': # O LikeAction
                                likes = i.get('userInteractionCount', 0)
                    elif isinstance(interactions, dict):
                         likes = interactions.get('userInteractionCount', 0)
                
                # Texto y Título
                texto = data.get('description') or data.get('name') or ""
                
                # Imagen
                if 'image' in data:
                    imagen = data['image']
                
                if fecha: break # Si encontramos fecha, salimos
            except:
                pass
        
        # Si falla el JSON, intentamos sacar texto del HTML visible
        if not texto:
            try:
                h1 = driver.find_element(By.TAG_NAME, 'h1')
                texto = h1.text
            except: pass

        return {
            "URL_Pin": url,
            "Fecha": fecha,
            "Likes": likes,
            "Texto": texto,
            "Imagen_URL": imagen
        }

    except Exception as e:
        print(f"Error en {url}: {e}")
        return None

# ----------------- EJECUCIÓN -----------------
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--log-level=3")
# User agent para que Pinterest no bloquee tanto
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

datos_totales = []

try:
    for i, q in enumerate(queries): # 'i' es el índice de la query
        urls = obtener_urls_de_feed(driver, q, NUM_PINS_A_ANALIZAR)
        
        print(f"   Analizando {len(urls)} pines al detalle...")
        for j, url_pin in enumerate(urls): # 'j' es el índice del pin dentro de esa query
            datos = extraer_detalles_pin(driver, url_pin)
            if datos:
                # --- LÓGICA DE FECHA ---
                fecha_original = datos.get('Fecha')
                anio_objetivo = obtener_anio_de_target(q)

                if anio_objetivo:
                    needs_correction = False
                    if not fecha_original: needs_correction = True
                    else:
                        try:
                            f_str = str(fecha_original)[:10]
                            fecha_dt = datetime.strptime(f_str, "%Y-%m-%d")
                            if fecha_dt.year != anio_objetivo: needs_correction = True
                        except: needs_correction = True

                    if needs_correction:
                        datos['Fecha'] = generar_fecha_inteligente(anio_objetivo, q)
                else:
                    # Lógica para añadir fecha actual si no se encuentra en modo diario
                    if not fecha_original:
                        # Si no encontró fecha en el HTML, le ponemos la de hoy
                        datos['Fecha'] = datetime.now().strftime("%Y-%m-%d")
                    else:
                        # Si la encontró, aseguramos que se quede solo con YYYY-MM-DD y no con horas o texto raro
                        datos['Fecha'] = str(fecha_original)[:10]
                
                # Limpieza de texto básica
                if datos["Texto"]:
                    datos["Texto"] = str(datos["Texto"]).replace("\n", " ").replace(";", ",")
                
                datos["Fuente"] = "Pinterest"
                datos["Query"] = q
                datos_totales.append(datos)
                print(f"    -> [{j+1}/{len(urls)}] Fecha: {datos['Fecha']} ")

except KeyboardInterrupt:
    print("\nDetenido por usuario. Guardando lo que hay...")

finally:
    driver.quit()
    
    # Guardado
    if datos_totales:
        df = pd.DataFrame(datos_totales)
        # Aseguramos que la columna fecha tenga formato limpio
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # GUARDADO INCREMENTAL CORRECTO
        if os.path.exists(CSV_PATH):
            df.to_csv(CSV_PATH, mode='a', header=False, index=False, encoding="utf-8-sig")
            print(f"\nAñadidos {len(df)} registros a: {CSV_PATH}")
        else:
            df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
            print(f"\nCreado archivo nuevo: {CSV_PATH}")
    else:
        print("No se extrajeron datos.")