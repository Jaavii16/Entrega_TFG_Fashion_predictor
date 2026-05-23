import csv
import os
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse, urljoin, urldefrag
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# ----------------- CONFIGURACIÓN -----------------
MAX_URLS_POR_REVISTA_HISTORICO = 55
MAX_URLS_POR_REVISTA_ACTUAL = 35

# Palabras que indican que la URL NO es un artículo de texto válido
# Si la URL contiene esto, la ignoramos para evitar "VACÍOS" en el extractor.
BAD_KEYWORDS = [
    "/gallery/", "/galeria/", "/video/", "/videos/", "/tv/", 
    "/tag/", "/tags/", "/tema/", "/topic/", "/category/", "/categoria/",
    "/search/", "/buscar/", "/login/", "/account/", "/subscribe/",
    "/shop/", "/tienda/", "/compras/", "/newsletter/", 
    "contact", "aviso-legal", "privacidad", "cookies",
    "/author/", "/autor/", "/encuesta/", "/quiz/"
]

REVISTAS = {
    "Vogue":    {"url": "https://www.vogue.es/moda", "domain": "vogue.es", "cdx_match": "vogue.es/moda/*"},
    "Elle":     {"url": "https://www.elle.com/es/moda/", "domain": "elle.com", "cdx_match": "elle.com/es/moda/*"},
    "Harpers":  {"url": "https://www.harpersbazaar.com/es/moda/", "domain": "harpersbazaar.com", "cdx_match": "harpersbazaar.com/es/moda/*"},
    "GQ":       {"url": "https://www.gq.com.mx/moda", "domain": "gq.com", "cdx_match": "gq.com.mx/moda/*"}, 
    "InStyle":  {"url": "https://www.instyle.es/moda", "domain": "instyle.es", "cdx_match": "instyle.es/moda/*"},
    "i-D":       {"url": "https://i-d.co/topic/fashion", "domain": "i-d.co", "cdx_match": "i-d.co/article/*"}, 
    "Dazed":     {"url": "https://www.dazeddigital.com/fashion", "domain": "dazeddigital.com", "cdx_match": "dazeddigital.com/fashion/*"},
    "Interview": {"url": "https://www.interviewmagazine.com/category/fashion", "domain": "interviewmagazine.com", "cdx_match": "interviewmagazine.com/fashion/*"},
    "Esquire":  {"url": "https://www.esquire.com/es/moda-hombre/", "domain": "esquire.com", "cdx_match": "esquire.com/es/moda-hombre/*"}
}

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "dataset_revistas")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "revistas_dataset_urls.csv")

# ----------------- FUNCIONES -----------------

def es_url_valida(url):
    """Filtra URLs que sabemos que dan problemas o son índices"""
    url_lower = url.lower()
    
    # 1. Filtro de palabras prohibidas (Galerías, vídeos, login...)
    if any(bad in url_lower for bad in BAD_KEYWORDS):
        return False
    
    # 2. Filtro de extensiones
    if url_lower.endswith(('.pdf', '.jpg', '.png', '.jpeg', '.gif')):
        return False

    # 3. FILTRO ANTI-ÍNDICES (Categorías principales)
    # Si la URL termina exactamente en una sección común, es un índice, no un artículo.
    secciones_raiz = [
        "/moda/", "/moda", "/fashion/", "/fashion", 
        "/style/", "/style", "/news/", "/news",
        "/trends/", "/trends", "/tendencias/", "/tendencias",
        "/modapedia", "/modapedia/", "/streetstyle", "/streetstyle/", 
        "/ninos", "/ninos/", "/pasarelas", "/pasarelas/", "/belleza", "/belleza/"
    ]
    
    # Comprobamos si la URL termina en alguna de estas secciones
    if any(url_lower.endswith(sec) for sec in secciones_raiz):
        return False
        
    return True

def limpiar_url(url):
    """Elimina parámetros de rastreo (utm_source, etc) y fragmentos (#)"""
    # Quitamos el fragmento (#ancla)
    url, _ = urldefrag(url)
    # Quitamos query params para evitar duplicados (?utm_source=...)
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

def crear_sesion_robusta():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def obtener_historico_api(nombre, config):
    print(f"   [HISTÓRICO] Consultando Wayback Machine para {nombre}...")
    urls_encontradas = set()
    
    # Filtramos por status 200 y mimetype html para evitar redirecciones rotas
    api_url = f"http://web.archive.org/cdx/search/cdx?url={config['cdx_match']}&output=json&collapse=urlkey&filter=statuscode:200&filter=mimetype:text/html&limit={MAX_URLS_POR_REVISTA_HISTORICO * 3}" 
    # Pedimos el triple de límite porque luego vamos a descartar muchas en el filtrado
    
    session = crear_sesion_robusta()
    
    try:
        r = session.get(api_url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if len(data) > 1:
                for row in data[1:]:
                    timestamp, original_url = row[1], row[2]
                    
                    if not es_url_valida(original_url):
                        continue
                    
                    # Construimos la URL de Wayback
                    wayback_url = f"https://web.archive.org/web/{timestamp}/{original_url}"
                    urls_encontradas.add(wayback_url)
                    
                    if len(urls_encontradas) >= MAX_URLS_POR_REVISTA_HISTORICO:
                        break
            print(f"   -> {len(urls_encontradas)} URLs históricas válidas.")
        else:
            print(f"   -> API Error: {r.status_code}")
    except Exception as e:
        print(f"   -> Error conexión API: {e}")
    
    return list(urls_encontradas)

def obtener_actual_selenium(driver, nombre, config):
    print(f"   [ACTUAL] Scrapeando portada de {nombre}...")
    lista_urls = []
    
    try:
        driver.get(config["url"])
        time.sleep(3) # Espera inicial
        
        # Scroll progresivo para activar Lazy Load
        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        articles = soup.find_all("a", href=True)
        
        keywords_positivas = ["/article/", "/story/", "/moda/", "/fashion/", "/noticias/", "/news/", "/tendencias/", "/trends/"]
        
        for a in articles:
            href = a['href']
            url_final = urljoin(config["url"], href)
            url_final = limpiar_url(url_final)
            
            # 1. Validar dominio
            if config['domain'] not in url_final:
                continue
            
            # 2. Validamos con los filtros
            if not es_url_valida(url_final):
                continue

            # 3. Miramos si es o parece un artículo. 
            # Debe tener cierta longitud o contener keywords de artículo
            es_articulo = False
            
            # Si contiene keywords típicas de artículo
            if any(k in url_final for k in keywords_positivas):
                es_articulo = True
            
            # O si el enlace tiene un texto largo (título del artículo)
            texto_link = a.get_text(strip=True)
            if len(texto_link) > 25 and len(url_final) > 40:
                es_articulo = True
            
            if es_articulo:
                lista_urls.append(url_final)
                
            if len(lista_urls) >= MAX_URLS_POR_REVISTA_ACTUAL * 2: # Recogemos de más para quitar duplicados luego
                break
        
        # Eliminar duplicados
        lista_urls = list(dict.fromkeys(lista_urls))
        # Recortar al límite deseado
        lista_urls = lista_urls[:MAX_URLS_POR_REVISTA_ACTUAL]
        
        print(f"   -> {len(lista_urls)} URLs actuales válidas.")
        
    except Exception as e:
        print(f"   -> Error Selenium: {e}")
    
    return lista_urls

# ----------------- EJECUCIÓN -----------------

# Modo 'w' para sobrescribir y empezar limpio, ya que vamos a generar nuevas URLs y las otras ya las tenemos en el siguiente paso (extractor_hibrido)
f = open(output_file, 'w', newline="", encoding="utf-8")
writer = csv.DictWriter(f, fieldnames=["Fuente", "URL", "Origen"])
writer.writeheader()

print(f"--- INICIANDO RECOLECCIÓN OPTIMIZADA ---")

options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--log-level=3")
# User agent real para evitar bloqueos simples
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

try:
    total_guardado = 0
    for nombre, config in REVISTAS.items():
        print(f"\n--- {nombre} ---")
        
        # 1. HISTÓRICO
        urls_old = obtener_historico_api(nombre, config)
        for u in urls_old:
            writer.writerow({"Fuente": nombre, "URL": u, "Origen": "Wayback"})
            total_guardado += 1
            
        # 2. ACTUAL
        urls_new = obtener_actual_selenium(driver, nombre, config)
        for u in urls_new:
            writer.writerow({"Fuente": nombre, "URL": u, "Origen": "Actual"})
            total_guardado += 1
            
        f.flush() # Guardar en disco tras cada revista

except KeyboardInterrupt:
    print("\n[!] DETENCIÓN MANUAL")

finally:
    f.close()
    driver.quit()
    print(f"\n[FIN] Recolección completada. URLs guardadas: {total_guardado}")