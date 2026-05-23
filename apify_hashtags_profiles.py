import csv
import time
import os
import requests
import re
import random
from datetime import datetime, timedelta
from apify_client import ApifyClient

# ----------------- CONFIGURACIÓN -----------------
APIFY_API_TOKEN = "" # Token personal eliminado por motivos de seguridad 

# DIARIO: Busca lo último (poco consumo)
# HISTORICO: Busca tendencias pasadas (consume más)
# Para la aplicacion, se puede configurar con variable de entorno para no tener que tocar el código
MODO_EJECUCION = os.environ.get("MODO_EJECUCION", "DIARIO")

if MODO_EJECUCION == "DIARIO":
    OBJETIVOS = [
        # Perfiles
        "haileybieber",
        "lilyachty",
        "kendalljenner",
        "asaprocky",
        "lilbieber",
        "linda.sza",
        "bellahadid",
        "natwinter",
        "briidgetbrown"

        # Hashtags actuales
        "#streetstyle",
        "#wdywt",
        "#cphfw",
        "#pfw",
        "#nyfw",
        "#lfw",
        "#mfw",
    ]
    LIMIT_PER_HASHTAG = 30
    LIMIT_PER_PROFILE = 18

elif MODO_EJECUCION == "HISTORICO":
    OBJETIVOS = [
        # 2020
        "#streetstyle2020",
        "#fashiontrends2020",

        # Por estaciones
        "#springsummer2020",
        "#fallwinter2020",
       
        # Fashion weeks
        "#pfw20", # Paris
        "#mfw20", # Milan
        "#nyfw20", # New York
        "#cphfw20", # Copenhagen

        # 2021
        "#streetstyle2021",
        "#fashiontrends2021",

        # Por estaciones
        "#springsummer2021",
        "#fallwinter2021",
       
        # Fashion weeks
        "#pfw21", # Paris
        "#mfw21", # Milan
        "#nyfw21", # New York
        "#cphfw21", # Copenhagen

        # 2022
        "#streetstyle2022",
        "#fashiontrends2022",

        # Por estaciones
        "#springsummer2022",
        "#fallwinter2022",
       
        # Fashion weeks
        "#pfw22", # Paris
        "#mfw22", # Milan
        "#nyfw22", # New York
        "#cphfw22", # Copenhagen

        # 2023
        "#streetstyle2023",
        "#fashiontrends2023",
       
        # Por estaciones
        "#springsummer2023",
        "#fallwinter2023",
       
        # Fashion weeks
        "#pfw23", # Paris
        "#mfw23", # Milan
        "#nyfw23", # New York
        "#cphfw23", # Copenhagen

        # 2024
        "#streetstyle2024",
        "#fashiontrends2024",
        
        # Por estaciones
        "#springsummer2024",
        "#fallwinter2024",

        # Fashion weeks
        "#pfw24", # Paris
        "#mfw24", # Milan
        "#nyfw24", # New York
        "#cphfw24", # Copenhagen

        # 2025
        "#streetstyle2025",
        "#fashiontrends2025",
        
        # Por estaciones
        "#springsummer2025",
        "#fallwinter2025",

        # Fashion weeks
        "#pfw25", # Paris
        "#mfw25", # Milan
        "#nyfw25", # New York
        "#cphfw25", # Copenhagen

    ]
    LIMIT_PER_HASHTAG = 35
    LIMIT_PER_PROFILE = 0

base_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(base_dir, "dataset_instagram", "instagram_posts.csv")
IMAGES_DIR = os.path.join(base_dir, "imagenes_descargadas")

# Aseguramos que las carpetas existen
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------- FUNCIONES AUXILIARES DE FECHA -----------------
def obtener_anio_de_target(target):
    """Extrae el año del hashtag o perfil objetivo."""
    target = str(target).lower()
    match_long = re.search(r'(202[0-6])', target)
    if match_long: return int(match_long.group(1))
    match_short = re.search(r'([2][0-6])$', target)
    if match_short: return 2000 + int(match_short.group(1))
    return None

def generar_fecha_inteligente(anio_target, texto_target):
    """Genera una fecha coherente con la estación."""
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

def formatear_fecha(fecha_str):
    if not fecha_str: return ""
    try:
        # Usamos fromisoformat para manejar bien las zonas horarias
        fecha = datetime.fromisoformat(fecha_str.replace("Z", "+00:00"))
        return fecha.strftime("%Y-%m-%d")
    except:
        return ""

def descargar_imagen_local(url_remota, nombre_archivo):
    """Descarga solo si no existe ya el archivo."""
    ruta_completa = os.path.join(IMAGES_DIR, nombre_archivo)
    
    # 1. BLOQUEO DE DUPLICADOS DE IMAGEN
    if os.path.exists(ruta_completa):
        # Ya la tenemos, devolvemos la ruta pero NO descargamos de nuevo
        return ruta_completa

    try:
        response = requests.get(url_remota, timeout=15)
        if response.status_code == 200:
            with open(ruta_completa, 'wb') as f:
                f.write(response.content)
            return ruta_completa
    except Exception as e:
        print(f"   [Error Img] {e}")
    
    return None

def cargar_urls_existentes(archivo):
    urls = set()
    if os.path.exists(archivo):
        try:
            with open(archivo, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Guardamos la URL del post para no repetir filas en el CSV
                    if "URL_Post" in row and row["URL_Post"]:
                        urls.add(row["URL_Post"])
        except: pass
    return urls

def guardar_csv_incremental(datos, archivo):
    if not datos: return
    archivo_existe = os.path.exists(archivo)
    modo = "a" if archivo_existe else "w"
    
    campos = [
        "Target_Busqueda", "Fuente", "Usuario_Post", "Texto", 
        "Likes", "Fecha", "URL_Post", "Imagen_URL_Original", "Ruta_Local_Imagen"
    ]
    
    with open(archivo, mode=modo, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        if not archivo_existe:
            writer.writeheader()
        writer.writerows(datos)

def scrapear_apify(client, objetivo, urls_ya_guardadas):
    print(f"\nProcesando: {objetivo}")
    resultados = []
    dataset_id = None
    
    # CONFIGURACIÓN DE SEGURIDAD
    # Si en 180 segundos (3 min) no ha terminado, cortamos
    TIEMPO_MAXIMO_POR_PERFIL = 200
    
    try:
        # Selección de actor según el objetivo (perfil o hashtag)
        if objetivo.startswith("#"):
            tag_name = objetivo.replace("#", "")
            actor_id = "apify/instagram-hashtag-scraper"
            run_input = {
                "hashtags": [tag_name], 
                "resultsLimit": LIMIT_PER_HASHTAG,
                # Evitamos bucles infinitos en hashtags porque a veces son muy grandes y pueden tardar mucho
                "maxRequestRetries": 2, 
            }
        else:
            username = objetivo.replace("@", "")
            actor_id = "apify/instagram-scraper"
            run_input = {
                "directUrls": [f"https://www.instagram.com/{username}/"],
                "resultsLimit": LIMIT_PER_PROFILE,
                "resultsType": "posts",
                "proxy": {"useApifyProxy": True},
                # -----------------LIMITES PARA NO QUEDARNOS MUCHO TIEMPO BLOQUEADOS  -----------------
                "maxRequestRetries": 3, # Si falla 2 veces, cortamos
                "pageTimeout": 60, # Si una página tarda 60s, fuera
                "handlePageTimeoutSecs": 2000, # Tiempo max total de navegación
            }

        # Llamada a Apify con timeout_secs asegura que no se quede pillado para siempre
        run = client.actor(actor_id).call(
            run_input=run_input, 
            timeout_secs=TIEMPO_MAXIMO_POR_PERFIL
        )
        
        if run:
            dataset_id = run["defaultDatasetId"]
            print(f"   Estado final Apify: {run.get('status')}")
        
    except Exception as e:
        print(f"   [ALERTA] Se excedió el tiempo o hubo error con {objetivo}. Saltando...")
        # No imprimimos todo el error si es muy largo, solo avisamos
        return []

    # Procesado de resultados (si tenemos dataset)
    if dataset_id:
        try:
            dataset = client.dataset(dataset_id)
            items = dataset.list_items().items
            
            if not items:
                print("   (0 items encontrados. Posible bloqueo o perfil privado)")
                return []

            print(f"   Items encontrados: {len(items)}")
            
            nuevos_guardados = 0
            
            for item in items:
                # Filtros básicos para evitar videos
                if item.get("type") == "Video" or item.get("isVideo", False): continue

                url_post = item.get("url") or f"https://www.instagram.com/p/{item.get('shortcode')}"
                
                # 2. BLOQUEO DE DUPLICADOS DE CSV
                if url_post in urls_ya_guardadas:
                    continue

                # Datos básicos
                shortcode = item.get("shortcode") or item.get("id")
                user_clean = (item.get("ownerUsername") or "unknown").replace(".", "_")
                fecha_clean = formatear_fecha(item.get("timestamp") or item.get("taken_at_timestamp"))
                caption = item.get("caption") or ""
                texto_post = caption.replace("\n", " ").replace("\r", " ") \
                                    .replace("\u2028", " ").replace("\u2029", " ") \
                                    .replace("\u200E", "") \
                                    .replace(";", ",")[:500]
                likes = item.get("likesCount") or item.get("likes", 0)

                # ----------------- LÓGICA DE CORRECCIÓN DE FECHA -----------------
                anio_objetivo = obtener_anio_de_target(objetivo)
                if anio_objetivo:
                    needs_correction = False
                    if not fecha_clean:
                        needs_correction = True
                    else:
                        try:
                            fecha_dt = datetime.strptime(fecha_clean, "%Y-%m-%d")
                            if fecha_dt.year != anio_objetivo:
                                needs_correction = True
                        except ValueError:
                            needs_correction = True # Fecha inválida

                    if needs_correction:
                        fecha_clean = generar_fecha_inteligente(anio_objetivo, objetivo)

                # Detección de imágenes
                lista_imagenes = []

                # Caso A: Carrusel/multifoto
                if item.get("type") == "Sidecar" or item.get("childPosts") or item.get("images"):
                    children = item.get("childPosts") or item.get("images") or []
                    for idx, child in enumerate(children):
                        if child.get("type") == "Video": continue
                        img_url = child.get("displayUrl") or child.get("url")
                        if img_url:
                            lista_imagenes.append((img_url, f"_slide{idx}"))
                
                # Caso B: Foto única
                if not lista_imagenes:
                    img_url = item.get("displayUrl") or item.get("imageUrl")
                    if img_url:
                        lista_imagenes.append((img_url, ""))

                # Descarga y guardado
                for (url_img_web, sufijo) in lista_imagenes:
                    nombre_archivo = f"{user_clean}_{shortcode}{sufijo}.jpg"
                    
                    ruta_local = descargar_imagen_local(url_img_web, nombre_archivo)
                    
                    if ruta_local:
                        resultados.append({
                            "Target_Busqueda": objetivo,
                            "Fuente": "Perfil" if not objetivo.startswith("#") else "Hashtag",
                            "Usuario_Post": user_clean,
                            "Texto": texto_post,
                            "Likes": likes,
                            "Fecha": fecha_clean,
                            "URL_Post": url_post,
                            "Imagen_URL_Original": url_img_web,
                            "Ruta_Local_Imagen": ruta_local
                        })
                        nuevos_guardados += 1
                
                urls_ya_guardadas.add(url_post)

            print(f"   Nuevas imágenes guardadas: {nuevos_guardados}")
            
        except Exception as e:
            print(f"[Error Procesando Dataset] {e}")

    return resultados

def main():
    if not APIFY_API_TOKEN:
        print("[ERROR]: Falta el token de Apify")
        return

    client = ApifyClient(APIFY_API_TOKEN)
    urls_en_memoria = cargar_urls_existentes(OUTPUT_CSV)
    print(f"Base de datos actual: {len(urls_en_memoria)} posts procesados.")

    for objetivo in OBJETIVOS:
        nuevos = scrapear_apify(client, objetivo, urls_en_memoria)
        if nuevos:
            guardar_csv_incremental(nuevos, OUTPUT_CSV)
        time.sleep(3) 

if __name__ == "__main__":
    main()