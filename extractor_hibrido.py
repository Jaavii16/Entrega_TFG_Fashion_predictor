import csv
import os
import time
import random
import requests
import sys
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import trafilatura 

# ----------------- CONFIGURACIÓN -----------------
base_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_dir, "dataset_revistas", "revistas_dataset_urls.csv")
output_file = os.path.join(base_dir, "dataset_revistas", "dataset_final_contenido.csv")

# Selectores manuales por revista
SELECTORES = {
    "Vogue":    ["div.body__inner-container", "div[data-testid='BodyWrapper']", "div.article__body", "div.c-article-body"],
    "Elle":     ["div.article-body", "div.standard-body", "section.body-text"],
    "Harpers":  ["div.article-body", "div.standard-body", "div.body-content"],
    "GQ":       ["div.article-body", "div.body__inner-container", "div.content-main"],
    "InStyle":  ["div.article-body__content", "div.m-component-content"],
    "i-D":       ["div.article__body", "div.sc-article-body", "div.content-wrapper"],
    "Dazed":    ["div.article__body", "div.entry-content", "section.article-content"],
    "Interview":["div.article-body", "div.post-content", "div.entry-content"],
    "Esquire":  ["div.article-body", "div.standard-body"]
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}

# ----------------- FUNCIONES DE EXTRACCIÓN -----------------

def limpiar_texto(texto):
    if not texto: return ""
    return " ".join(texto.split())

def extraer_fecha_bs(soup):
    """Intenta sacar la fecha manualmente buscando en los meta tags comunes."""
    try:
        metas = [
            {"property": "article:published_time"},
            {"name": "date"},
            {"name": "publication_date"},
            {"itemprop": "datePublished"},
            {"name": "parsely-pub-date"}
        ]
        
        for m in metas:
            tag = soup.find("meta", m)
            if tag:
                content = tag.get("content")
                if content:
                    return content[:10] # Nos quedamos solo con YYYY-MM-DD
        
        time_tag = soup.find("time")
        if time_tag:
            return (time_tag.get("datetime") or time_tag.get_text())[:10]
    except: pass
    return None

def procesar_con_trafilatura(html_content):
    """Usa Trafilatura para sacar Texto y Fecha."""
    try:
        texto = trafilatura.extract(html_content, include_comments=False, include_tables=False)
        metadata = trafilatura.extract_metadata(html_content)
        fecha = metadata.date if metadata and metadata.date else None
        return fecha, limpiar_texto(texto)
    except:
        return None, None

def extraer_contenido_hibrido(url, revista, driver=None):
    """Devuelve: (Fecha, Texto)"""
    
    #REQUESTS
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, "html.parser")
            
            # A) Intentamos con los selectores manuales primero
            selectores = SELECTORES.get(revista, []) + ["article", "main"]
            texto_manual = None
            
            for sel in selectores:
                elemento = soup.select_one(sel)
                if elemento:
                    t = limpiar_texto(elemento.get_text(separator=' '))
                    if len(t) > 200:
                        texto_manual = t
                        break
            
            if texto_manual:
                fecha_manual = extraer_fecha_bs(soup)
                # Si falta la fecha, pedimos ayuda a Trafilatura solo para eso
                if not fecha_manual:
                    meta = trafilatura.extract_metadata(r.content)
                    if meta: fecha_manual = meta.date
                return fecha_manual, texto_manual

            # B) Si fallan los selectores, usamos Trafilatura completo
            fecha_traf, texto_traf = procesar_con_trafilatura(r.content)
            if texto_traf and len(texto_traf) > 200:
                return fecha_traf, texto_traf
                
    except Exception as e:
        pass 

    # --- SELENIUM ---
    if driver:
        try:
            if "archive.org" not in url: 
                driver.get(url)
                time.sleep(2) 
                html_selenium = driver.page_source
                fecha_sel, texto_sel = procesar_con_trafilatura(html_selenium)
                if texto_sel and len(texto_sel) > 200:
                    return fecha_sel, texto_sel
        except: pass

    return None, None

# ----------------- PROCESO PRINCIPAL -----------------

def main():
    if not os.path.exists(input_file):
        print("ERROR: Ejecuta primero el recolector_urls.py")
        return

    # Cargar estado para no repetir
    urls_hechas = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                reader = csv.DictReader(f)
                for row in reader: urls_hechas.add(row["URL"])
            except: pass

    pendientes = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["URL"] not in urls_hechas: pendientes.append(row)

    print(f"--- EXTRACTOR FINAL ---")
    print(f"Pendientes: {len(pendientes)}")
    
    if not pendientes: return

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--log-level=3")
    options.add_argument("--blink-settings=imagesEnabled=false")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # Modo append importante para no borrar lo que ya teníamos
    f_out = open(output_file, "a", newline="", encoding="utf-8")
    
    writer = csv.DictWriter(f_out, fieldnames=["Fuente", "URL", "Fecha", "Contenido"])
    
    if os.path.getsize(output_file) == 0: writer.writeheader()

    try:
        for i, fila in enumerate(pendientes, 1):
            url = fila["URL"]
            revista = fila["Fuente"]
            
            print(f"[{i}/{len(pendientes)}] {revista}...", end=" ")
            
            fecha, contenido = extraer_contenido_hibrido(url, revista, driver)
            
            if contenido:
                if not fecha: fecha = "Desconocida"
                
                writer.writerow({
                    "Fuente": revista,
                    "URL": url,
                    "Fecha": fecha,
                    "Contenido": contenido
                })
                f_out.flush()
                print(f"OK | {len(contenido)} chars")
            else:
                print("VACÍO")

    except KeyboardInterrupt:
        print("\nStop.")
    finally:
        f_out.close()
        driver.quit()

if __name__ == "__main__":
    main()