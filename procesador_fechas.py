import csv
import os
import re

# ----------------- CONFIGURACIÓN -----------------
base_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_dir, "dataset_revistas", "dataset_final_contenido.csv")
output_file = os.path.join(base_dir, "dataset_revistas", "dataset_final_normalizado.csv")

# Mapa de meses en español e inglés 
MESES = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12",
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12"
}

def extraer_fecha_wayback(url):
    match = re.search(r'/web/(\d{4})(\d{2})(\d{2})', url)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None

def normalizar_fecha(fecha_texto, url):
    if not fecha_texto: fecha_texto = ""
    fecha_texto = str(fecha_texto).lower().strip()

    # 1. Prioridad Wayback
    if "web.archive.org" in url:
        f_wayback = extraer_fecha_wayback(url)
        if f_wayback: return f_wayback

    # 2. ISO (2023-05-12)
    match_iso = re.search(r'(\d{4})-(\d{2})-(\d{2})', fecha_texto)
    if match_iso:
        return f"{match_iso.group(1)}-{match_iso.group(2)}-{match_iso.group(3)}"

    # 3. Texto (12 de mayo de 2023)
    match_es = re.search(r'(\d{1,2})\s*(?:de|/|-)?\s*([a-z]+)\s*(?:de|/|-|,)?\s*(\d{4})', fecha_texto)
    if match_es:
        dia, mes_txt, anio = match_es.groups()
        if mes_txt in MESES:
            return f"{anio}-{MESES[mes_txt]}-{dia.zfill(2)}"

    return None

# ----------------- EJECUCIÓN -----------------

if not os.path.exists(input_file):
    print(f"[ERROR] No encuentro el archivo: {input_file}")
    exit()

print(f"Leyendo: {input_file}")
print("---------------------------------------------------")

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", newline="", encoding="utf-8") as f_out:
    
    reader = csv.DictReader(f_in)
    
    # Aseguramos las columnas correctas
    fieldnames = ["Fuente", "URL", "Fecha", "Contenido"]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    
    total_leido = 0
    guardados = 0
    descartados = 0
    
    for row in reader:
        # 1. Filtro de cabeceras repetidas y filas vacías
        if row.get("Fuente") == "Fuente" or not row.get("URL"):
            continue

        original = row.get("Fecha", "")
        url = row.get("URL", "")
        
        # 2. Intentamos obtener fecha válida
        nueva_fecha = normalizar_fecha(original, url)
        
        # Si no hay fecha (es None), saltamos al siguiente sin guardar
        if not nueva_fecha:
            descartados += 1
            continue
            
        # Si hay fecha, actualizamos y guardamos
        row["Fecha"] = nueva_fecha
        
        # Limpiamos columnas extrañas
        fila_limpia = {k: row.get(k, "") for k in fieldnames}
        
        writer.writerow(fila_limpia)
        guardados += 1
        total_leido += 1

print("\n---------------------------------------------------")
print(f"PROCESO COMPLETADO")
print(f"   -> Artículos procesados: {total_leido + descartados}")
print(f"   -> Artículos GUARDADOS (Con Fecha): {guardados}")
print(f"   -> Artículos DESCARTADOS (Sin Fecha): {descartados}")
print(f"Archivo final listo: {output_file}")
print("---------------------------------------------------")