import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------- CONFIGURACIÓN -----------------
base_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(base_dir,"dataset_pinterest","pinterest_dataset_completo.csv")
img_folder = os.path.join(base_dir, "imagenes_descargadas")
os.makedirs(img_folder, exist_ok=True)

# Headers para reducir bloqueos
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
}

# ----------------- DESCARGA DE IMAGENES -----------------
def descargar_imagen(args):
    """
    Descarga imagen si no existe y devuelve:
    (idx, ruta_local o None)
    """
    idx, url = args

    if pd.isna(url) or not str(url).startswith("http"):
        return idx, None

    try:
        # Extensión segura
        ext = url.split('.')[-1].split('?')[0].lower()
        if ext not in ["jpg", "jpeg", "png", "webp"]:
            ext = "jpg"

        nombre_archivo = f"pinterest_{idx}.{ext}"
        ruta_completa = os.path.join(img_folder, nombre_archivo)

        # 1. NO DUPLICAR
        if os.path.exists(ruta_completa):
            return idx, ruta_completa

        # 2. DESCARGA
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            with open(ruta_completa, "wb") as f:
                f.write(response.content)
            return idx, ruta_completa

    except Exception as e:
        print(f"[Error Img] {url} -> {e}")

    return idx, None


# ----------------- EJECUCIÓN PRINCIPAL -----------------
if not os.path.exists(input_csv):
    print(f"No encuentro el CSV: {input_csv}")
    exit()

print(f"Leyendo CSV: {input_csv}")
df = pd.read_csv(input_csv)

if "Imagen_URL" not in df.columns:
    print("El CSV no tiene la columna 'Imagen_URL'")
    exit()

# Creamos la columna si no existe
if "Ruta_Local_Imagen" not in df.columns:
    df["Ruta_Local_Imagen"] = None

print(f"Total filas: {len(df)}")
print("Iniciando descargas en paralelo...")

tareas = [(i, row["Imagen_URL"]) for i, row in df.iterrows()]
descargadas = 0

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(descargar_imagen, t) for t in tareas]

    for future in as_completed(futures):
        idx, ruta_local = future.result()
        if ruta_local:
            df.at[idx, "Ruta_Local_Imagen"] = ruta_local
            descargadas += 1

# Guardamos CSV actualizado
df.to_csv(input_csv, index=False)

print(f"Descarga finalizada")
print(f"Imágenes descargadas / encontradas: {descargadas}")
print(f"CSV actualizado con columna 'Ruta_Local_Imagen'")
