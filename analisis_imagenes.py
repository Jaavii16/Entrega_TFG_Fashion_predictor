import pandas as pd
import os
from PIL import Image
from transformers import pipeline
import torch

# ----------------- CONFIGURACIÓN -----------------
base_dir = os.path.dirname(os.path.abspath(__file__))
FILE_DIC = os.path.join(base_dir, "diccionario_moda.csv")

# Archivos a analizar
files = [
    os.path.join(base_dir, "dataset_instagram", "instagram_posts.csv"),
    os.path.join(base_dir, "dataset_pinterest", "pinterest_dataset_completo.csv")
]
img_folder = os.path.join(base_dir, "imagenes_descargadas")

# ----------------- CARGA DE ETIQUETAS DESDE EL DICCIONARIO DE MODA -----------------
print("--- Configurando IA de Visión ---")

if not os.path.exists(FILE_DIC):
    print(f"ERROR: No encuentro el diccionario en {FILE_DIC}")
    exit()

df_dic = pd.read_csv(FILE_DIC)

# 1. Filtramos: Queremos todo menos 'parte_prenda' porque CLIP suele fallar mucho detectándolas
mask_utiles = df_dic['category'] != 'parte_prenda'
terminos_diccionario = df_dic[mask_utiles]['term_en'].dropna().str.lower().str.strip().unique().tolist()


# 2. Lista de términos para pasar a CLIP
CANDIDATE_LABELS = list(set(terminos_diccionario))

print(f"Diccionario cargado correctamente.")
print(f"Se buscarán {len(CANDIDATE_LABELS)} conceptos visuales en cada foto.")
print(f"Ejemplos: {CANDIDATE_LABELS[:10]}...")

# ----------------- CARGA DE CLIP -----------------
print("Cargando modelo CLIP...")
try:
    # Usamos CPU por defecto, GPU si está disponible
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-image-classification", 
                          model="openai/clip-vit-base-patch32", 
                          device=device)
except Exception as e:
    print(f"Error cargando modelo: {e}")
    exit()

def analizar_dataset(csv_path):
    if not os.path.exists(csv_path):
        print(f"Saltando {csv_path} (No existe)")
        return

    print(f"\nProcesando archivo: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    
    # Crear columna si no existe
    if 'Etiquetas_Visuales' not in df.columns:
        df['Etiquetas_Visuales'] = ""

    total = len(df)
    procesados = 0
    modificados = 0
    

    # Usar un umbral de 0.02 (2%) es suficiente para quitar ruido puro, pero dejar pasar algunas prendas
    UMBRAL_MINIMO = 0.02 
    
    for i, row in df.iterrows():
        # Si ya tiene etiquetas, saltamos
        if pd.notna(row['Etiquetas_Visuales']) and str(row['Etiquetas_Visuales']).strip() != "":
            continue
            
        ruta_relativa = row.get('Ruta_Local_Imagen')
        if not ruta_relativa: continue 
        
        path_completo = os.path.join(img_folder, str(ruta_relativa))
        
        if os.path.exists(path_completo):
            try:
                image = Image.open(path_completo)
                
                # --- ANÁLISIS IA ---
                # CLIP devuelve una lista ordenada de mayor a menor probabilidad
                results = classifier(image, candidate_labels=CANDIDATE_LABELS)
                
                
                # 1. Cogemos siempre los 4 primeros resultados (los que la IA cree que son más probables)
                top_resultados = results[:4]
                
                # 2. Aplicamos el filtro que marcamos antes con el umbral solo para quitar errores graves
                tags_finales = [res['label'] for res in top_resultados if res['score'] > UMBRAL_MINIMO]
                
                # 3. Guardamos
                if tags_finales:
                    tags_str = ", ".join(tags_finales)
                    df.at[i, 'Etiquetas_Visuales'] = tags_str
                    modificados += 1
                    
                    if modificados % 10 == 0:
                        # Imprimimos también el score 
                        score_top = round(top_resultados[0]['score'], 3)
                        print(f"   [Foto {i}] (Confianza {score_top}): {tags_str}")
                else:
                    # Si no encuentra nada, ponemos 'fashion' por defecto para no dejarlo vacío
                    df.at[i, 'Etiquetas_Visuales'] = "fashion"

            except Exception as e:
                print(f"   [Error Foto] {ruta_relativa}: {e}")
        
        procesados += 1
        
        # Guardado de seguridad
        if modificados > 0 and modificados % 20 == 0:
            df.to_csv(csv_path, index=False)
            print(f"   (Guardado parcial: {modificados} fotos)")

    # Guardado final y unificación
    df['Texto_Completo'] = df['Texto'].fillna("") + " " + df['Etiquetas_Visuales'].fillna("")
    df.to_csv(csv_path, index=False)
    print(f"Finalizado {os.path.basename(csv_path)}. {modificados} imágenes procesadas.")

# ----------------- EJECUCIÓN -----------------
for f in files:
    analizar_dataset(f)