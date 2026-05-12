import pandas as pd
from googletrans import Translator
import os

# 1. Listas de términos de moda en inglés
prendas_en = ["dress", "coat", "jacket", "skirt", "pants", "shirt", "blouse",
              "sweater", "cardigan", "trench coat", "bomber jacket", "jeans",
              "maxi dress", "mini skirt", "hoodie", "t-shirt", "cargo pants", "shorts",
              "jumpsuit", "romper", "blazer", "vest", "tunic", "culottes", "suit",
              "overalls", "peacoat", "parka", "windbreaker", "anorak", "tank top", 
              "crop top", "puffer jacket", "denim jacket", "leather jacket", "sundress",
              "bodycon dress", "wrap dress", "pencil skirt", "pleated skirt"]
colores_en = ["red", "blue", "green", "yellow", "pink", "brown", "black", "white",
              "beige", "gray", "turquoise", "navy", "burgundy", "khaki", "olive",
              "camel", "periwinkle", "magenta", "lavender", "mint", "coral", "teal", "mustard", 
              "charcoal", "cream", "fuchsia", "indigo", "peach", "plum", "salmon", "tan", "violet",
              "apricot", "aqua", "bronze", "copper", "emerald", "gold", "silver", "ivory", "purple",
              "violet"]
materiales_en = ["leather", "denim", "lace", "silk", "cotton", "wool", "velvet",
                 "linen", "corduroy", "satin", "suede", "chiffon", "faux leather", "plastic",
                 "nylon", "polyester", "rayon", "spandex", "tulle", "velour", "viscose"]
estilos_en = ["casual", "vintage", "minimalist", "avant-garde", "bohemian",
              "streetwear", "preppy", "y2k", "grunge", "athleisure", "punk",
              "gothic", "retro", "chic", "classic", "formal", "business casual",
              "romantic", "edgy", "hippie", "sophisticated", "eclectic", "modern", "futuristic", "artsy",
              "androgynous", "sporty", "urban", "festival", "kawaii", "cyberpunk", "steampunk", "rocker",
              "boho-chic"]
partes_prenda_en = ["sleeve", "collar", "waistband", "hem", "cuff", "pocket", "lapel", "zipper",
                    "button", "hood", "belt", "lining", "pleat", "ruffle", "fringe", "panel",
                    "yoke", "dart", "gather", "slit", "vent", "strap", "buckle", "drawstring",
                    "cowl", "peplum", "placket", "epaulet", "tab", "gusset", "inset", "overlay",
                    "trim", "applique", "embroidery", "patch", "seam", "tassel"]

# Unificamos todos los términos en una sola lista con su categoría
terms_en = []
for t in prendas_en: 
    terms_en.append((t, "prenda"))
for t in colores_en: 
    terms_en.append((t, "color"))
for t in materiales_en: 
    terms_en.append((t, "material"))
for t in estilos_en: 
    terms_en.append((t, "estilo"))
for t in partes_prenda_en: 
    terms_en.append((t, "parte_prenda"))

# 2. Traducción al español
translator = Translator()
terms_bilingual = []
for term_en, category in terms_en:
    # Limpieza básica:
    term_clean = term_en.lower().strip()
    # traducir al español:
    try:
        res = translator.translate(term_clean, src="en", dest="es")
        term_es = res.text.lower()
    except Exception as e:
        term_es = term_clean  # fallback si falla la traducción
    terms_bilingual.append({"term_en": term_clean, "term_es": term_es, "category": category})

# 3. Creamos el DataFrame y guardamos en CSV
df_dict = pd.DataFrame(terms_bilingual)
# Eliminar duplicados
df_dict = df_dict.drop_duplicates(subset=["term_en"])
output_path = os.path.join(os.getcwd(), "diccionario_moda.csv")
df_dict.to_csv(output_path, index=False, encoding="utf-8")
print(f"Diccionario generado con {len(df_dict)} términos bilingües.")
print(f"Archivo guardado en: {output_path}")
