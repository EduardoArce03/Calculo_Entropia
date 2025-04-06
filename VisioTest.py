import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
from skimage import io, color
from scipy.stats import entropy

# === Función para descomprimir imágenes ===
def unzip_images(zip_path, extract_to):
    """
    Descomprime un archivo ZIP en la carpeta especificada.
    """
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(extract_to)


# === Función para calcular la entropía de una imagen ===
def calculate_entropy(image):
    # Función que calcula la entropía de una imagen
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalizamos el histograma
    return -np.sum(hist * np.log2(hist + 1e-6))  # Evitar log(0) añadiendo un valor pequeño

# === Función para procesar imágenes y calcular entropías ===
def process_images(root_folder, gray_output_folder):
    """
    Procesa todas las imágenes en un directorio y calcula la entropía
    tanto en color como en escala de grises. También guarda las imágenes en gris.
    """
    entropies_color = {}
    entropies_gray = {}

    for dataset in os.listdir(root_folder):
        dataset_path = os.path.join(root_folder, dataset)
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if os.path.isdir(category_path):
                key = f"{dataset}/{category}"
                entropies_color[key] = []
                entropies_gray[key] = []

                gray_category_path = os.path.join(gray_output_folder, dataset, category)
                os.makedirs(gray_category_path, exist_ok=True)

                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    if os.path.isfile(img_path):
                        img = io.imread(img_path)

                        # --- Entropía en color ---
                        entropies_color[key].append(calculate_entropy(img))

                        # --- Verificar si la imagen ya está en escala de grises ---
                        if len(img.shape) == 3:  # Imagen a color
                            gray_img = color.rgb2gray(img)
                        else:  # Imagen ya en escala de grises
                            gray_img = img

                        entropies_gray[key].append(calculate_entropy(gray_img))

                        # Guardar imagen en escala de grises
                        gray_img_uint8 = (gray_img * 255).astype(np.uint8)
                        output_path = os.path.join(gray_category_path, img_name)
                        io.imsave(output_path, gray_img_uint8)

    return entropies_color, entropies_gray



# === Función para graficar diagramas de cajas y bigotes ===
def plot_boxplot(entropies, title):
    """
    Genera un diagrama de cajas y bigotes para visualizar la distribución de la entropía por categoría.
    """
    data = [(category, value) for category, values in entropies.items() for value in values]
    categories, values = zip(*data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=categories, y=values)
    plt.title(title)
    plt.xlabel('Categorías')
    plt.ylabel('Entropía')
    plt.xticks(rotation=45)
    plt.show()


# === Función para aplicar modificaciones a una imagen ===
import numpy as np
import skimage.util as util

def generate_modification_params(image_shape, level):
    h, w = image_shape[:2]

    if level == "small":
        size = min(100, h // 2, w // 2)
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        return {"level": level, "size": size, "y": y, "x": x}

    elif level == "medium":
        size = min(300, h // 2, w // 2)
        y_src = random.randint(0, h - size * 2)
        x_src = random.randint(0, w - size * 2)
        y_dst = random.randint(0, h - size)
        x_dst = random.randint(0, w - size)
        return {"level": level, "size": size, "y_src": y_src, "x_src": x_src, "y_dst": y_dst, "x_dst": x_dst}

    elif level == "large":
        size = min(100, h // 3, w // 3)
        y_start = random.randint(0, h - size)
        x_start = random.randint(0, w - size)

        glitch_blocks = []
        for _ in range(3):
            block_size = min(60, h // 4, w // 4)
            y1 = random.randint(0, h - block_size)
            x1 = random.randint(0, w - block_size)
            y2 = random.randint(0, h - block_size)
            x2 = random.randint(0, w - block_size)
            glitch_blocks.append((block_size, y1, x1, y2, x2))

        return {
            "level": level,
            "size": size,
            "y_start": y_start,
            "x_start": x_start,
            "glitch_blocks": glitch_blocks
        }


def modify_image_with_params(image, params):
    modified = image.copy()
    h, w = modified.shape[:2]
    level = params["level"]

    if level == "small":
        y, x, size = params["y"], params["x"], params["size"]
        if len(modified.shape) == 3:
            modified[y:y + size, x:x + size] = [255, 0, 0]
        else:
            modified[y:y + size, x:x + size] = 255

    elif level == "medium":
        size = params["size"]
        y_src, x_src = params["y_src"], params["x_src"]
        y_dst, x_dst = params["y_dst"], params["x_dst"]
        modified[y_dst:y_dst + size, x_dst:x_dst + size] = modified[y_src:y_src + size, x_src:x_src + size]
    elif level == "large":
        size = params["size"]
        y_start, x_start = params["y_start"], params["x_start"]

        # 1. Rotar una región aleatoria 180 grados
        region = modified[y_start:y_start + size, x_start:x_start + size]
        rotated = np.rot90(region, k=2)  # Rotación de 180 grados
        modified[y_start:y_start + size, x_start:x_start + size] = rotated

        # 2. Efecto glitch más dramático: más bloques con tamaños variados y desplazamientos aleatorios
        for _ in range(5):  # Aumentamos la cantidad de bloques para más glitch
            block_size = random.randint(40, 100)  # Tamaños de bloque aleatorios
            y1 = random.randint(0, h - block_size)
            x1 = random.randint(0, w - block_size)
            y2 = random.randint(0, h - block_size)
            x2 = random.randint(0, w - block_size)

            block_data = modified[y1:y1 + block_size, x1:x1 + block_size].copy()
            modified[y2:y2 + block_size, x2:x2 + block_size] = block_data

        # 3. Distorsionar aún más partes de la imagen mediante un "shift" o desplazamiento en una sección
        shift_size = random.randint(50, 150)  # Desplazamiento aleatorio para la distorsión
        y_shift = random.randint(0, h - shift_size)
        x_shift = random.randint(0, w - shift_size)

        # Cortar y mover la sección hacia un nuevo lugar
        section = modified[y_shift:y_shift + shift_size, x_shift:x_shift + shift_size]
        new_y = random.randint(0, h - shift_size)
        new_x = random.randint(0, w - shift_size)
        modified[new_y:new_y + shift_size, new_x:new_x + shift_size] = section

    return modified



# === Función para analizar cambios en la entropía tras modificaciones ===
def analyze_entropy_changes_with_keys(root_folder, gray_folder, num_images=10, output_folder="html_output"):
    os.makedirs(output_folder, exist_ok=True)

    selected_images = []
    image_info_list = []

    for dataset in os.listdir(root_folder):
        dataset_path = os.path.join(root_folder, dataset)
        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            if os.path.isdir(category_path):
                key = f"{dataset}/{category}"
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    if os.path.isfile(img_path):
                        selected_images.append((img_path, key))

    random.shuffle(selected_images)
    selected_images = selected_images[:num_images]

    entropy_changes = {
        "color": {"original": [], "small": [], "medium": [], "large": []},
        "gray": {"original": [], "small": [], "medium": [], "large": []}
    }

    for idx, (img_path, key) in enumerate(selected_images):
        img_color = io.imread(img_path)
        relative_path = os.path.relpath(img_path, root_folder)
        gray_img_path = os.path.join(gray_folder, relative_path)
        gray_img = io.imread(gray_img_path)

        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Clave: {key} | Imagen: {os.path.basename(img_path)}")

        if len(img_color.shape) == 2:
            img_color = color.gray2rgb(img_color)
        if len(gray_img.shape) == 3 and gray_img.shape[2] == 3:
            gray_img = color.rgb2gray(gray_img)

        orig_entropy_color = calculate_entropy(img_color)
        entropy_changes["color"]["original"].append(orig_entropy_color)
        axs[0, 0].imshow(img_color)
        axs[0, 0].set_title(f"Color Original ({orig_entropy_color:.2f})")
        axs[0, 0].axis("off")

        orig_entropy_gray = calculate_entropy(gray_img)
        entropy_changes["gray"]["original"].append(orig_entropy_gray)
        axs[1, 0].imshow(gray_img, cmap='gray')
        axs[1, 0].set_title(f"Gris Original ({orig_entropy_gray:.2f})")
        axs[1, 0].axis("off")

        for i, level in enumerate(["small", "medium", "large"]):
            params = generate_modification_params(img_color.shape, level)

            mod_color = modify_image_with_params(img_color, params)
            mod_gray = modify_image_with_params(gray_img, params)

            ent_color = calculate_entropy(mod_color)
            ent_gray = calculate_entropy(mod_gray)

            entropy_changes["color"][level].append(ent_color)
            entropy_changes["gray"][level].append(ent_gray)

            axs[0, i + 1].imshow(mod_color)
            axs[0, i + 1].set_title(f"{level.capitalize()} ({ent_color:.2f})")
            axs[0, i + 1].axis("off")

            axs[1, i + 1].imshow(mod_gray, cmap='gray')
            axs[1, i + 1].set_title(f"{level.capitalize()} ({ent_gray:.2f})")
            axs[1, i + 1].axis("off")

        plt.tight_layout()

        img_filename = f"panel_{idx+1}_{os.path.basename(img_path).replace('.', '_')}.png"
        img_output_path = os.path.join(output_folder, img_filename)
        plt.savefig(img_output_path)
        plt.close(fig)

        # Guardamos la información para el HTML por separado
        image_info_list.append({
            "key": key,
            "image_name": os.path.basename(img_path),
            "panel_path": img_filename
        })

    return entropy_changes, image_info_list



# === Función para graficar un gráfico de barras ===
def plot_bar_chart_comparison(entropy_data):
    """
    Genera dos gráficos de barras separados para comparar entropía en imágenes a color y en escala de grises.
    """
    levels = ["original", "small", "medium", "large"]
    num_images = len(entropy_data["color"]["original"])
    index = np.arange(num_images)
    bar_width = 0.2

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Comparación de Entropía: Color vs Gris")

    # --- GRÁFICO 1: COLOR ---
    for i, level in enumerate(levels):
        axs[0].bar(index + i * bar_width, entropy_data["color"][level], width=bar_width, label=level.capitalize())
    axs[0].set_title("Imágenes a Color")
    axs[0].set_xlabel("Índice de Imagen")
    axs[0].set_ylabel("Entropía")
    axs[0].set_xticks(index + bar_width * 1.5)
    axs[0].set_xticklabels(range(1, num_images + 1))
    axs[0].legend()

    # --- GRÁFICO 2: GRIS ---
    for i, level in enumerate(levels):
        axs[1].bar(index + i * bar_width, entropy_data["gray"][level], width=bar_width, label=level.capitalize())
    axs[1].set_title("Imágenes en Escala de Grises")
    axs[1].set_xlabel("Índice de Imagen")
    axs[1].set_xticks(index + bar_width * 1.5)
    axs[1].set_xticklabels(range(1, num_images + 1))
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def analyze_entropy_by_category(entropies):
    """
    Analiza qué clases o categorías tienen un mayor nivel de entropía y explica posibles causas.

    Parámetros:
    - entropies: diccionario con categorías como claves y listas de valores de entropía como valores.

    Retorna:
    - Un diccionario con la entropía media por categoría ordenada de mayor a menor.
    """
    avg_entropy = {category: np.mean(values) for category, values in entropies.items() if values}

    # Ordenar las categorías por entropía media en orden descendente
    sorted_entropy = dict(sorted(avg_entropy.items(), key=lambda item: item[1], reverse=True))

    print("\n📊 **Entropía media por categoría (ordenada de mayor a menor):**")
    for category, entropy_value in sorted_entropy.items():
        print(f"- {category}: {entropy_value:.4f}")

    # Explicación del fenómeno
    print("\n🔍 **Análisis de entropía:**")
    print("Las categorías con mayor entropía suelen ser aquellas con:")
    print("- Mayor variabilidad en las imágenes (formas, colores, texturas, iluminación).")
    print("- Objetos con muchos detalles o texturas complejas.")
    print("- Fondos no uniformes o con ruido visual.")

    return sorted_entropy

def generate_entropy_html(image_info_list, output_folder="html_output", html_filename="resultado_entropia.html"):
    html_path = os.path.join(output_folder, html_filename)
    html_lines = [
        "<html><head><title>Resultados de Entropía</title></head><body>",
        "<h1>Análisis de Entropía de Imágenes</h1>"
    ]

    for info in image_info_list:
        html_lines.append(f"<h2>Clave: {info['key']}</h2>")
        html_lines.append(f"<p><b>Imagen:</b> {info['image_name']}</p>")
        html_lines.append(f"<img src='{info['panel_path']}' style='width:100%; max-width:800px'><br><br>")

    html_lines.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

    print(f"\n✅ HTML generado en: {html_path}")


# === Ejecución principal ===
if __name__ == "__main__":
    # Ruta al directorio de imágenes
    root_folder = "images"
    gray_output_folder = "gray_output"

    # Paso 1: Procesar imágenes y calcular entropías
    print("Calculando entropías...")
    entropies_color, entropies_gray = process_images(root_folder, gray_output_folder)

    # Paso 2: Graficar diagramas de cajas y bigotes
    print("Generando diagramas de cajas y bigotes...")
    plot_boxplot(entropies_color, "Entropía de Imágenes a Color")
    plot_boxplot(entropies_gray, "Entropía de Imágenes en Escala de Grises")

    # Paso 3: Analizar cambios en la entropía tras modificaciones
    print("Analizando cambios en la entropía tras modificaciones...")
    entropy_changes, image_info_list = analyze_entropy_changes_with_keys(root_folder, gray_output_folder , num_images=10)
    generate_entropy_html(image_info_list)
    # Paso 4: Graficar gráfico de barras
    print("Generando gráfico de barras...")
    plot_bar_chart_comparison(entropy_changes)

    sorted_entropies_color = analyze_entropy_by_category(entropies_color)
    sorted_entropies_gray = analyze_entropy_by_category(entropies_gray)
