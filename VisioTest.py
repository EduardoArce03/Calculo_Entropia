import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
from skimage import io, color
from scipy.stats import entropy
from skimage.measure import shannon_entropy


# === Función para descomprimir imágenes ===
def unzip_images(zip_path, extract_to):
    """
    Descomprime un archivo ZIP en la carpeta especificada.
    """
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(extract_to)

def plot_and_save_histograms(image_color, image_gray, hist_color_path, hist_gray_path, title):
    """
    Grafica y guarda los histogramas de una imagen en escala de grises y a color.
    Incluye un título descriptivo en el gráfico.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    # Histograma en escala de grises
    hist_gray, bins_gray = np.histogram(image_gray.flatten(), bins=256, range=[0, 256])
    axes[0].plot(bins_gray[:-1], hist_gray, color='black')
    axes[0].set_title("Histograma en Escala de Grises")
    axes[0].set_xlabel("Intensidad")
    axes[0].set_ylabel("Frecuencia")

    # Histograma a color (RGB)
    colors = ('red', 'green', 'blue')
    for i, color in enumerate(colors):
        hist_color, bins_color = np.histogram(image_color[:, :, i].flatten(), bins=256, range=[0, 256])
        axes[1].plot(bins_color[:-1], hist_color, color=color)
    axes[1].set_title("Histograma a Color (RGB)")
    axes[1].set_xlabel("Intensidad")
    axes[1].set_ylabel("Frecuencia")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(hist_gray_path)
    plt.savefig(hist_color_path)
    plt.close(fig)

# === Función para calcular la entropía de una imagen ===
def calculate_entropy(image):
    """
    Calcula la entropía de una imagen utilizando shannon_entropy de scikit-image,
    replicando el comportamiento de np.histogram con 256 bins y density=True.

    :param image: Imagen de entrada (puede estar en color o en escala de grises).
    :return: Valor de entropía.
    """
    # Escalar la imagen al rango [0, 255] y convertirla a uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Calcular la entropía usando shannon_entropy
    return shannon_entropy(image)

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
def plot_and_save_boxplots(entropies_gray, entropies_color, output_folder="html_output"):
    """
    Genera y guarda dos diagramas de cajas y bigotes:
    - Uno para entropía en escala de grises.
    - Uno para entropía a color.
    """
    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Función auxiliar para graficar un boxplot
    def plot_single_boxplot(entropies, title, file_name):
        data = [(category, value) for category, values in entropies.items() for value in values]
        categories, values = zip(*data)

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=categories, y=values)
        plt.title(title)
        plt.xlabel('Categorías')
        plt.ylabel('Entropía')
        plt.xticks(rotation=45)

        # Guardar el gráfico como archivo
        file_path = os.path.join(output_folder, file_name)
        plt.savefig(file_path)
        plt.close()

        return file_path

    # Generar y guardar el boxplot para escala de grises
    gray_boxplot_path = plot_single_boxplot(
        entropies_gray,
        title="Distribución de Entropía en Escala de Grises",
        file_name="boxplot_gray.png"
    )

    # Generar y guardar el boxplot para color
    color_boxplot_path = plot_single_boxplot(
        entropies_color,
        title="Distribución de Entropía a Color",
        file_name="boxplot_color.png"
    )

    print(f"✅ Boxplots generados y guardados en: {output_folder}")
    return gray_boxplot_path, color_boxplot_path


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

        # Verificar si las imágenes están en escala de grises o a color
        if len(img_color.shape) == 2:
            img_color = color.gray2rgb(img_color)
        if len(gray_img.shape) == 3 and gray_img.shape[2] == 3:
            gray_img = color.rgb2gray(gray_img)

        # Generar título para los histogramas
        img_name = os.path.basename(img_path)
        histogram_title = f"Histogramas - Clave: {key} | Imagen: {img_name}"

        # Guardar histogramas
        hist_gray_path = os.path.join(output_folder, f"hist_gray_{idx+1}.png")
        hist_color_path = os.path.join(output_folder, f"hist_color_{idx+1}.png")
        plot_and_save_histograms(img_color, gray_img, hist_color_path, hist_gray_path, histogram_title)

        # Procesar modificaciones
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Clave: {key} | Imagen: {os.path.basename(img_path)}")

        # Mostrar imagen original y calcular entropía
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

        # Guardar panel de imágenes
        panel_filename = f"panel_{idx+1}_{os.path.basename(img_path).replace('.', '_')}.png"
        panel_path = os.path.join(output_folder, panel_filename)
        plt.savefig(panel_path)
        plt.close(fig)

        # Guardar información para el HTML
        image_info_list.append({
            "key": key,
            "image_name": os.path.basename(img_path),
            "panel_path": panel_filename,
            "hist_gray_path": os.path.basename(hist_gray_path),
            "hist_color_path": os.path.basename(hist_color_path)
        })

    return entropy_changes, image_info_list

# === Función para graficar un gráfico de barras ===
def plot_bar_chart_comparison(entropy_data, output_folder="html_output", filename="bar_chart.png"):
    """
    Genera dos gráficos de barras separados para comparar entropía en imágenes a color y en escala de grises.
    Guarda el gráfico como un archivo de imagen.

    Parámetros:
    - entropy_data: Diccionario con los datos de entropía.
    - output_folder: Carpeta donde se guardará el gráfico.
    - filename: Nombre del archivo de salida.

    Retorna:
    - Ruta completa al archivo de imagen generado.
    """
    levels = ["original", "small", "medium", "large"]
    num_images = len(entropy_data["color"]["original"])
    index = np.arange(num_images)
    bar_width = 0.2

    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Crear los gráficos de barras
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

    # Ajustar el diseño
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    file_path = os.path.join(output_folder, filename)
    plt.savefig(file_path)
    plt.close(fig)

    print(f"✅ Gráfico de barras guardado en: {file_path}")
    return file_path

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
    return sorted_entropy


def generate_entropy_html(image_info_list, gray_boxplot_path, color_boxplot_path, sorted_entropies_color,
                          sorted_entropies_gray, output_folder="html_output", html_filename="resultado_entropia.html", bar_chart_path='bar_chart.png'):
    """
    Genera un archivo HTML con los resultados del análisis de entropía, incluyendo:
    - Diagramas de caja y bigotes.
    - Entropía media por categoría.
    - Análisis textual de las categorías.
    - Paneles de imágenes con sus histogramas.

    Parámetros:
    - image_info_list: Lista de información sobre las imágenes procesadas.
    - gray_boxplot_path: Ruta del diagrama de caja y bigotes para imágenes en escala de grises.
    - color_boxplot_path: Ruta del diagrama de caja y bigotes para imágenes en color.
    - sorted_entropies_color: Diccionario con la entropía media por categoría (color).
    - sorted_entropies_gray: Diccionario con la entropía media por categoría (escala de grises).
    - output_folder: Carpeta de salida para el archivo HTML.
    - html_filename: Nombre del archivo HTML generado.
    """
    with open("VisioTest.py", "r", encoding="utf-8") as f:
        script_content = f.read()

    # Convertir las rutas absolutas a relativas
    gray_boxplot_relpath = os.path.relpath(gray_boxplot_path, output_folder)
    color_boxplot_relpath = os.path.relpath(color_boxplot_path, output_folder)
    bar_chart_relpath = os.path.basename(bar_chart_path)
    import html
    escaped_script_content = html.escape(script_content)

    # Inicializar el contenido HTML
    html_lines = [
        "<html>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "<title>Resultados de Entropía</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; background-color: #f9f9f9; color: #333; }",
        "h1, h2 { color: #2c3e50; }",
        "h1 { text-align: center; margin-bottom: 20px; }",
        "h2 { border-bottom: 2px solid #2c3e50; padding-bottom: 5px; margin-top: 30px; }",
        "p { margin: 10px 0; }",
        "ul { margin: 10px 0; padding-left: 20px; }",
        "li { margin: 5px 0; }",
        "table { width: 100%; border-collapse: collapse; margin: 20px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #2c3e50; color: white; }",
        "img { max-width: 100%; height: auto; display: block; margin: 10px 0; }",
        ".container { max-width: 1200px; margin: 0 auto; }",
        "</style>",
        "<h2>Nombre: Eduardo Arce</h2>",
        "</head>",
        "<body>",
        "<h2>Nombre: Eduardo Arce</h2>",
        "<div class='container'>",
        "<h1>Análisis de Entropía de Imágenes</h1>",
        "<p><b>Explicación sobre la Entropía:</b></p>",
        "<p>En términos generales, la entropía mide la cantidad de incertidumbre, desorden o información en una imagen. Esto se calcula a partir de la distribución de intensidades de píxeles en la imagen.</p><br>",
        "<h2>Entropía en Imágenes en Escala de Grises</h2>",
        "<p>Una imagen en escala de grises tiene un solo canal de intensidad, donde cada píxel representa un valor de luminosidad entre 0 (negro) y 255 (blanco).</p>",
        "<h2>Entropía en Imágenes en Color</h2>",
        "<p>Una imagen a color tiene tres canales principales: rojo (R), verde (G) y azul (B). Cada canal es similar a una imagen en escala de grises, pero representa una componente específica del color.</p>",
        "<h2>Diagramas de Cajas y Bigotes</h2>",
        f"<img src='{gray_boxplot_relpath}' alt='Boxplot de Entropía en Escala de Grises'><br>",
        "<p><b>Boxplot de Entropía en Escala de Grises</b></p>",
        f"<img src='{color_boxplot_relpath}' alt='Boxplot de Entropía a Color'><br>",
        "<p><b>Boxplot de Entropía a Color</b></p><br>"
    ]

    for info in image_info_list:
        html_lines.append(f"<h2>Clave: {info['key']}</h2>")
        html_lines.append(f"<p><b>Imagen:</b> {info['image_name']}</p>")
        html_lines.append(f"<img src='{info['panel_path']}' alt='Panel de Imagen'><br>")
        html_lines.append("<h3>Histogramas:</h3>")
        html_lines.append(f"<img src='{info['hist_gray_path']}' alt='Histograma en Escala de Grises'>")
        html_lines.append(f"<img src='{info['hist_color_path']}' alt='Histograma a Color'><br><br>")

    html_lines.append("<h2>Entropía Media por Categoría</h2>")
    html_lines.append("<table>")
    html_lines.append("<tr><th>Categoría</th><th>Entropía Media (Color)</th><th>Entropía Media (Gris)</th></tr>")
    for category in sorted_entropies_color.keys():
        color_entropy = sorted_entropies_color.get(category, "N/A")
        gray_entropy = sorted_entropies_gray.get(category, "N/A")
        html_lines.append(f"<tr><td>{category}</td><td>{color_entropy:.4f}</td><td>{gray_entropy:.4f}</td></tr>")
    html_lines.append("</table>")

    html_lines.append("<h2>Análisis de Entropía por Categoría</h2>")
    html_lines.append("<p>Las categorías con mayor entropía suelen ser aquellas con:</p>")
    html_lines.append("<ul>")
    html_lines.append(
        "<li><strong>Mayor variabilidad en las imágenes:</strong> Por lo general imágenes de perros, ya que estos tienen características más complejas, como por ejemplo el pelaje más denso.</li>")
    html_lines.append(
        "<li><strong>Objetos con muchos detalles o texturas complejas:</strong> Por ejemplo, animales con patrones intrincados.</li>")
    html_lines.append(
        "<li><strong>Fondos no uniformes o con ruido visual:</strong> Fondos con más elementos o menos homogeneidad.</li>")
    html_lines.append("</ul>")
    html_lines.append("<p>")
    html_lines.append(
        "En la mayoría de casos, la categoría <strong>test/dogs</strong> tiene la mayor entropía media, seguida de <strong>train/cats</strong>. ")
    html_lines.append(
        "Esto puede deberse a que las imágenes de perros tienen más variabilidad en sus características visuales, como patrones de pelaje, posturas y fondos.")
    html_lines.append("</p><br>")

    html_lines.append("<h2>Gráfico de Barras de Comparación de Entropía</h2>")
    html_lines.append(f"<img src='{bar_chart_relpath}' alt='Gráfico de Barras'><br>")

    html_lines.append("<h2>Código Fuente del Script</h2>")
    html_lines.append("<pre><code>")
    html_lines.append(escaped_script_content)
    html_lines.append("</code></pre>")

    html_lines.append("</div></body></html>")

    html_path = os.path.join(output_folder, html_filename)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

    print(f"\n✅ HTML generado en: {html_path}")

import os

def genpdf():
    from weasyprint import HTML

    html_file = "html_output/resultado_entropia.html"

    pdf_file = "html_output/resultado_entropia.pdf"

    HTML(html_file).write_pdf(pdf_file)

    print(f"✅ PDF generado en: {pdf_file}")


# === Ejecución principal ===
if __name__ == "__main__":
    root_folder = "images"
    gray_output_folder = "gray_output"
    output_folder = "html_output"

    print("Calculando entropías...")
    entropies_color, entropies_gray = process_images(root_folder, gray_output_folder)
    print("color",entropies_color)
    print("gris",entropies_gray)
    print("Generando diagramas de cajas y bigotes...")
    gray_boxplot_path, color_boxplot_path = plot_and_save_boxplots(
        entropies_gray=entropies_gray,
        entropies_color=entropies_color,
        output_folder=output_folder
    )
    print("Analizando cambios en la entropía tras modificaciones...")
    entropy_changes, image_info_list = analyze_entropy_changes_with_keys(
        root_folder=root_folder,
        gray_folder=gray_output_folder,
        num_images=10,
        output_folder=output_folder
    )
    sorted_entropies_color = analyze_entropy_by_category(entropies_color)
    sorted_entropies_gray = analyze_entropy_by_category(entropies_gray)

    generate_entropy_html(
        image_info_list=image_info_list,
        gray_boxplot_path=gray_boxplot_path,
        color_boxplot_path=color_boxplot_path,
        sorted_entropies_color=sorted_entropies_color,
        sorted_entropies_gray=sorted_entropies_gray,
        output_folder=output_folder,
        html_filename="resultado_entropia.html"
    )

    print("Generando gráfico de barras...")
    plot_bar_chart_comparison(entropy_changes)
    genpdf()
