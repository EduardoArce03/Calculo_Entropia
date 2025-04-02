import zipfile
import numpy as np
import skimage.io as io
import skimage.color as color
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import json
import random
import skimage.util as util

def unzip_images(zip_path, extract_to):
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(extract_to)

def calculate_entropy(image):
    hist, _ = np.histogram(image.ravel(), bins=256, density=True)
    return entropy(hist)

def process_images(root_folder):
    entropies_color = {}
    entropies_gray = {}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for dataset in os.listdir(root_folder):
        dataset_path = os.path.join(root_folder, dataset)

        if os.path.isdir(dataset_path):
            for category in os.listdir(dataset_path):
                category_path = os.path.join(dataset_path, category)

                if os.path.isdir(category_path):
                    key = f"{dataset}/{category}"
                    entropies_color[key] = []
                    entropies_gray[key] = []

                    for img_name in os.listdir(category_path):
                        img_path = os.path.join(category_path, img_name)

                        if os.path.isfile(img_path) and any(img_name.lower().endswith(ext) for ext in valid_extensions):
                            img = io.imread(img_path)

                            # Verificar si la imagen es en escala de grises o en color
                            if len(img.shape) == 2:  # Imagen en escala de grises
                                gray_img = img  # No es necesario convertirla
                            else:
                                gray_img = color.rgb2gray(img)  # Convertir si es RGB

                            entropies_color[key].append(calculate_entropy(img))
                            entropies_gray[key].append(calculate_entropy(gray_img))

    return entropies_color, entropies_gray


def plot_boxplot(entropies, title):
    data = [(category, value) for category, values in entropies.items() for value in values]
    categories, values = zip(*data)

    plt.figure()
    sns.boxplot(x=categories, y=values)
    plt.title(title)
    plt.xlabel('Categories')
    plt.ylabel('Entropy')
    plt.xticks(rotation=45)
    plt.show()

def save_data_json(entropies_color, entropies_gray, filename="test.json"):
    entropies_color = {key: [float(value) for value in values] for key, values in entropies_color.items()}
    entropies_gray = {key: [float(value) for value in values] for key, values in entropies_gray.items()}
    data = {
        "entropies_color": entropies_color,
        "entropies_gray": entropies_gray,
    }
    with open(filename, "w") as f:
        json.dump(data, f)
    return "Exito al guardar"


def modify_image(image, level):
    modified = image.copy()
    h, w, _ = modified.shape

    if level == "small":
        y, x = random.randint(0, h - 5), random.randint(0, w - 5)
        modified[y:y + 5, x:x + 5] = [255, 0, 0]

    elif level == "medium":
        y, x = random.randint(0, h // 2), random.randint(0, w // 2)
        modified[y:y + 50, x:x + 50] = modified[y + 50:y + 100, x + 50:x + 100]

    elif level == "large":
        modified[h // 4:h // 2, w // 4:w // 2] = util.invert(modified[h // 4:h // 2, w // 4:w // 2])

    return modified


def apply_modifications(root_folder):
    entropies_modified = {"small": {}, "medium": {}, "large": {}}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for dataset in os.listdir(root_folder):
        dataset_path = os.path.join(root_folder, dataset)

        if os.path.isdir(dataset_path):
            for category in os.listdir(dataset_path):
                category_path = os.path.join(dataset_path, category)

                if os.path.isdir(category_path):
                    key = f"{dataset}/{category}"
                    entropies_modified["small"][key] = []
                    entropies_modified["medium"][key] = []
                    entropies_modified["large"][key] = []

                    for img_name in os.listdir(category_path):
                        img_path = os.path.join(category_path, img_name)

                        if os.path.isfile(img_path) and any(img_name.lower().endswith(ext) for ext in valid_extensions):
                            img = io.imread(img_path)

                            for level in ["small", "medium", "large"]:
                                mod_img = modify_image(img, level)
                                entropies_modified[level][key].append(calculate_entropy(mod_img))

    return entropies_modified


def save_data_json(entropies_color, entropies_gray, entropies_modified, filename="test.json"):
    data = {
        "entropies_color": {key: [float(v) for v in values] for key, values in entropies_color.items()},
        "entropies_gray": {key: [float(v) for v in values] for key, values in entropies_gray.items()},
        "entropies_modified": {level: {key: [float(v) for v in values] for key, values in entropies.items()} for
                               level, entropies in entropies_modified.items()}
    }
    with open(filename, "w") as f:
        json.dump(data, f)
    return "Exito al guardar"


def plot_entropy_comparison(entropies_original, entropies_modified):
    levels = ["small", "medium", "large"]
    categories = entropies_original.keys()

    plt.figure(figsize=(12, 6))

    for i, level in enumerate(levels):
        diffs = [
            np.mean(entropies_modified[level][cat]) - np.mean(entropies_original[cat])
            for cat in categories
        ]
        plt.bar(np.arange(len(categories)) + i * 0.25, diffs, width=0.25, label=level)

    plt.xticks(np.arange(len(categories)) + 0.25, categories, rotation=45)
    plt.ylabel("Change in Entropy")
    plt.title("Entropy Change After Modifications")
    plt.legend()
    plt.show()


#unzip_images("images_pr0.zip", "images")
#entropies_color, entropies_gray = process_images("images")
#plot_boxplot(entropies_color, "Entropy color")
#plot_boxplot(entropies_gray, "Entropy gray")
#save_data_json(entropies_color, entropies_gray)


# TEST DE ALTERACIONES EN IMAGENES
# Graficar los resultados
entropies = {"original": [], "small": [], "medium": [], "large": []}
x_labels = [f"Img {i+1}" for i in range(10)]
x = np.arange(len(x_labels))

plt.figure(figsize=(12, 6))
width = 0.2

plt.bar(x - width*1.5, entropies["original"], width, label="Original", color="blue")
plt.bar(x - width/2, entropies["small"], width, label="Pequeña", color="green")
plt.bar(x + width/2, entropies["medium"], width, label="Mediana", color="orange")
plt.bar(x + width*1.5, entropies["large"], width, label="Grande", color="red")

plt.xticks(x, x_labels, rotation=45)
plt.ylabel("Entropía")
plt.title("Cambio de entropía con diferentes modificaciones")
plt.legend()
plt.tight_layout()
plt.show()


##test de modificacionentropies_modified = apply_modifications(root_folder)
root_folder = "images/test"
entropies_color, entropies_gray = process_images(root_folder)
entropies_modified = apply_modifications(root_folder)
save_data_json(entropies_color, entropies_gray, entropies_modified)

# Graficar comparación de entropía antes y después de modificaciones
plot_entropy_comparison(entropies_color, entropies_modified)