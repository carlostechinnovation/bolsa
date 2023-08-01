import os


current_directory = os.getcwd()
print("Directorio actual donde se ejecuta Python:", current_directory)

root_dir = "../docs/"  # Change this to the path of your root folder
print("Directorio DOCS: root_dir=" + root_dir)

def generate_index():
    def process_folder(folder_path):
        folder_name = os.path.basename(folder_path)
        return f"<li><b>{folder_name}</b><ul>{process_files(folder_path)}</ul></li>"

    def process_files(folder_path):
        files_html = ""
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file.endswith(".html") and "pasado" not in os.path.abspath(file_path):
                files_html += f"<li><a href=\"{file_path}\" target=\"_blank\">{file}</a></li>"
        return files_html

    index_html = f"<!DOCTYPE html><html><head><title>Index</title></head><body>\n"
    index_html += "<h1>BOLSA ML - Entregables</h1><ul>\n"

    index_html += "<p>Proyecto: Predictor de subidas en bolsa NASDAQ</p>\n"
    index_html += "<p>Release: 1.0.0-stable</p>\n"
    index_html += "<p>Descripción: es un proyecto piloto para aprender tecnologías. No es rentable porque la Bolsa no es predecible con las variables disponibles.</p>\n"
    index_html += "<h3>Entrenamiento (pasado)</h3>\n"

    index_html += "<h3>Predicciones (futuro)</h3>\n"
    listaFicheros = os.listdir(root_dir)
    listaFicheros.sort()
    for folder in listaFicheros:
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path) and "pasado" not in os.path.abspath(folder_path):
            index_html += process_folder(folder_path)

    index_html += "</ul></body></html>"

    with open(os.path.join(root_dir, "index.html"), "w") as f:
        f.write(index_html)

generate_index()

