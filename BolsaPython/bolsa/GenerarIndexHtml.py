import os


# Function to calculate the root_dir
def calculate_root_dir():
    print("Calculando directorio DOCS...")
    current_directory = os.getcwd()
    print("Directorio actual donde se ejecuta Python:", current_directory)
    tipo1 = current_directory+"/../../docs/"
    tipo2 = "./"
    if os.path.exists(tipo1):
        print("calculate_root_dir()-1")
        root_dir = tipo1

    else:
        print("calculate_root_dir()-2")
        root_dir = tipo2

    print("Directorio DOCS: root_dir=" + root_dir)
    return root_dir


def generate_index():
    root_dir = calculate_root_dir()

    def process_folder(folder_path):
        folder_name = os.path.basename(folder_path)
        return f"<li><b>{folder_name}</b><ul>{process_files(folder_path)}</ul></li>"

    def process_files(folder_path):
        # print("Procesando ficheros del folder " + folder_path + " ...")
        files_html = ""
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file.endswith(".html"):
                print("files_html -->" + files_html)
                files_html += f"<li><a href=\"{file_path}\" target=\"_blank\">{file}</a></li>"
            elif os.path.isdir(file_path):
                files_html += process_folder(file_path)
        return files_html

    index_html = f"<!DOCTYPE html><html><head><title>BOLSA ML - Entregables</title></head><body>\n"
    index_html += "<h1>BOLSA ML - Entregables</h1><ul>\n"

    index_html += "<p>Proyecto: Predictor de subidas en bolsa NASDAQ</p>\n"
    index_html += "<p>Release: 1.0.0-stable</p>\n"
    index_html += "<p>Descripción: es un proyecto piloto para aprender tecnologías. No es rentable porque la Bolsa no es predecible con las variables disponibles.</p>\n"

    directorios = os.listdir(root_dir)
    directorios.sort()

    index_html += "<h3>ENTRENAMIENTO (pasado):</h3>\n"
    for folder in directorios:
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path) and "pasado" in os.path.abspath(folder_path):
            index_html += process_folder(folder_path)

    index_html += "<h3>PREDICCIONES (futuro):</h3>\n"

    for folder in directorios:
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path) and "pasado" not in os.path.abspath(folder_path):
            index_html += process_folder(folder_path)

    index_html += "</ul></body></html>"

    pathSalida = os.path.join(root_dir, "index.html")
    with open(pathSalida, "w") as f:
        print("Fichero SALIDA: " + pathSalida)
        f.write(index_html)


generate_index()
