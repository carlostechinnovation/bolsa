print("--- PRUEBAS ---")
entrada_csv_subgrupo = "/bolsa/futuro/subgrupos/SG_0/"
modoTiempo = "futuro"

if modoTiempo == "futuro":
    entrada_csv_subgrupo = entrada_csv_subgrupo.replace("futuro", "pasado")
    print(entrada_csv_subgrupo)




