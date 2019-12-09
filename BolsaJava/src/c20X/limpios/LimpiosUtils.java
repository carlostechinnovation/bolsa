package c20X.limpios;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LimpiosUtils {

	public static final String DIR_LIMPIOS = "/bolsa/pasado/limpios/";

	/**
	 * Lee un fichero y carga los datos en una lista de listas. Es decir, una lista
	 * de columnas.
	 * 
	 * @param pathFicheroIn
	 * @return
	 * @throws IOException
	 */
	public static List<List<String>> leerFicheroHaciaListasDeColumnas(String pathFicheroIn) throws IOException {

		FileReader fr = new FileReader(pathFicheroIn);
		BufferedReader br = new BufferedReader(fr);
		String actual;
		boolean primeraLinea = true;

		List<List<String>> datos = new ArrayList<List<String>>();

		while ((actual = br.readLine()) != null) {
			if (primeraLinea == false && actual.contains("|")) {

				String[] partes = actual.split("\\|");

				for (int j = 0; j < partes.length; j++) {
					datos.get(j).add(partes[j]);
				}

			} else {
				// La cabecera me dirá el numero de columnas
				String[] partes = actual.split("\\|");
				int numColumnas = partes.length;
				datos = new ArrayList<List<String>>();

				for (int i = 1; i <= numColumnas; i++) {
					datos.add(new ArrayList<String>());
					datos.get(i - 1).add(partes[i - 1]);
				}

			}
			primeraLinea = false;
		}
		br.close();

		return datos;
	}

}
