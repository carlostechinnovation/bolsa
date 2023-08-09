package c20X.limpios;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LimpiosUtils {

	public static final String DIR_LIMPIOS = "/bolsa/futuro/limpios/";
	public static final String P_INICIO = "20001111"; // default
	public static final String P_FIN = "20991111"; // default
	public static final String SEPARADOR = "|";

	/**
	 * Lee un fichero y carga los datos en una lista de listas. Es decir, una lista
	 * de columnas.
	 * 
	 * @param pathFicheroIn
	 * @param numMaxLineasLeidas (opcional) Número máximo de líneas leídas del
	 *                           fichero de entrada. Si no se indica, se leerá el
	 *                           fichero hasta el final.
	 * @return
	 * @throws IOException
	 */
	public static List<List<String>> leerFicheroHaciaListasDeColumnas(String pathFicheroIn, Long numMaxLineasLeidas)
			throws IOException {

		FileReader fr = new FileReader(pathFicheroIn);
		BufferedReader br = new BufferedReader(fr);
		String actual;
		boolean primeraLinea = true;

		List<List<String>> datos = new ArrayList<List<String>>();
		long numfila = 0;

		while ((actual = br.readLine()) != null) {

			numfila++;
			if (numMaxLineasLeidas == null || numfila <= numMaxLineasLeidas) {

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

		}
		br.close();

		return datos;
	}

}
