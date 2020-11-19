package c10X.brutos;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;

/**
 *
 */
public class InvertirFilasDeFichero {

	public static final String NASDAQ_TICKERS_CSV = "src/main/resources/nasdaq_tickers_v2.csv";
	public static final String SALIDA = "src/main/resources/nasdaq_tickers_invertidos_v2.csv";

	public static void main(String[] args) throws IOException {
		invertir(NASDAQ_TICKERS_CSV, SALIDA, Boolean.TRUE);
	}

	/**
	 * @param pathEntrada
	 * @param pathSalida
	 * @param hayCabecera
	 * @throws IOException
	 */
	public static void invertir(final String pathEntrada, final String pathSalida, final Boolean hayCabecera)
			throws IOException {
		List<String> values = Files.readAllLines(Paths.get(pathEntrada));
		FileWriter writer = new FileWriter(pathSalida);

		String cabecera = "";
		if (hayCabecera) {
			cabecera = values.get(0);
			values.remove(0);
		}

		Collections.reverse(values); // Leer las filas de final a inicio, sin ordenar alfabeticamente

		if (hayCabecera) {
			writer.write(cabecera + System.lineSeparator());
		}
		for (String str : values) {
			writer.write(str + System.lineSeparator());
		}
		writer.close();
	}

}
