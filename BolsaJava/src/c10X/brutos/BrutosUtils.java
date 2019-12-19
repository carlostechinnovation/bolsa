package c10X.brutos;

import java.io.File;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;
import java.util.Locale;

public class BrutosUtils {

	public static final String DIR_BRUTOS = "/bolsa/pasado/brutos/";
	public static final String DIR_BRUTOS_CSV = "/bolsa/pasado/brutos_csv/";
	public static final String NASDAQ_TICKERS_CSV = "src/main/resources/nasdaq_tickers.csv";
	public static final String DESCONOCIDOS_CSV = "src/main/resources/desconocidos.csv";

	public static final String NASDAQOLD = "NQO";
	public static final String YAHOOFINANCE = "YF";
	public static final String FINVIZ = "FZ";
	public static final String BRUTO_FINAL = "BR"; // Prefijo del fichero final

	public static final String MERCADO_NQ = "NASDAQ";

	public static final int ESPERA_ALEATORIA_MSEG_MIN = 200;
	public static final int ESPERA_ALEATORIA_SEG_MAX = 2;
	public static final int NUM_EMPRESAS_PRUEBAS = 60;

	public static final String NULO = "null";
	public static final String ESCALA_UNO = "uno";
	public static final String ESCALA_M = "millones";

	/**
	 * @param in
	 * @param escala Numero en escala "uno" o en "millones"
	 * @return
	 */
	public static String tratamientoLigero(String in, String escala) {

		String out = NULO;

		Locale locale = new Locale("en", "UK");
		DecimalFormat df = (DecimalFormat) NumberFormat.getNumberInstance(locale);
		df.applyPattern("#0.######");

		if (in != null) {
			out = in.trim();

			if (!out.isEmpty() && !out.equals("-") && !out.equals("--") && !out.equals("N/A")) {

				// TIENE DATO. Entonces lo trato

				if (in.contains("%")) {
					out = in.replace("%", "").trim();
				}

				if (in.contains(".")) {
					// NUMERICO CON DECIMALES

					Float numero;

					// Lo convierto a tanto por uno siempre
					if (in.contains("K")) {
						out = in.replace("K", "").trim();
						numero = (Float.valueOf(out) * 1000F);
					} else if (in.contains("M")) {
						out = in.replace("M", "").trim();
						numero = Float.valueOf(out) * 1000000F;
					} else if (in.contains("B")) {
						out = in.replace("B", "").trim();
						numero = (Float.valueOf(out) * 1000000000F);
					} else {
						numero = Float.valueOf(out);
					}

					// Si se necesita, se escala y se lleva a salida OUT
					if (escala != null && escala.equals(ESCALA_M)) {
						numero = Float.valueOf(numero) / 1000000F;
						out = df.format(numero);
					} else {
						out = df.format(numero);
					}

				}

			}
		}

		return out;
	}

	/**
	 * Dada una carpeta, la recorre (y sus subcarpetas) acumulando los ficheros que
	 * cumplen un patron.
	 * 
	 * @param pattern PATRON
	 * @param folder  Carpeta a rastrear
	 * @param result  Lista de ficheros encontrados
	 */
	public static void encontrarFicherosEnCarpeta(final String pattern, final File folder, List<String> result) {
		for (final File f : folder.listFiles()) {

			if (f.isDirectory()) {
				encontrarFicherosEnCarpeta(pattern, f, result);
			}

			if (f.isFile()) {
				if (f.getName().matches(pattern)) {
					result.add(f.getAbsolutePath());
				}
			}

		}
	}

}
