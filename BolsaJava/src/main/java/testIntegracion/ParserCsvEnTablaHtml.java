/**
 * 
 */
package testIntegracion;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

/**
 * Lee un fichero CSV, lo convierte en una tabla HTML y la escribe en fichero
 * HTML.
 *
 */
public class ParserCsvEnTablaHtml {

	private static final long serialVersionUID = 1L;

	static Logger MY_LOGGER = Logger.getLogger(ExtractorFeatures.class);
	private static ParserCsvEnTablaHtml instancia = null;

	private ParserCsvEnTablaHtml() {
		super();
	}

	public static ParserCsvEnTablaHtml getInstance() {
		if (instancia == null)
			instancia = new ParserCsvEnTablaHtml();
		return instancia;
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {

		Object appendersAcumulados = Logger.getRootLogger().getAllAppenders();
		if (appendersAcumulados instanceof NullEnumeration) {
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(PatternLayout.TTCC_CONVERSION_PATTERN)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

		String pathEntrada = "/tmp/entrada.csv"; // DEFAULT
		String pathSalida = "/tmp/salida.html"; // DEFAULT
		String separadorCsv = "|"; // DEFAULT
		String modo = "append"; // append o nuevo

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 4) {
			MY_LOGGER.error("Parametros de entrada incorrectos (se estan recibiendo " + args.length + "parametros)!!");
			System.exit(-1);
		} else {
			pathEntrada = args[0];
			pathSalida = args[1];
			separadorCsv = args[2];
			modo = args[3];

			MY_LOGGER.info("pathEntrada:" + pathEntrada);
			MY_LOGGER.info("pathSalida:" + pathSalida);
			MY_LOGGER.info("separadorCsv:" + separadorCsv);
			MY_LOGGER.info("modo:" + modo);
		}

		String cadenaHtml = nucleo(pathEntrada, separadorCsv, modo);

		// ---------- ESCRITURA --------------
		MY_LOGGER.info("Escribiendo en: " + pathSalida);
		BufferedWriter writer = new BufferedWriter(new FileWriter(pathSalida, modo.equalsIgnoreCase("append")));
		writer.write(cadenaHtml);
		writer.close();

		MY_LOGGER.info("FIN");
	}

	/**
	 * @param pathEntrada
	 * @param separadorCsv
	 * @param modo
	 * @throws IOException
	 */
	public static String nucleo(String pathEntrada, String separadorCsv, String modo) throws IOException {

		String tabla = "<table>";

		List<String> lineas = Files.readAllLines(Paths.get(pathEntrada));
		int contador = 0;
		for (String linea : lineas) {
			if (contador == 0) {
				tabla += parsearFilaCsvEnHtml(linea, separadorCsv, "th");
			} else {
				tabla += parsearFilaCsvEnHtml(linea, separadorCsv, "td");
			}
			contador++;
		}

		tabla += "</table>";
		return tabla;
	}

	/**
	 * @param in
	 * @param sepEntrada
	 * @param sepHtml
	 * @return
	 */
	public static String parsearFilaCsvEnHtml(String in, String sepEntrada, String sepHtml) {
		String out = "<tr>";
		String[] partes = in.split(sepEntrada);
		for (String parte : partes) {
			out += "<" + sepHtml + ">" + parte + "</" + sepHtml + ">";
		}
		out += "</tr>";
		return out;
	}

}
