package c10X.brutos;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import coordinador.Principal;

/**
 * Juntar ESTATICOS + DINAMICOS
 *
 */
public class LimpiarCSVBrutosTemporales {

	static Logger MY_LOGGER = Logger.getLogger(LimpiarCSVBrutosTemporales.class);

	private static LimpiarCSVBrutosTemporales instancia = null;

	private LimpiarCSVBrutosTemporales() {
		super();
	}

	public static LimpiarCSVBrutosTemporales getInstance() {
		if (instancia == null)
			instancia = new LimpiarCSVBrutosTemporales();

		return instancia;
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {

		Object appendersAcumulados = Logger.getRootLogger().getAllAppenders();
		if (appendersAcumulados instanceof NullEnumeration) {
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(Principal.LOG_PATRON)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

		String dirBrutoCsv = BrutosUtils.DIR_BRUTOS_CSV; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 1) {
			MY_LOGGER.error("Parametros de entrada incorrectos!! --> " + args.length);
			int numParams = args.length;
			MY_LOGGER.error("Numero de parametros: " + numParams);
			for (String param : args) {
				MY_LOGGER.error("Param: " + param);
			}

			System.exit(-1);

		} else {
			dirBrutoCsv = args[0];
			MY_LOGGER.info("PARAMS -> " + dirBrutoCsv);
		}
		nucleo(dirBrutoCsv);

		MY_LOGGER.info("FIN");

	}

	/**
	 * @param dirBrutoCsv
	 * @throws IOException
	 */
	public static void nucleo(String dirBrutoCsv) throws IOException {

		String dirBorrables = dirBrutoCsv.endsWith("/") ? (dirBrutoCsv.substring(0, dirBrutoCsv.length() - 1) + "_OLD")
				: (dirBrutoCsv + "_OLD");
		if (Files.exists(Paths.get(dirBorrables))) {
			FileUtils.cleanDirectory(new File(dirBorrables)); // borra posible contenido previo
			Files.deleteIfExists(Paths.get(dirBorrables)); // borra directorio
		}
		Files.createDirectory(Paths.get(dirBorrables));// crea directorio vacio

		File dirBrutoCsvFile = new File(dirBrutoCsv);
		String[] listaBorrables = dirBrutoCsvFile.list(new FilenameFilter() {

			@Override
			public boolean accept(File dir, String name) {

				boolean temporalNasdaqOld = name.startsWith(BrutosUtils.NASDAQOLD + "_") && name.endsWith(".csv");
				boolean temporalYahoo = name.startsWith(BrutosUtils.YAHOOFINANCE + "_") && name.endsWith(".csv");
				boolean temporalFinviz = name.startsWith(BrutosUtils.FINVIZ_ESTATICOS + "_") && name.endsWith(".csv");
				boolean temporalFinvizInsiders = name.startsWith(BrutosUtils.FINVIZ_INSIDERS + "_")
						&& name.endsWith(".csv");
				boolean temporalFinvizNoticias = name.startsWith(BrutosUtils.FINVIZ_NOTICIAS + "_")
						&& name.endsWith(".csv");

				if (temporalNasdaqOld || temporalYahoo || temporalFinviz || temporalFinvizInsiders
						|| temporalFinvizNoticias) {
					return true;
				}
				return false;
			}
		});

		MY_LOGGER.info("Borrado lógico de ficheros CSV temporales. Movemos todos desde: " + dirBrutoCsv + "  hasta "
				+ dirBorrables + "/");
		for (String pathBorrable : listaBorrables) {
			MY_LOGGER.debug("Moviendo fichero temporal desde: " + dirBrutoCsv + pathBorrable + "  hasta " + dirBorrables
					+ "/" + pathBorrable);
			Files.move(Paths.get(dirBrutoCsv + pathBorrable), Paths.get(dirBorrables + "/" + pathBorrable));
		}

	}

}
