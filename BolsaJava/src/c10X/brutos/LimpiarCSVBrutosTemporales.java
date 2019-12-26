package c10X.brutos;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

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
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(PatternLayout.TTCC_CONVERSION_PATTERN)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

		nucleo();
		MY_LOGGER.info("FIN");

	}

	/**
	 * @throws IOException
	 */
	public static void nucleo() throws IOException {

		File dirBrutoCsv = new File(BrutosUtils.DIR_BRUTOS_CSV);
		String[] listaBorrables = dirBrutoCsv.list(new FilenameFilter() {

			public boolean accept(File dir, String name) {

				boolean temporalNasdaqOld = name.startsWith(BrutosUtils.NASDAQOLD + "_") && name.endsWith(".csv");
				boolean temporalYahoo = name.startsWith(BrutosUtils.YAHOOFINANCE + "_") && name.endsWith(".csv");
				boolean temporalFinviz = name.startsWith(BrutosUtils.FINVIZ + "_") && name.endsWith(".csv");

				if (temporalNasdaqOld || temporalYahoo || temporalFinviz) {
					return true;
				}
				return false;
			}
		});

		for (String pathBorrable : listaBorrables) {
			MY_LOGGER.info("Borrando temporal: " + BrutosUtils.DIR_BRUTOS_CSV + pathBorrable);
			Files.deleteIfExists(Paths.get(BrutosUtils.DIR_BRUTOS_CSV + pathBorrable));
		}

		// VELAS DE NASDAQ
		String velasNasdaq = BrutosUtils.DIR_BRUTOS_CSV + "VELAS_" + BrutosUtils.MERCADO_NQ + ".csv";
		Files.deleteIfExists(Paths.get(velasNasdaq));
	}

}
