package c10X.brutos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

/**
 * Datos de otras fuentes
 *
 */
public class Otros02Parsear {

	static Logger MY_LOGGER = Logger.getLogger(Otros02Parsear.class);

	public Otros02Parsear() {
		super();
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		MY_LOGGER.info("INICIO");

		BasicConfigurator.configure();
		MY_LOGGER.setLevel(Level.INFO);

		if (args.length != 0) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		}

		List<Otros02NasdaqTicker> tickersNasdaq = cargarNasdaqTickers();

		MY_LOGGER.info("FIN");
	}

	/**
	 * @param csvFile
	 * @param delimitador
	 */
	public static List<Otros02NasdaqTicker> cargarNasdaqTickers() {

		String csvFile = "C:\\DATOS\\GITHUB_REPOS\\bolsa\\BolsaJava\\src\\main\\resources\\nasdaq_tickers.csv";
		String delimitador = ",";
		MY_LOGGER.info("Cargando NASDAQ-TICKERS de: " + csvFile);

		List<Otros02NasdaqTicker> out = new ArrayList<Otros02NasdaqTicker>();
		try {
			File file = new File(csvFile);
			FileReader fr = new FileReader(file);
			BufferedReader br = new BufferedReader(fr);

			String line = "";
			String[] tempArr;
			boolean primeraLinea = true;
			String lineaLimpia = "";

			while ((line = br.readLine()) != null) {

				lineaLimpia = line.replace("\"", ""); // limpiar comillas dobles

				if (primeraLinea) {
					primeraLinea = false;

				} else {
					tempArr = lineaLimpia.split(delimitador);
					out.add(new Otros02NasdaqTicker(tempArr[0], tempArr[1], tempArr[2], tempArr[3], tempArr[4],
							tempArr[5], tempArr[6], tempArr[7]));
				}
			}
			br.close();

			MY_LOGGER.info("NASDAQ-TICKERS leidos: " + out.size());

		} catch (IOException ioe) {
			ioe.printStackTrace();
		}

		return out;
	}

}
