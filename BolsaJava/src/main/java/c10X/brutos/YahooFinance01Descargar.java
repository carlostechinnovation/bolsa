package c10X.brutos;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import coordinador.Principal;

/**
 * Descarga de datos de Yahoo Finance
 *
 */
public class YahooFinance01Descargar implements Serializable {

	static Logger MY_LOGGER = Logger.getLogger(YahooFinance01Descargar.class);

	private static YahooFinance01Descargar instancia = null;

	private YahooFinance01Descargar() {
		super();
	}

	public static YahooFinance01Descargar getInstance() {
		if (instancia == null)
			instancia = new YahooFinance01Descargar();

		return instancia;
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		Object appendersAcumulados = Logger.getRootLogger().getAllAppenders();
		if (appendersAcumulados instanceof NullEnumeration) {
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(Principal.LOG_PATRON)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

		Integer numMaxEmpresas = BrutosUtils.NUM_EMPRESAS_PRUEBAS; // DEFAULT
		String directorioOut = BrutosUtils.DIR_BRUTOS; // DEFAULT
		String modo = BrutosUtils.PASADO; // DEFAULT
		String rango = BrutosUtils.RANGO_YF_1Y; // DEFAULT
		String velaYF = BrutosUtils.VELA_YF_1D; // DEFAULT
		Integer entornoDeValidacion = BrutosUtils.ES_ENTORNO_VALIDACION;// DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");

		} else if (args.length != 6) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			int numParams = args.length;
			MY_LOGGER.info("Numero de parametros: " + numParams);
			for (String param : args) {
				MY_LOGGER.info("Param: " + param);
			}
			System.exit(-1);

		} else {
			numMaxEmpresas = Integer.valueOf(args[0]);
			directorioOut = args[1];
			modo = args[2];
			rango = args[3];
			velaYF = args[4];
			entornoDeValidacion = Integer.valueOf(args[5]);
			MY_LOGGER.info("Parametros de entrada -> " + numMaxEmpresas + " | " + directorioOut + " | " + modo + "|"
					+ rango + "|" + velaYF + "|" + entornoDeValidacion);
		}

		EstaticosNasdaqDescargarYParsear.getInstance();
		List<EstaticoNasdaqModelo> nasdaqEstaticos1 = EstaticosNasdaqDescargarYParsear
				.descargarNasdaqEstaticosSoloLocal1(entornoDeValidacion);
		descargarNasdaqDinamicos01(nasdaqEstaticos1, numMaxEmpresas, directorioOut, modo, rango, velaYF);

		MY_LOGGER.info("FIN");
	}

	/**
	 * NASDAQ - DINAMICOS-1
	 * 
	 * @param nasdaqEstaticos1
	 * @param numMaxEmpresas
	 * @param directorioOut
	 * @param modo             pasado o futuro
	 * @param rango            6mo (6 meses), 1y (1 año)...
	 * @param velaYF
	 * @return
	 * @throws Exception
	 */
	public static Boolean descargarNasdaqDinamicos01(List<EstaticoNasdaqModelo> nasdaqEstaticos1,
			Integer numMaxEmpresas, String directorioOut, String modo, String rango, String velaYF) throws Exception {

		MY_LOGGER.info("descargarNasdaqDinamicos01 --> " + numMaxEmpresas + "|" + directorioOut);

		String mercado = "NASDAQ"; // DEFAULT
		Boolean out = false;
		String ticker;
		long msegEspera;
		YahooFinance01Descargar instancia = new YahooFinance01Descargar();

		MY_LOGGER.info("mercado=" + mercado);

		for (int i = 0; i < nasdaqEstaticos1.size(); i++) {

			if (i < numMaxEmpresas) {
				ticker = nasdaqEstaticos1.get(i).symbol;

				String pathOut = directorioOut + BrutosUtils.YAHOOFINANCE + "_" + mercado + "_" + ticker + ".txt";
				String URL_yahoo_ticker = getUrlYahooFinance(ticker, modo, rango, velaYF);

				if (i % 10 == 1) {
					MY_LOGGER.info("Empresa numero = " + (i + 1) + " (" + ticker + ")");
				}
				MY_LOGGER.debug("pathOut=" + pathOut);
				MY_LOGGER.debug("URL_yahoo_ticker=" + URL_yahoo_ticker);

				Files.deleteIfExists(Paths.get(pathOut)); // Borramos el fichero de salida si existe

				// espera aleatoria
				msegEspera = (long) (BrutosUtils.ESPERA_ALEATORIA_MSEG_MIN
						+ Math.random() * 1000 * BrutosUtils.ESPERA_ALEATORIA_SEG_MAX);
				MY_LOGGER.debug("Espera aleatoria " + msegEspera + " mseg...");
				Thread.sleep(msegEspera);

				out = instancia.descargarPagina(pathOut, true, URL_yahoo_ticker);
				if (out.booleanValue() == false) {
					MY_LOGGER.error("La descarga de datos estaticos 1 de " + mercado + " - " + ticker
							+ " ha fallado. Saliendo...");
				}

			} else {
				break;
			}
		}

		return out;
	}

	/**
	 * Dada una URL, descarga su contenido de Internet en una carpeta indicada.
	 * 
	 * @param pathOut        Path absoluto del FICHERO (no carpeta) donde se van a
	 *                       GUARDAR los DATOS BRUTOS.
	 * @param borrarSiExiste BOOLEAN que indica si se quiere BORRAR el FICHERO (no
	 *                       la carpeta) de donde se van a guardar los DATOS BRUTOS.
	 * @param urlEntrada     URL de la pagina web a descargar
	 */
	public static Boolean descargarPagina(String pathOut, Boolean borrarSiExiste, String urlEntrada) {

		// MY_LOGGER.info("descargarPagina --> " + urlEntrada + " | " + pathOut + " | "
		// + borrarSiExiste);
		Boolean out = false;

		try {
			if (Files.exists(Paths.get(pathOut)) && borrarSiExiste) {
				MY_LOGGER.debug("Borrando fichero de salida preexistente...");
				Files.delete(Paths.get(pathOut));
			}

			// MY_LOGGER.info("--- Peticion HTTP normal ---");
			// Request
			URL url = new URL(urlEntrada);
			HttpURLConnection.setFollowRedirects(true);
			HttpURLConnection con = (HttpURLConnection) url.openConnection();
			con.setRequestMethod("GET");
			// con.setDoOutput(true); // Conexion usada para output

			// Request Headers
//			con.setRequestProperty("Content-Type",
//					"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8");

			// TIMEOUTs
//			con.setConnectTimeout(5000);
//			con.setReadTimeout(5000);

			// Handling Redirects
//			con.setInstanceFollowRedirects(false);

			// MY_LOGGER.info("--- Peticion HTTP con REDIRECTS ---");
			// HttpURLConnection.setFollowRedirects(true);
			// con = (HttpURLConnection) url.openConnection();
			// con.connect();// COMUNICACION

			// CODIGO de RESPUESTA
			int status = con.getResponseCode();
			if (status == HttpURLConnection.HTTP_MOVED_TEMP || status == HttpURLConnection.HTTP_MOVED_PERM) {
				MY_LOGGER.debug("--- Peticion HTTP escapando caracteres espacio en URL ---");
				String location = con.getHeaderField("Location");
				URL newUrl = new URL(location);
				con = (HttpURLConnection) newUrl.openConnection();
			}

			BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
			String inputLine;
			StringBuffer content = new StringBuffer();
			while ((inputLine = in.readLine()) != null) {
				content.append(inputLine);
			}
			in.close();

			// close the connection
			con.disconnect();

			// Escribir SALIDA
			boolean escribible = Files.isWritable(Paths.get(pathOut));
			MY_LOGGER.debug("Carpeta escribible:" + escribible);
			MY_LOGGER.debug("Escribiendo a fichero...");
			MY_LOGGER.debug("StringBuffer con " + content.length() + " elementos de 16-bits)");
			// contenido = content.toString();

			Files.write(Paths.get(pathOut), content.toString().getBytes());

			out = true;

		} catch (IOException e) {
			MY_LOGGER.error("descargarPagina() - ERROR: " + e.getMessage());
			// e.printStackTrace();
		}

		return out;
	}

	/**
	 * @param ticker
	 * @param modo
	 * @param rango  6mo (6 meses), 1y (1 año)...
	 * @param velaYF
	 * @return
	 * @throws Exception
	 */
	public static String getUrlYahooFinance(String ticker, String modo, String rango, String velaYF) throws Exception {
		String url = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker + "?symbol=" + ticker;

		// Vela 5 minutos
//		if (modo.equals(BrutosUtils.PASADO)) {
//			url += "&range=1mo&interval=5m";
//		} else if (modo.equals(BrutosUtils.FUTURO)) {
//			url += "&range=1mo&interval=5m";
//		} else {
//			throw new Exception("Modo pasado/futuro no explicito. Saliendo...");
//		}

		// Vela 30 minutos
//		if (modo.equals(BrutosUtils.PASADO)) {
//			url += "&range=1mo&interval=30m";
//		} else if (modo.equals(BrutosUtils.FUTURO)) {
//			url += "&range=1mo&interval=30m";
//		} else {
//			throw new Exception("Modo pasado/futuro no explicito. Saliendo...");
//		}

		// Vela 1 día
		if (modo.equals(BrutosUtils.PASADO)) {
			url += "&range=" + rango + "&interval=" + velaYF;
		} else if (modo.equals(BrutosUtils.FUTURO)) {
			url += "&range=" + rango + "&interval=" + velaYF;
		} else {
			throw new Exception("Modo pasado/futuro no explicito. Saliendo...");
		}
		return url;

	}

}
