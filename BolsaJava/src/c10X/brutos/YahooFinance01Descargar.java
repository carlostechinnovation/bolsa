package c10X.brutos;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

/**
 * Descarga de datos de Yahoo Finance
 *
 */
public class YahooFinance01Descargar {

	static Logger MY_LOGGER = Logger.getLogger(YahooFinance01Descargar.class);

	public YahooFinance01Descargar() {
		super();
	}

	/**
	 * @param args
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public static void main(String[] args) throws IOException, InterruptedException {

		MY_LOGGER.info("INICIO");

		BasicConfigurator.configure();
		MY_LOGGER.setLevel(Level.INFO);

		Integer numMaxEmpresas = BrutosUtils.NUM_EMPRESAS_PRUEBAS; // DEFAULT
		String directorioOut = BrutosUtils.DIR_BRUTOS; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 2) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			numMaxEmpresas = Integer.valueOf(args[0]);
			directorioOut = args[1];
		}

		List<EstaticoNasdaqModelo> nasdaqEstaticos1 = EstaticosNasdaqDescargarYParsear.descargarNasdaqEstaticosSoloLocal1();
		descargarNasdaqDinamicos01(nasdaqEstaticos1, numMaxEmpresas, directorioOut);

		MY_LOGGER.info("FIN");
	}

	/**
	 * NASDAQ - DINAMICOS-1
	 * 
	 * @param nasdaqEstaticos1
	 * @param numMaxEmpresas
	 * @param directorioOut
	 * @return
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public static Boolean descargarNasdaqDinamicos01(List<EstaticoNasdaqModelo> nasdaqEstaticos1,
			Integer numMaxEmpresas, String directorioOut) throws IOException, InterruptedException {

		MY_LOGGER.info("descargarNasdaqDinamicos01 --> " + numMaxEmpresas + "|" + directorioOut);

		String mercado = "NASDAQ"; // DEFAULT
		Boolean out = false;
		String ticker;
		long msegEspera;
		YahooFinance01Descargar instancia = new YahooFinance01Descargar();

		MY_LOGGER.info("mercado=" + mercado);

		for (int i = 0; i < nasdaqEstaticos1.size(); i++) {

			if (i <= numMaxEmpresas) {
				ticker = nasdaqEstaticos1.get(i).symbol;

				String pathOut = directorioOut + BrutosUtils.YAHOOFINANCE + "_" + mercado + "_" + ticker + ".txt";
				String URL_yahoo_ticker = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker + "?symbol="
						+ ticker + "&range=6mo&interval=60m";

				MY_LOGGER.info("pathOut=" + pathOut);
				MY_LOGGER.info("URL_yahoo_ticker=" + URL_yahoo_ticker);

				Files.deleteIfExists(Paths.get(pathOut)); // Borramos el fichero de salida si existe

				// espera aleatoria
				msegEspera = (long) (BrutosUtils.ESPERA_ALEATORIA_MSEG_MIN
						+ Math.random() * 1000 * BrutosUtils.ESPERA_ALEATORIA_SEG_MAX);
				MY_LOGGER.info("Espera aleatoria " + msegEspera + " mseg...");
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

		MY_LOGGER.info("descargarPagina --> " + urlEntrada + " | " + pathOut + " | " + borrarSiExiste);
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
				MY_LOGGER.info("--- Peticion HTTP escapando caracteres espacio en URL ---");
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
			MY_LOGGER.error("Error:" + e.getMessage());
			e.printStackTrace();
		}

		return out;
	}

}
