package c10X.brutos;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

/**
 * Descarga de pruebas de una página de Yahoo Finance
 *
 */
public class YahooFinance01Descargar {

	static Logger MY_LOGGER = Logger.getLogger(YahooFinance01Descargar.class);

	public YahooFinance01Descargar() {
		super();
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		MY_LOGGER.info("INICIO");

		BasicConfigurator.configure();
		MY_LOGGER.setLevel(Level.INFO);

		String mercado = "NASDAQ"; // DEFAULT
		String ticker = "CGIX"; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 2) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			mercado = args[0];
			ticker = args[1];
		}

		MY_LOGGER.info("mercado=" + mercado);
		MY_LOGGER.info("ticker=" + ticker);

		String pathOut = "C:\\bolsa\\pasado\\brutos\\bruto_" + mercado + "_" + ticker + ".txt";
		String URL_yahoo_ticker = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker + "?symbol=" + ticker
				+ "&range=6mo&interval=60m";

		MY_LOGGER.info("pathOut=" + pathOut);
		MY_LOGGER.info("URL_yahoo_ticker=" + URL_yahoo_ticker);

		YahooFinance01Descargar instancia = new YahooFinance01Descargar();
		Boolean out = instancia.descargarPagina(pathOut, true, URL_yahoo_ticker);

		MY_LOGGER.info("Resultado: " + out);
		MY_LOGGER.info("FIN");
	}

	/**
	 * Dada una URL, descarga su contenido de Internet en una carpeta indicada.
	 * 
	 * @param pathOut        Path absoluto del FICHERO (no carpeta) donde se van a
	 *                       GUARDAR los DATOS BRUTOS.
	 * @param borrarSiExiste BOOLEAN que indica si se quiere BORRAR el FICHERO (no
	 *                       la carpeta) de donde se van a guardar los DATOS BRUTOS.
	 * @param urlEntrada     URL de la pÃ¡gina web a descargar
	 */
	public Boolean descargarPagina(String pathOut, Boolean borrarSiExiste, String urlEntrada) {

		MY_LOGGER.info("[URL|pathOut|borrarSiExiste] --> " + urlEntrada + " | " + pathOut + " | " + borrarSiExiste);
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
