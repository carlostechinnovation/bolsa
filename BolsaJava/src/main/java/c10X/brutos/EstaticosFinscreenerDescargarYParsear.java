package c10X.brutos;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import coordinador.Principal;

/**
 * Datos DINAMICOS sobre EARNINGS (pasados o futuros) de página
 * www.finscreener.com
 *
 */
public class EstaticosFinscreenerDescargarYParsear {

	static Logger MY_LOGGER = Logger.getLogger(EstaticosFinscreenerDescargarYParsear.class);

	private static EstaticosFinscreenerDescargarYParsear instancia = null;

	private EstaticosFinscreenerDescargarYParsear() {
		super();
	}

	public static EstaticosFinscreenerDescargarYParsear getInstance() {
		if (instancia == null)
			instancia = new EstaticosFinscreenerDescargarYParsear();

		return instancia;
	}

	/**
	 * @param args
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public static void main(String[] args) throws IOException, InterruptedException {

		Object appendersAcumulados = Logger.getRootLogger().getAllAppenders();
		if (appendersAcumulados instanceof NullEnumeration) {
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(Principal.LOG_PATRON)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		}

		// PENDIENTE
		// descargarPaginaFinscreenerPasados("2020-06-08", "2020-06-30",
		// "/bolsa/pasado/earningsdates.html");

		MY_LOGGER.info("FIN");
	}

	/**
	 * Descarga datos PASADOS sobre FECHAS DE EARNINGS
	 * 
	 * @param pathOut
	 * @param fechaInicio Ej. 2020-06-08
	 * @param fechaFin    Ej. 2020-06-30
	 * @return
	 */
	public static Boolean descargarPaginaFinscreenerPasados(String fechaInicio, String fechaFin, String pathOut) {

		MY_LOGGER.debug("descargarPaginaFinscreener --> " + fechaInicio + " | " + fechaInicio + " | " + fechaFin);
		Boolean out = false;

		try {
			if (Files.exists(Paths.get(pathOut))) {
				MY_LOGGER.debug("Borrando fichero de salida preexistente...");
				Files.delete(Paths.get(pathOut));
			}

			// MY_LOGGER.info("--- Peticion HTTP normal ---");
			// Request
			URL url = new URL("https://www.finscreener.com/earnings/earnings-reported");
			HttpURLConnection.setFollowRedirects(true);
			HttpURLConnection con = (HttpURLConnection) url.openConnection();
			con.setRequestMethod("GET");
			// con.setDoOutput(true); // Conexion usada para output

			// Request Headers
//			con.setRequestProperty("Content-Type","text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8");

//			con.setRequestProperty("Host", "www.finscreener.com");
//			con.setRequestProperty("User-Agent",
//					"Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:64.0) Gecko/20100101 Firefox/64.0");
			con.setRequestProperty("Accept", "text/html, */*; q=0.01");
			con.setRequestProperty("Accept-Language", "en-US,en;q=0.5");
			con.setRequestProperty("Accept-Encoding", "gzip, deflate, br");
//			con.setRequestProperty("Referer", "https://www.google.com");
			con.setRequestProperty("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8");
//			con.setRequestProperty("X-Requested-With", "XMLHttpRequest");
//			con.setRequestProperty("Content-Length", "38709");
//			con.setRequestProperty("Connection", "keep-alive");
			con.setRequestProperty("Cookie", "");

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

			BufferedReader in = null;
			// El contenido viene comprimido en GZIP
			if (con.getHeaderField("Content-Encoding") != null
					&& con.getHeaderField("Content-Encoding").equals("gzip")) {
				in = new BufferedReader(new InputStreamReader(new GZIPInputStream(con.getInputStream())));
			} else {
				in = new BufferedReader(new InputStreamReader(con.getInputStream()));
			}

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
