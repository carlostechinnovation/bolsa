package c10X.brutos;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

/**
 * Descarga de pruebas de una página de Yahoo Finance
 *
 */
public class YahooFinance03JuntarEnCSVunico {

	static Logger MY_LOGGER = Logger.getLogger(YahooFinance03JuntarEnCSVunico.class);

	public YahooFinance03JuntarEnCSVunico() {
		super();
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.println("Bolsa - YahooFinance - Prueba - INICIO");

		String ficheroPath = "C:\\DATOS\\GITHUB_REPOS\\bolsa\\knime_mockdata\\YahooFinance_prueba.txt";

		YahooFinance03JuntarEnCSVunico instancia = new YahooFinance03JuntarEnCSVunico();
		String contenidoDescargado = instancia.descargarPagina(ficheroPath, true,
				"https://query1.finance.yahoo.com/v8/finance/chart/CGIX?symbol=CGIX&range=6mo&interval=60m");
		// https://finance.yahoo.com/quote/CGIX/chart?p=CGIX&_guc_consent_skip=1575197868

		instancia.parsearJson(ficheroPath);

		System.out.println("Bolsa - YahooFinance - Prueba - FIN");
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
	public String descargarPagina(String pathOut, Boolean borrarSiExiste, String urlEntrada) {

		MY_LOGGER.info("[URL|pathOut|borrarSiExiste] --> " + urlEntrada + " | " + pathOut + " | " + borrarSiExiste);
		String contenido = null;

		try {
			if (Files.exists(Paths.get(pathOut)) && borrarSiExiste) {
				MY_LOGGER.debug("Borrando fichero de salida preexistente...");
				Files.delete(Paths.get(pathOut));
			}

			MY_LOGGER.info("--- Peticion HTTP normal ---");
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
			contenido = content.toString();

			Files.write(Paths.get(pathOut), content.toString().getBytes());

		} catch (IOException e) {
			MY_LOGGER.error("Error:" + e.getMessage());
			e.printStackTrace();
		}

		return contenido;

	}

	/**
	 * @param pathJsonEntrada
	 */
	public void parsearJson(String pathJsonEntrada) {

		MY_LOGGER.info("Parseando JSON...");

		JSONParser parser = new JSONParser();

		try (Reader reader = new FileReader(pathJsonEntrada)) {

			JSONObject primerJson = (JSONObject) parser.parse(reader);

			Map<String, JSONObject> mapaChart = (HashMap<String, JSONObject>) primerJson.get("chart");
			Object resultValor = mapaChart.get("result");
			JSONArray a1 = (JSONArray) resultValor;
			JSONObject a2 = (JSONObject) a1.get(0);
			Set<String> claves = a2.keySet();
			for (String clave : claves) {
				System.out.println(clave);
			}

			Object meta = a2.get("meta");
			Object indicators = a2.get("indicators");
			Object tiempos = a2.get("timestamp");

			System.out.println("meta ---> " + meta);
			System.out.println("indicators ---> " + indicators);
			System.out.println("tiempos ---> " + tiempos);

//			// loop array
//			JSONArray msg = (JSONArray) jsonObject.get("messages");
//			Iterator<String> iterator = msg.iterator();
//			while (iterator.hasNext()) {
//				System.out.println(iterator.next());
//			}

		} catch (IOException e) {
			System.err.println(e.getMessage());
		} catch (ParseException e) {
			System.err.println(e.getMessage());
		}

	}

}
