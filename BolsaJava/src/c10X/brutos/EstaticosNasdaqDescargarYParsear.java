package c10X.brutos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

/**
 * Datos ESTATICOS
 *
 */
public class EstaticosNasdaqDescargarYParsear {

	static Logger MY_LOGGER = Logger.getLogger(EstaticosNasdaqDescargarYParsear.class);

	static String MERCADO_NQ = "NASDAQ";
	static String ID_SRL = "stockreportslink";

	public EstaticosNasdaqDescargarYParsear() {
		super();
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {

		MY_LOGGER.info("INICIO");

		BasicConfigurator.configure();
		MY_LOGGER.setLevel(Level.INFO);

		Integer numMaxEmpresas = 2; // DEFAULT
		String dirBruto = "/bolsa/pasado/brutos/"; // DEFAULT
		String dirBrutoCsv = "/bolsa/pasado/brutos_csv/"; // DEFAULT

		if (args.length != 0) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		}

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 3) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			numMaxEmpresas = Integer.valueOf(args[0]);
			dirBruto = args[1];
			dirBrutoCsv = args[2];
		}

		List<EstaticoNasdaqModelo> nasdaqEstaticos1 = descargarNasdaqEstaticos1();
		descargarYparsearNasdaqEstaticos2(nasdaqEstaticos1, dirBruto, dirBrutoCsv, numMaxEmpresas);

		MY_LOGGER.info("FIN");
	}

	/**
	 * NASDAQ - ESTATICOS-1: datos cargados desde un fichero local.
	 * 
	 * @return Lista de empresas del NASDAQ con algunos datos ESTATICOS
	 */
	public static List<EstaticoNasdaqModelo> descargarNasdaqEstaticos1() {

		MY_LOGGER.info("descargarNasdaqEstaticos1...");
		String csvFile = "C:\\DATOS\\GITHUB_REPOS\\bolsa\\BolsaJava\\src\\main\\resources\\nasdaq_tickers.csv";
		MY_LOGGER.info("Cargando NASDAQ-TICKERS de: " + csvFile);

		List<EstaticoNasdaqModelo> out = new ArrayList<EstaticoNasdaqModelo>();
		try {
			File file = new File(csvFile);
			FileReader fr = new FileReader(file);
			BufferedReader br = new BufferedReader(fr);

			String line = "";
			String[] tempArr;
			boolean primeraLinea = true;
			String lineaLimpia = "";

			while ((line = br.readLine()) != null) {

				lineaLimpia = line.replace("|", ""); // quitamos posibles apariciones del nuevo separador
				lineaLimpia = lineaLimpia.replace("\",\"", "|"); // cambiarSeparadores
				lineaLimpia = lineaLimpia.replace("\"", ""); // limpiar comillas dobles
				lineaLimpia = lineaLimpia.replace(",", ""); // limpiar comas
				lineaLimpia = lineaLimpia.replace("\'", ""); // limpiar comillas simples

				if (primeraLinea) {
					primeraLinea = false;

				} else {
					tempArr = lineaLimpia.split("\\|");
					out.add(new EstaticoNasdaqModelo(tempArr[0], tempArr[1], tempArr[2], tempArr[3], tempArr[4],
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

	/**
	 * NASDAQ - ESTATICOS-2 - Descargar+parsear: datos financieros sacados de
	 * OLD-NASDAQ
	 * 
	 * @param nasdaqEstaticos1
	 * @param dirBruto
	 * @param dirBrutoCsv
	 * @param numMaxEmpresas
	 * @return
	 * @throws IOException
	 */
	public static Boolean descargarYparsearNasdaqEstaticos2(List<EstaticoNasdaqModelo> nasdaqEstaticos1,
			String dirBruto, String dirBrutoCsv, Integer numMaxEmpresas) throws IOException {

		MY_LOGGER.info("descargarYparsearNasdaqEstaticos2 --> " + nasdaqEstaticos1.size() + "|" + dirBruto + "|"
				+ dirBrutoCsv + "|" + numMaxEmpresas);
		Boolean out = false;
		String rutaHtmlBruto1, rutaHtmlBruto2, rutaCsvBruto;
		int numEmpresasLeidas = 0;
		Map<String, String> mapaExtraidos = new HashMap<String, String>();

		for (EstaticoNasdaqModelo modelo : nasdaqEstaticos1) {

			numEmpresasLeidas++;
			mapaExtraidos.clear();

			if (numEmpresasLeidas <= numMaxEmpresas) {

				rutaHtmlBruto1 = dirBruto + "bruto_BORRAR1_" + MERCADO_NQ + "_" + modelo.symbol + ".html";
				rutaHtmlBruto2 = dirBruto + "bruto_BORRAR2_" + MERCADO_NQ + "_" + modelo.symbol + ".html";
				rutaCsvBruto = dirBrutoCsv + "bruto_ESTAT_" + MERCADO_NQ + "_" + modelo.symbol + ".csv";

				MY_LOGGER.info("Descarga OLD-NASDAQ - SUMMARY: " + rutaHtmlBruto1);
				descargarPagina(rutaHtmlBruto1, true, modelo.summaryQuote);
				parsearNasdaqEstatico2(rutaHtmlBruto1, mapaExtraidos);

				MY_LOGGER.info("Descarga OLD-NASDAQ - SUMMARY: " + rutaHtmlBruto1);
				descargarPagina(rutaHtmlBruto2, true, mapaExtraidos.get(ID_SRL));
				parsearNasdaqEstatico3(rutaHtmlBruto2, mapaExtraidos);

				volcarEnCSV(MERCADO_NQ, modelo.symbol, mapaExtraidos, rutaCsvBruto);

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

	/**
	 * NASDAQ - ESTATICOS-2 - Parsear (nucleo)
	 * 
	 * @param rutaHtmlBruto
	 * @param mapaExtraidos Lista de salida, que hay que rellenar con lo extraido.
	 * @return
	 * @throws IOException
	 */
	public static void parsearNasdaqEstatico2(String rutaHtmlBruto, Map<String, String> mapaExtraidos)
			throws IOException {

		MY_LOGGER.info("parsearNasdaqEstaticos2 --> " + rutaHtmlBruto);

		byte[] encoded = Files.readAllBytes(Paths.get(rutaHtmlBruto));
		String in = new String(encoded, Charset.forName("ISO-8859-1"));

		Document doc = Jsoup.parse(in);

		// --------------------- Algunos datos estaticos ---------------
		String claseBuscada = "row overview-results relativeP";

		Elements items1 = doc.getElementsByClass(claseBuscada);
		Elements items2 = items1.get(0).children();

		Elements columna1 = items2.get(1).children().get(0).children();
		Elements columna2 = items2.get(2).children().get(0).children();

		for (Element col1Item : columna1) {
			extraerUnIndicadorDeTablaConcreta(col1Item, mapaExtraidos);
		}
		for (Element col2Item : columna2) {
			extraerUnIndicadorDeTablaConcreta(col2Item, mapaExtraidos);
		}

		// ----URL del STOCK REPORT: para sacar después otros datos estáticos -------
		mapaExtraidos.put(ID_SRL, doc.getElementById(ID_SRL).attr("href"));

	}

	/**
	 * Parsea un trozo concreto de la página NASDAQ-OLD - Summary
	 * 
	 * @param in
	 * @param mapaExtraidos
	 */
	private static void extraerUnIndicadorDeTablaConcreta(Element in, Map<String, String> mapaExtraidos) {

		Elements items1 = in.children();
		Element items11 = items1.get(0).child(0);
		Element items12 = items1.get(1);

		String text11 = items11.text();
		String text12 = items12.text();

		// TODO pendiente
		int x = 0;

	}

	/**
	 * @param rutaHtmlBruto
	 * @param mapaExtraidos
	 * @throws IOException
	 */
	public static void parsearNasdaqEstatico3(String rutaHtmlBruto, Map<String, String> mapaExtraidos)
			throws IOException {

		MY_LOGGER.info("descargarYparsearNasdaqEstaticos3 --> " + rutaHtmlBruto);

		byte[] encoded = Files.readAllBytes(Paths.get(rutaHtmlBruto));
		String in = new String(encoded, Charset.forName("ISO-8859-1"));

		Document doc = Jsoup.parse(in);

		// TODO pendiente
		int x = 0;

	}

	public static void volcarEnCSV(String mercado, String empresa, Map<String, String> mapaExtraidos,
			String rutaCsvBruto) {

		MY_LOGGER.info("descargarYparsearNasdaqEstaticos3 --> " + mercado + "|" + empresa + "|" + mapaExtraidos.size()
				+ "|" + rutaCsvBruto);

		// TODO pendiente
		int x = 0;
	}

}
