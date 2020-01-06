package c10X.brutos;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

/**
 * Datos ESTATICOS
 *
 */
public class EstaticosFinvizDescargarYParsear {

	static Logger MY_LOGGER = Logger.getLogger(EstaticosFinvizDescargarYParsear.class);

	static String ID_SRL = "stockreportslink";

	public static final String SECTOR_BM = "basicmaterials";
	public static final String SECTOR_CONG = "conglomerates";
	public static final String SECTOR_CONSGO = "consumergoods";
	public static final String SECTOR_FIN = "financial";
	public static final String SECTOR_HC = "healthcare";
	public static final String SECTOR_IG = "industrialgoods";
	public static final String SECTOR_SERV = "services";
	public static final String SECTOR_TECH = "technology";
	public static final String SECTOR_UTIL = "utilities";

	private static EstaticosFinvizDescargarYParsear instancia = null;

	private EstaticosFinvizDescargarYParsear() {
		super();
	}

	public static EstaticosFinvizDescargarYParsear getInstance() {
		if (instancia == null)
			instancia = new EstaticosFinvizDescargarYParsear();

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
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(PatternLayout.TTCC_CONVERSION_PATTERN)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

		Integer numMaxEmpresas = BrutosUtils.NUM_EMPRESAS_PRUEBAS; // DEFAULT
		String dirBruto = BrutosUtils.DIR_BRUTOS; // DEFAULT
		String dirBrutoCsv = BrutosUtils.DIR_BRUTOS_CSV; // DEFAULT
		Integer entornoDeValidacion = BrutosUtils.ES_ENTORNO_VALIDACION;// DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 4) {
			MY_LOGGER.error("Parametros de entrada incorrectos!! --> " + args.length);
			int numParams = args.length;
			MY_LOGGER.error("Numero de parametros: " + numParams);
			for (String param : args) {
				MY_LOGGER.error("Param: " + param);
			}

			System.exit(-1);

		} else {
			numMaxEmpresas = Integer.valueOf(args[0]);
			dirBruto = args[1];
			dirBrutoCsv = args[2];
			entornoDeValidacion = Integer.valueOf(args[3]);
			MY_LOGGER.debug("PARAMS -> " + numMaxEmpresas + " | " + dirBruto + " | " + dirBrutoCsv + "|"
					+ entornoDeValidacion.toString());
		}

		Map<String, String> mapaExtraidos = new HashMap<String, String>();
		List<EstaticoNasdaqModelo> nasdaqEstaticos1 = EstaticosNasdaqDescargarYParsear
				.descargarNasdaqEstaticosSoloLocal1(entornoDeValidacion);

		String rutaHtmlBruto;
		long msegEspera = -1;

		for (int i = 0; i < Math.min(numMaxEmpresas, nasdaqEstaticos1.size()); i++) {
			if (i % 10 == 1) {
				MY_LOGGER.info("Empresa numero = " + (i + 1));
			}
			mapaExtraidos.clear();

			String empresa = nasdaqEstaticos1.get(i).symbol;

			String urlFinvizEmpresa = "https://finviz.com/quote.ashx?t=" + empresa;
			rutaHtmlBruto = dirBruto + BrutosUtils.FINVIZ + "_" + BrutosUtils.MERCADO_NQ + "_" + empresa + ".html";

			MY_LOGGER.debug("URL | destino --> " + urlFinvizEmpresa + " | " + rutaHtmlBruto);
			msegEspera = (long) (BrutosUtils.ESPERA_ALEATORIA_MSEG_MIN
					+ Math.random() * 1000 * BrutosUtils.ESPERA_ALEATORIA_SEG_MAX);
			Thread.sleep(msegEspera);
			boolean descargaBien = descargarPaginaFinviz(rutaHtmlBruto, true, urlFinvizEmpresa);

			if (descargaBien) {
				parsearFinviz1(empresa, rutaHtmlBruto, mapaExtraidos);

				if (mapaExtraidos.size() > 0) {
					String rutaCsvBruto = dirBrutoCsv + BrutosUtils.FINVIZ + "_" + BrutosUtils.MERCADO_NQ + "_"
							+ nasdaqEstaticos1.get(i).symbol + ".csv";
					volcarEnCSV(BrutosUtils.MERCADO_NQ, nasdaqEstaticos1.get(i).symbol, mapaExtraidos, rutaCsvBruto);
				}

			} else {
//				Files.write(Paths.get(BrutosUtils.DESCONOCIDOS_CSV), (empresa + "\n").getBytes(),
//						StandardOpenOption.APPEND);

				File outputFile = new File(BrutosUtils.DESCONOCIDOS_CSV);
				FileWriter out = new FileWriter(outputFile, true); // append
				out.write(empresa);
				out.write("\n");
				out.close();

			}
		}

		MY_LOGGER.debug("FIN");
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
	public static Boolean descargarPaginaFinviz(String pathOut, Boolean borrarSiExiste, String urlEntrada) {

		MY_LOGGER.debug("descargarPaginaFinviz --> " + urlEntrada + " | " + pathOut + " | " + borrarSiExiste);
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
	public static void parsearFinviz1(String idEmpresa, String rutaHtmlBruto, Map<String, String> mapaExtraidos)
			throws IOException {

		MY_LOGGER.debug("parsearNasdaqEstaticos2 --> " + idEmpresa + "|" + rutaHtmlBruto);

		byte[] encoded = Files.readAllBytes(Paths.get(rutaHtmlBruto));
		String in = new String(encoded, Charset.forName("ISO-8859-1"));

		Document doc = Jsoup.parse(in);

		// -----------Sector + Industria + GEO -----------------
		String claseSectoresYOtros = "fullview-links";
		Elements items = doc.getElementsByClass(claseSectoresYOtros);
		extraerSectorIndustriaGeo(items, mapaExtraidos);

		// --------------------- Algunos datos estaticos ---------------
		String claseBuscada = "table-dark-row";
		Elements tablas = doc.getElementsByClass(claseBuscada);

		for (Element t : tablas) {

			extraerInfoDeFila("P/E", t, mapaExtraidos, BrutosUtils.ESCALA_UNO);
			extraerInfoDeFila("Insider Own", t, mapaExtraidos, BrutosUtils.ESCALA_UNO);
			extraerInfoDeFila("Market Cap", t, mapaExtraidos, BrutosUtils.ESCALA_M);
			extraerInfoDeFila("EPS next Y", t, mapaExtraidos, BrutosUtils.ESCALA_UNO);
			extraerInfoDeFila("Inst Own", t, mapaExtraidos, BrutosUtils.ESCALA_UNO);
			extraerInfoDeFila("Dividend %", t, mapaExtraidos, BrutosUtils.ESCALA_UNO);
			extraerInfoDeFila("Quick Ratio", t, mapaExtraidos, BrutosUtils.ESCALA_UNO);
			extraerInfoDeFila("Employees", t, mapaExtraidos, BrutosUtils.ESCALA_UNO);
			extraerInfoDeFila("Current Ratio", t, mapaExtraidos, BrutosUtils.ESCALA_UNO);
			extraerInfoDeFila("Debt/Eq", t, mapaExtraidos, BrutosUtils.ESCALA_UNO);
			extraerInfoDeFila("LT Debt/Eq", t, mapaExtraidos, BrutosUtils.ESCALA_UNO);

		}

	}

	/**
	 * En la pagina de FINVIZ, extrae los campos de: Sector Económico, Industria y
	 * Zona Geográfica
	 * 
	 * @param t
	 * @param mapaExtraidos
	 */
	private static void extraerSectorIndustriaGeo(Elements items, Map<String, String> mapaExtraidos) {

		for (Element t : items) {
			String cad = t.toString();
			if (cad.contains("sec_") && cad.contains("ind_") && cad.contains("geo_")) {

				String[] partesSec = cad.substring(cad.indexOf("sec_")).split("\"", 2);
				String sector = partesSec[0].replace("sec_", "");
				mapaExtraidos.put("sector", sector);

				String[] partesInd = cad.substring(cad.indexOf("ind_")).split("\"", 2);
				String industria = partesInd[0].replace("ind_", "");
				mapaExtraidos.put("industria", industria);

				String[] partesGeo = cad.substring(cad.indexOf("geo_")).split("\"", 2);
				String geo = partesGeo[0].replace("geo_", "");
				mapaExtraidos.put("geo", geo);
			}

		}

	}

	/**
	 * @param datoBuscado
	 * @param t
	 * @param mapaExtraidos
	 */
	private static void extraerInfoDeFila(String datoBuscado, Element t, Map<String, String> mapaExtraidos,
			String escala) {

		int i = 0;
		if (t.toString().contains(">" + datoBuscado + "<")) {
			Elements parejas = t.children();
			for (i = 0; i < parejas.size(); i = i + 2) {

				if (parejas.get(i).toString().contains(">" + datoBuscado + "<")) {

					Element dato = parejas.get(i + 1).children().get(0);
					String datoString = BrutosUtils.tratamientoLigero(dato.text(), escala);

					mapaExtraidos.put(datoBuscado, datoString);
					break;
				}

			}
		}

	}

	/**
	 * @param mercado
	 * @param empresa
	 * @param mapaExtraidos
	 * @param rutaCsvBruto
	 * @throws IOException
	 */
	public static void volcarEnCSV(String mercado, String empresa, Map<String, String> mapaExtraidos,
			String rutaCsvBruto) throws IOException {

		MY_LOGGER.debug("volcarEnCSV --> " + mercado + "|" + empresa + "|" + mapaExtraidos.size() + "|" + rutaCsvBruto);

		// ---------------------------- ESCRITURA ---------------
		MY_LOGGER.debug("Escritura...");
		File fout = new File(rutaCsvBruto);
		FileOutputStream fos = new FileOutputStream(fout, false);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));

		// Cabecera
		String cabecera = "mercado|empresa";
		for (String s : mapaExtraidos.keySet()) {
			cabecera += "|" + s;
		}

		bw.write(cabecera);
		bw.newLine();

		String fila = mercado + "|" + empresa;
		for (String item : mapaExtraidos.values()) {
			fila += "|" + item;
		}
		bw.write(fila);
		bw.newLine();

		bw.close();

	}

}
