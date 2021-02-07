package c10X.brutos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
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
public class EstaticosNasdaqDescargarYParsear implements Serializable {

	static Logger MY_LOGGER = Logger.getLogger(EstaticosNasdaqDescargarYParsear.class);

	static String ID_SRL = "stockreportslink";

	private static EstaticosNasdaqDescargarYParsear instancia = null;

	// En la lista DIRECTA de empresas, saltamos todas las empresas cuyo ticker
	// empiece por una letra anterior a la indicada (orden alfabético).
	// Ej: si letra=N, saltamos todas las empresas cuyo ticker empieza por A-N
	private static String LETRA_INICIO_LISTA_DIRECTA = "A"; // Default= A

	private EstaticosNasdaqDescargarYParsear() {
		super();
	}

	public static EstaticosNasdaqDescargarYParsear getInstance() {
		MY_LOGGER.info("getInstance...");
		if (instancia == null)
			instancia = new EstaticosNasdaqDescargarYParsear();

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
		Integer entornoDeValidacion = 1; // DEFAULT

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

		List<EstaticoNasdaqModelo> nasdaqEstaticos1 = descargarNasdaqEstaticosSoloLocal1(entornoDeValidacion);
		descargarYparsearNasdaqEstaticos2(nasdaqEstaticos1, dirBruto, dirBrutoCsv, numMaxEmpresas);

		MY_LOGGER.info("FIN");
	}

	/**
	 * NASDAQ - ESTATICOS-1: datos cargados desde un fichero local.
	 * 
	 * @param entornoDeValidacion
	 * @return Lista de empresas del NASDAQ con algunos datos ESTATICOS
	 */
	public static List<EstaticoNasdaqModelo> descargarNasdaqEstaticosSoloLocal1(final Integer entornoDeValidacion) {

		MY_LOGGER.info("descargarNasdaqEstaticos1...");

		List<String> desconocidos = new ArrayList<String>();
		try {
			File file = new File(BrutosUtils.DESCONOCIDOS_CSV);
			FileReader fr = new FileReader(file);
			BufferedReader br = new BufferedReader(fr);

			String line = "";

			while ((line = br.readLine()) != null) {
				if (line != null && !line.isEmpty()) {
					desconocidos.add(line);
				}
			}
			br.close();

			MY_LOGGER.info("DESCONOCIDOS leidos: " + desconocidos.size());

		} catch (IOException ioe) {
			ioe.printStackTrace();
		}

		String pathTickers;

		if (entornoDeValidacion == 1) {
			pathTickers = BrutosUtils.NASDAQ_TICKERS_CSV_INVERTIDOS;
		} else {
			pathTickers = BrutosUtils.NASDAQ_TICKERS_CSV;
		}

		char[] ALFABETO = "abcdefghijklmnopqrstuvwxyz".toCharArray();

		MY_LOGGER.info("Cargando NASDAQ-TICKERS de: " + pathTickers);
		List<String> empresasDescargables = new ArrayList<String>();
		List<EstaticoNasdaqModelo> out = new ArrayList<EstaticoNasdaqModelo>();
		try {
			File file = new File(pathTickers);
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

				if (primeraLinea && lineaLimpia != null && !lineaLimpia.isEmpty()) {
					primeraLinea = false;

				} else if (primeraLinea == false && lineaLimpia != null && !lineaLimpia.isEmpty()) {

					tempArr = lineaLimpia.split("\\|", -1); // El -1 indica coger las cadenas vacías!!

					// Lista acumulada de desconocidos:
					boolean empresaEnListaDesconocidos = desconocidos.contains(tempArr[0]);
					boolean empresaEnRangoDeLetrasPermitido = letraEstaPermitida(ALFABETO, tempArr[0],
							LETRA_INICIO_LISTA_DIRECTA);
					// Si ya la hemos descargado, evitamos volver a traerla:
					boolean empresaYaDescargada = empresasDescargables.contains(tempArr[0]);
					// Empresa de la que desconoce su URL de Nasdaq.old:
					boolean sinURLnasdaqOld = (tempArr[7] == null || tempArr[7].isEmpty());

					if (empresaEnListaDesconocidos) {
						MY_LOGGER.info("Empresa de la que sabemos que desconocemos datos en alguna de las fuentes: "
								+ tempArr[0]);

					} else if (!empresaYaDescargada && empresaEnRangoDeLetrasPermitido) {

						if (sinURLnasdaqOld) {
							out.add(new EstaticoNasdaqModelo(tempArr[0], tempArr[1], tempArr[2], tempArr[3], tempArr[4],
									tempArr[5], tempArr[6], tempArr[7]));
							empresasDescargables.add(tempArr[0]);

						} else {

							out.add(new EstaticoNasdaqModelo(tempArr[0], tempArr[1], tempArr[2], tempArr[3], tempArr[4],
									tempArr[5], tempArr[6], tempArr[7]));
							empresasDescargables.add(tempArr[0]);
						}

					}
				}
			}
			br.close();

			MY_LOGGER.info("NASDAQ-TICKERS leidos: " + out.size());

		} catch (IOException ioe) {
			ioe.printStackTrace();
		}

		MY_LOGGER.info("Empresas DESCONOCIDAS (sabemos que falta info en alguna de las fuentes de datos): "
				+ BrutosUtils.DESCONOCIDOS_CSV + " --> Son " + desconocidos.size() + " empresas");

		// Colocar a la empresa de referencia la primera!!
		return colocar(out, entornoDeValidacion);
	}

	/**
	 * @param lista
	 * @param ordenAlfabeticoDirecto True=directo, False=inverso
	 * @return
	 */
	public static List<EstaticoNasdaqModelo> colocar(List<EstaticoNasdaqModelo> lista, Integer ordenAlfabeticoDirecto) {

		// Empresa de referencia
		int indiceItemReferencia = -1;
		for (EstaticoNasdaqModelo item : lista) {
			if (item.getSymbol().equals(BrutosUtils.NASDAQ_REFERENCIA)) {
				indiceItemReferencia = lista.indexOf(item);
			}
		}

		List<EstaticoNasdaqModelo> listaOrdenada = new ArrayList<EstaticoNasdaqModelo>();

		if (indiceItemReferencia > -1) {
			EstaticoNasdaqModelo enmRef = lista.get(indiceItemReferencia);

			listaOrdenada.add(enmRef);

			// Quitar la empresa de referencia
			lista.remove(indiceItemReferencia);

			// Ahora que ya hemos sacado a la empresa de referencia, ordenamos
			// alfabeticamente

			if (ordenAlfabeticoDirecto == 0) {
				Collections.sort(lista);

			} else if (ordenAlfabeticoDirecto == 1) {
				Collections.sort(lista);// primero ordenamos alfabeticamente
				Collections.reverse(lista); // le damos la vuelta
			}

			// Añadimos todos a la lista de salida
			listaOrdenada.addAll(lista);

		} else {
			MY_LOGGER.error("La empresa de referencia " + BrutosUtils.NASDAQ_REFERENCIA
					+ " no aparece en la lista de empresas descargables. Abortamos!!");
			System.exit(-1);
		}

		return listaOrdenada;

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
	 * @throws InterruptedException
	 */
	public static Boolean descargarYparsearNasdaqEstaticos2(List<EstaticoNasdaqModelo> nasdaqEstaticos1,
			String dirBruto, String dirBrutoCsv, Integer numMaxEmpresas) throws IOException, InterruptedException {

		MY_LOGGER.info("descargarYparsearNasdaqEstaticos2 --> " + nasdaqEstaticos1.size() + "|" + dirBruto + "|"
				+ dirBrutoCsv + "|" + numMaxEmpresas);
		Boolean out = false;
		String rutaHtmlBruto1, rutaHtmlBruto2, rutaCsvBruto, rutaHtmlBruto3;
		int numEmpresasLeidas = 0;
		Map<String, String> mapaExtraidos = new HashMap<String, String>();
		long msegEspera = -1;

		for (EstaticoNasdaqModelo modelo : nasdaqEstaticos1) {

			numEmpresasLeidas++;
			mapaExtraidos.clear();

			if (numEmpresasLeidas <= numMaxEmpresas) {

				rutaHtmlBruto1 = dirBruto + BrutosUtils.NASDAQOLD + "_BORRAR1_" + BrutosUtils.MERCADO_NQ + "_"
						+ modelo.symbol + ".html";
				rutaHtmlBruto2 = dirBruto + BrutosUtils.NASDAQOLD + "_BORRAR2_" + BrutosUtils.MERCADO_NQ + "_"
						+ modelo.symbol + ".html";
				rutaHtmlBruto3 = dirBruto + BrutosUtils.NASDAQOLD + "_BORRAR3_" + BrutosUtils.MERCADO_NQ + "_"
						+ modelo.symbol + ".html";
				rutaCsvBruto = dirBrutoCsv + BrutosUtils.NASDAQOLD + "_" + BrutosUtils.MERCADO_NQ + "_" + modelo.symbol
						+ ".csv";

				MY_LOGGER.info("Descarga OLD-NASDAQ - SUMMARY: " + rutaHtmlBruto1);

				if (modelo.summaryQuote != null && !modelo.summaryQuote.isEmpty()) {

					msegEspera = (long) (BrutosUtils.ESPERA_ALEATORIA_MSEG_MIN
							+ Math.random() * 1000 * BrutosUtils.ESPERA_ALEATORIA_SEG_MAX);
					Thread.sleep(msegEspera);
					descargarPagina(rutaHtmlBruto1, true, modelo.summaryQuote);
					parsearNasdaqEstatico2(rutaHtmlBruto1, mapaExtraidos);

					MY_LOGGER.info("Descarga OLD-NASDAQ - SUMMARY: " + rutaHtmlBruto1);
					msegEspera = (long) (BrutosUtils.ESPERA_ALEATORIA_MSEG_MIN
							+ Math.random() * 1000 * BrutosUtils.ESPERA_ALEATORIA_SEG_MAX);
					Thread.sleep(msegEspera);
					descargarPagina(rutaHtmlBruto2, true, mapaExtraidos.get(ID_SRL));
					parsearNasdaqEstatico3(rutaHtmlBruto2, rutaHtmlBruto3, mapaExtraidos);

				} else {
					MY_LOGGER.info("Descarga OLD-NASDAQ - SUMMARY: " + rutaHtmlBruto1 + " --> No sabemos su URL");
				}

				volcarEnCSV(BrutosUtils.MERCADO_NQ, modelo.symbol, mapaExtraidos, rutaCsvBruto);

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

		// ----URL del STOCK REPORT: para sacar despu�s otros datos est�ticos -------
		mapaExtraidos.put(ID_SRL, doc.getElementById(ID_SRL).attr("href"));

	}

	/**
	 * Parsea un trozo concreto de la pagina NASDAQ-OLD - Summary
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

		mapaExtraidos.put(text11, text12);
	}

	/**
	 * @param rutaHtmlBruto
	 * @param rutaHtmlReportBruto
	 * @param mapaExtraidos
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public static void parsearNasdaqEstatico3(String rutaHtmlBruto, String rutaHtmlReportBruto,
			Map<String, String> mapaExtraidos) throws IOException, InterruptedException {

		MY_LOGGER.info("descargarYparsearNasdaqEstaticos3 --> " + rutaHtmlBruto);

		byte[] encoded = Files.readAllBytes(Paths.get(rutaHtmlBruto));
		String in = new String(encoded, Charset.forName("ISO-8859-1"));

		Document doc = Jsoup.parse(in);

		// -------------------- Tabla de ARRIBA (contiene el dividendo_pct y PER)
		Elements t1 = doc.getElementsByClass("marginB5px");
		Elements t11 = t1.get(1).children().get(0).children();

		List<String> tempItemValor = new ArrayList<String>();

		for (Element t111 : t11) {

			Elements t2 = t111.children();

			for (Element t21 : t2) {

				if (!t21.ownText().trim().isEmpty()) {
					tempItemValor.add(t21.ownText().trim());
				} else {
					Element t3 = t21.children().get(0);
					Elements t31 = t3.getAllElements();
					Element t311 = t31.get(0);
					Element t4 = t311.getAllElements().get(0);
					tempItemValor.add(t4.ownText());
				}

			}
		}

		int i = 0;
		for (i = 0; i < tempItemValor.size(); i = i + 2) {
			// MY_LOGGER.info(tempItemValor.get(i) + "-->" + tempItemValor.get(i + 1));

			// De los datos que hay, solo cojo algunos
			if (tempItemValor.get(i).contains("Dividend Yield") || tempItemValor.get(i).contains("P/E Ratio")) {
				mapaExtraidos.put(tempItemValor.get(i), tempItemValor.get(i + 1));
			}
		}

		// -------------------- Tabla de ARRIBA (contiene el dividendo_pct y PER)
		Element p2 = doc.getElementById("stockreportInfo");
		Element p21 = p2.children().get(0);
		String urlStockReport = p21.attr("src");

		long msegEspera = (long) (BrutosUtils.ESPERA_ALEATORIA_MSEG_MIN
				+ Math.random() * 1000 * BrutosUtils.ESPERA_ALEATORIA_SEG_MAX);
		// TODO Thread.sleep(msegEspera);
		// TODO descargarPagina(rutaHtmlReportBruto, true, urlStockReport);
		// TODO parsearNasdaqEstatico4(rutaHtmlReportBruto, mapaExtraidos);

		// TODO pendiente
		int x = 0;

	}

	/**
	 * Parsea STOCK REPORT (NASDAQ EDGAR)
	 * 
	 * @param rutaHtmlReportBruto
	 * @param mapaExtraidos
	 * @throws IOException
	 */
	public static void parsearNasdaqEstatico4(String rutaHtmlReportBruto, Map<String, String> mapaExtraidos)
			throws IOException {

		MY_LOGGER.info("parsearNasdaqEstatico4 --> " + rutaHtmlReportBruto);

		byte[] encoded = Files.readAllBytes(Paths.get(rutaHtmlReportBruto));
		String in = new String(encoded, Charset.forName("ISO-8859-1"));

		Document doc = Jsoup.parse(in);

		Elements t = doc.getElementById("pnlReport").children().get(0).children().get(0).children().get(0).children()
				.get(0).children();

		for (Element e : t) {
			// BALANCE SHEET: activos y pasivos totales
			procesarNasdaqEstatico4Tabla(e, "Total Assets", mapaExtraidos);
			procesarNasdaqEstatico4Tabla(e, "Total Liabilities", mapaExtraidos);

			// STOCK OWNERSHIP: porcentaje de particiones por parte de instituciones
			// (fondos)
			procesarNasdaqEstatico4Tabla(e, "Institutional", mapaExtraidos);

		}

	}

	/**
	 * @param t
	 * @param datoBuscado
	 * @param mapaExtraidos
	 */
	private static void procesarNasdaqEstatico4Tabla(Element t, String datoBuscado, Map<String, String> mapaExtraidos) {

		if (t.toString().contains(">" + datoBuscado + "<")) {
			MY_LOGGER.info("procesarNasdaqEstatico4Tabla --> " + datoBuscado);
			MY_LOGGER.info("-----------------------------------");
			MY_LOGGER.info(t.toString());
			MY_LOGGER.info("-----------------------------------");

			// TODO
			int x = 0;
		}

	}

	/**
	 * @param mercado
	 * @param empresa
	 * @param mapaExtraidos
	 * @param rutaCsvBruto
	 */
	public static void volcarEnCSV(String mercado, String empresa, Map<String, String> mapaExtraidos,
			String rutaCsvBruto) {

		MY_LOGGER.info("volcarEnCSV --> " + mercado + "|" + empresa + "|" + mapaExtraidos.size() + "|" + rutaCsvBruto);

		// TODO pendiente
		int x = 0;
	}

	/**
	 * @param alfabeto
	 * @param ticker
	 * @param letraInicio
	 * @return
	 */
	public static boolean letraEstaPermitida(char[] alfabeto, String ticker, String letraInicio) {

		char letraTickerAnalizada = ticker.charAt(0);

		boolean letraInicioEncontrada = false, letraTickerAnalizadaEncontrada = false, permitida = false;

		for (char c : alfabeto) {
			if (letraInicio.equalsIgnoreCase(String.valueOf(c))) {
				letraInicioEncontrada = true;
			}

			// Si encuentro la letra de inicio, miramos si esa letra o las restantes del
			// alfabeto son la primera letra del ticker analizada
			if (letraInicioEncontrada && String.valueOf(letraTickerAnalizada).equalsIgnoreCase(String.valueOf(c))) {
				permitida = true;
			}
		}

		if (ticker.equals(BrutosUtils.NASDAQ_REFERENCIA)) {
			permitida = true; // La excepcion
		}

		MY_LOGGER.info("letraEstaPermitida --> Letra_inicial=" + letraInicio + " --> ticker=" + ticker
				+ " --> Permitida=" + permitida);

		return permitida;
	}

}
