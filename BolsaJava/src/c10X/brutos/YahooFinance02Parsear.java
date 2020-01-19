package c10X.brutos;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Serializable;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

/**
 * Parsear datos de Yahoo Finance
 */
public class YahooFinance02Parsear implements Serializable {

	static Logger MY_LOGGER = Logger.getLogger(YahooFinance02Parsear.class);
	static DateFormat df = new SimpleDateFormat("yyyy|MM|dd|HH|mm");

	private static YahooFinance02Parsear instancia = null;

	private YahooFinance02Parsear() {
		super();
	}

	public static YahooFinance02Parsear getInstance() {
		if (instancia == null)
			instancia = new YahooFinance02Parsear();

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

		String directorioIn = BrutosUtils.DIR_BRUTOS; // DEFAULT
		String directorioOut = BrutosUtils.DIR_BRUTOS_CSV; // DEFAULT
		String modo = BrutosUtils.PASADO; // DEFAULT
		Integer entornoDeValidacion = BrutosUtils.ES_ENTORNO_VALIDACION;// DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 4) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			int numParams = args.length;
			MY_LOGGER.info("Numero de parametros: " + numParams);
			for (String param : args) {
				MY_LOGGER.info("Param: " + param);
			}
			System.exit(-1);
		} else {
			directorioIn = args[0];
			directorioOut = args[1];
			modo = args[2];
			entornoDeValidacion = Integer.valueOf(args[3]);
		}

		// EMPRESAS NASDAQ
		List<EstaticoNasdaqModelo> nasdaqEstaticos1 = EstaticosNasdaqDescargarYParsear
				.descargarNasdaqEstaticosSoloLocal1(entornoDeValidacion);

		// VELAS (tomando una empresa buena, que tendra todo relleno)
		Map<String, Integer> velas = new HashMap<String, Integer>();
		extraerVelasReferencia(velas, directorioIn, directorioOut, modo);// coger todas las velas pasadas
																			// de la empresa de referencia

		// DATOS DINAMICOS DE TODAS LAS EMPRESAS
		parsearDinamicos01(nasdaqEstaticos1, directorioIn, directorioOut, false, modo);
		rellenarVelasDiariasHuecoyAntiguedad02(nasdaqEstaticos1, directorioOut, velas);

		MY_LOGGER.info("FIN");
	}

	/**
	 * EXTRAE LAS VELAS CON UNA EMPRESA DE REFERENCIA DE UN MERCADO
	 * 
	 * @param velas
	 * @param directorioIn
	 * @param directorioOut
	 * @param modo
	 * @throws IOException
	 */
	public static void extraerVelasReferencia(Map<String, Integer> velas, String directorioIn, String directorioOut,
			String modo) throws IOException {

		String mercadoReferencia = BrutosUtils.MERCADO_NQ;
		String valorReferencia = "AAPL";

		String ficheroConVelasYTiempos = parsearDinamicosEmpresa01(mercadoReferencia, valorReferencia, directorioIn,
				directorioOut, true, modo);

		// --------- Leer fichero -------------
		File file = new File(ficheroConVelasYTiempos);
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		String actual, antiguedad, fechaStr;
		int indexPrimerPipe;
		boolean primeraLinea = true;

		while ((actual = br.readLine()) != null) {
			if (primeraLinea == false) {
				indexPrimerPipe = actual.indexOf("|");
				antiguedad = actual.substring(0, indexPrimerPipe);
				fechaStr = actual.substring(indexPrimerPipe + 1);
				velas.put(fechaStr, Integer.valueOf(antiguedad));
			}
			primeraLinea = false;
		}
		br.close();

		MY_LOGGER.debug("Velas leidas: " + velas.size());
	}

	/**
	 * @param nasdaqEstaticos1
	 * @param directorioIn
	 * @param directorioOut
	 * @param soloVelas
	 * @param modo
	 * @return
	 * @throws IOException
	 */
	public static List<String> parsearDinamicos01(List<EstaticoNasdaqModelo> nasdaqEstaticos1, String directorioIn,
			String directorioOut, Boolean soloVelas, String modo) throws IOException {

		MY_LOGGER.debug("parsearNasdaqDinamicos01 --> " + directorioIn + "|" + directorioOut);

		String mercado = "NASDAQ"; // DEFAULT
		List<String> ficherosSalida = new ArrayList<String>();

		MY_LOGGER.debug("mercado=" + mercado);

		for (int i = 0; i < nasdaqEstaticos1.size(); i++) {
			ficherosSalida.add(parsearDinamicosEmpresa01(mercado, nasdaqEstaticos1.get(i).symbol, directorioIn,
					directorioOut, soloVelas, modo));

		}
		return ficherosSalida;
	}

	/**
	 * @param mercado
	 * @param ticker
	 * @param directorioIn
	 * @param directorioOut
	 * @param soloVelas
	 * @param modo
	 * @return
	 * @throws IOException
	 */
	public static String parsearDinamicosEmpresa01(String mercado, String ticker, String directorioIn,
			String directorioOut, Boolean soloVelas, String modo) throws IOException {

		Boolean out = false;
		String pathBruto;
		String pathBrutoCsv;
		pathBruto = directorioIn + BrutosUtils.YAHOOFINANCE + "_" + mercado + "_" + ticker + ".txt";
		pathBrutoCsv = soloVelas ? (directorioOut + "VELAS_" + mercado + ".csv")
				: (directorioOut + BrutosUtils.YAHOOFINANCE + "_" + mercado + "_" + ticker + ".csv");

		MY_LOGGER.debug("pathBruto|pathBrutoCsv =" + pathBruto + "|" + pathBrutoCsv);

		if (Files.exists(Paths.get(pathBruto))) {
			Files.deleteIfExists(Paths.get(pathBrutoCsv)); // Borramos el fichero de salida si existe

			out = parsearJson(mercado, ticker, pathBruto, pathBrutoCsv, soloVelas, modo);

			if (out.booleanValue() == false) {
				MY_LOGGER.error(
						"La descarga de datos estaticos 1 de " + mercado + " - " + ticker + " ha fallado. Saliendo...");
			}

		} else {
			MY_LOGGER.debug("Fichero de entrada no existe: " + pathBruto + " Continua...");
		}
		return pathBrutoCsv;
	}

	/**
	 * Lee un fichero bruto de datos, los extrae y los escribe en un CSV
	 * (estructurados)
	 * 
	 * @param mercado
	 * @param empresa
	 * @param pathBrutoEntrada
	 * @param pathBrutoCsvSalida
	 * @param soloVelas
	 * @param modo
	 * @return
	 */
	public static Boolean parsearJson(String mercado, String empresa, String pathBrutoEntrada,
			String pathBrutoCsvSalida, Boolean soloVelas, String modo) {

		MY_LOGGER.debug("parsearJson... --> " + pathBrutoEntrada + "|" + pathBrutoCsvSalida);
		Boolean out = false;
		JSONParser parser = new JSONParser();

		String contenido = "";

		try {
			List<String> lines = Files.readAllLines(Paths.get(pathBrutoEntrada), Charset.defaultCharset());

			for (String cad : lines) {
				contenido += cad;
			}

			// ---------------------------- LECTURA ------------------
			MY_LOGGER.debug("Lectura...");

			JSONObject primerJson = (JSONObject) parser.parse(contenido);

//			JSONObject primerJson = (JSONObject) parser.parse(reader);

			Map<String, JSONObject> mapaChart = (HashMap<String, JSONObject>) primerJson.get("chart");
			Object resultValor = mapaChart.get("result");
			JSONArray a1 = (JSONArray) resultValor;
			JSONObject a2 = (JSONObject) a1.get(0);

			JSONObject indicators = (JSONObject) a2.get("indicators");
			JSONArray tiemposEnSegundosDesde1970 = (JSONArray) a2.get("timestamp");

			JSONArray quote1 = (JSONArray) indicators.get("quote");
			JSONObject quote2 = (JSONObject) quote1.get(0);

			JSONArray listaVolumenes = (JSONArray) quote2.get("volume");
			JSONArray listaPreciosHigh = (JSONArray) quote2.get("high");
			JSONArray listaPreciosLow = (JSONArray) quote2.get("low");
			JSONArray listaPreciosClose = (JSONArray) quote2.get("close");
			JSONArray listaPreciosOpen = (JSONArray) quote2.get("open");

			if (listaVolumenes != null) {

				MY_LOGGER.debug("Tamanios --> " + listaVolumenes.size() + "|" + listaPreciosHigh.size() + "|"
						+ listaPreciosLow.size() + "|" + listaPreciosClose.size() + "|" + listaPreciosOpen.size());

				// Detectar anomalias gigantes (posibles Splits o contrasplits)
				detectarAnomaliasGigantes(empresa, listaPreciosClose, listaPreciosOpen);

				// ---------------------------- ESCRITURA ---------------
				MY_LOGGER.debug("Escritura...");
				File fout = new File(pathBrutoCsvSalida);
				FileOutputStream fos = new FileOutputStream(fout, false);
				BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));

				// Cabecera
				String cabecera = soloVelas ? "antiguedad|anio|mes|dia|hora|minuto"
						: "mercado|empresa|antiguedad|anio|mes|dia|hora|minuto|volumen|high|low|close|open";
				bw.write(cabecera);
				bw.newLine();

				int i = 0;

				String cad;
				int numVelas = listaVolumenes.size();
				for (i = 0; i < numVelas; i++) {

					long msegDesde1970 = (Long) tiemposEnSegundosDesde1970.get(i) * 1000L;

					if (soloVelas) {
						cad = String.valueOf(numVelas - i - 1);
						cad += "|" + df.format(new Date(msegDesde1970));

					} else {

						cad = mercado + "|" + empresa;
						cad += "|" + df.format(new Date(msegDesde1970));
						cad += "|" + BrutosUtils.tratamientoLigero(
								listaVolumenes.get(i) == null ? BrutosUtils.NULO : listaVolumenes.get(i).toString(),
								BrutosUtils.ESCALA_M);
						cad += "|" + BrutosUtils.tratamientoLigero(
								listaPreciosHigh.get(i) == null ? BrutosUtils.NULO : listaPreciosHigh.get(i).toString(),
								BrutosUtils.ESCALA_UNO);
						cad += "|" + BrutosUtils.tratamientoLigero(
								listaPreciosLow.get(i) == null ? BrutosUtils.NULO : listaPreciosLow.get(i).toString(),
								BrutosUtils.ESCALA_UNO);
						cad += "|" + BrutosUtils.tratamientoLigero(listaPreciosClose.get(i) == null ? BrutosUtils.NULO
								: listaPreciosClose.get(i).toString(), BrutosUtils.ESCALA_UNO);
						cad += "|" + BrutosUtils.tratamientoLigero(
								listaPreciosOpen.get(i) == null ? BrutosUtils.NULO : listaPreciosOpen.get(i).toString(),
								BrutosUtils.ESCALA_UNO);

					}

					bw.write(cad);
					bw.newLine();
				}

				bw.close();

				out = true;

			}

		} catch (IOException e) {
			MY_LOGGER.error(e.getMessage());
		} catch (ParseException e) {
			MY_LOGGER.error(e.getMessage());
		}

		return out;
	}

	/**
	 * Todos los tickers deben tener
	 * 
	 * @param nasdaqEstaticos1
	 * @param directorioCsv
	 * @param velas
	 * @throws IOException
	 */
	public static void rellenarVelasDiariasHuecoyAntiguedad02(List<EstaticoNasdaqModelo> nasdaqEstaticos1,
			String directorioCsv, Map<String, Integer> velas) throws IOException {

		// Recorrer todo el directorio, cogiendo los ficheros de Yahoo Finance y
		// metiendoles las velas que falten

		final File folder = new File(directorioCsv);
		List<String> result = new ArrayList<String>();
		BrutosUtils.encontrarFicherosEnCarpeta("YF.*\\.csv", folder, result);

		for (String pathFichero : result) {
			MY_LOGGER.debug(pathFichero);
			rellenarVelasDiariasHuecoyAntiguedadPorFichero03(pathFichero, velas);
		}
	}

	/**
	 * Dado un fichero CSV de Yahoo Finance (con precios, etc), rellena las velas
	 * (antiguedad). Genera las filas HUECO que falten, poniendo PRECIO
	 * ARRASTRADO (de la ultima vela conocida) y VOLUMEN CERO.
	 * 
	 * @param pathFicheroIn
	 * @param velas
	 * @throws IOException
	 */
	public static void rellenarVelasDiariasHuecoyAntiguedadPorFichero03(String pathFicheroIn,
			Map<String, Integer> velas) throws IOException {

		List<String> lista = new ArrayList<String>();

		File file = new File(pathFicheroIn);
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);

		String actual = "", anterior = null;
		Integer velaAnterior = null;
		String ultimaLineaRellenaCompletaConocida = null;
		boolean primeraLinea = true;
		boolean filaActualEsCompleta = false;

		long num = 1;
		String[] actualArray = {};

		while ((actual = br.readLine()) != null) {

			filaActualEsCompleta = false; // default

			int numeroCamposActual = 0;
			if (actual != null & !actual.isEmpty() && actual.contains("|")) {
				numeroCamposActual = actual.split("\\|").length;

				actualArray = actual.split("\\|");
			}

			if (primeraLinea == false) {// excluye la cabecera

				String claveActual = actualArray[2] + "|" + actualArray[3] + "|" + actualArray[4] + "|" + actualArray[5]
						+ "|" + actualArray[6];
				Integer velaActual = velas.get(claveActual);

				if (numeroCamposActual >= 12) {
					String[] lineArray = actual.split("\\|");

					// RELLENOS: volumen y precio de cierre
					Boolean parte7NoNula = !lineArray[7].isEmpty() && !lineArray[7].equalsIgnoreCase(BrutosUtils.NULO);
					Boolean parte10NoNula = !lineArray[10].isEmpty()
							&& !lineArray[10].equalsIgnoreCase(BrutosUtils.NULO);
					filaActualEsCompleta = parte7NoNula && parte10NoNula;

					ultimaLineaRellenaCompletaConocida = filaActualEsCompleta ? actual
							: ultimaLineaRellenaCompletaConocida;

					if (anterior != null && ultimaLineaRellenaCompletaConocida != null) {
						String[] anteriorArray = actual.split("\\|");
						String claveAnterior = anteriorArray[2] + "|" + anteriorArray[3] + "|" + anteriorArray[4] + "|"
								+ anteriorArray[5] + "|" + anteriorArray[6];
						velaAnterior = velas.get(claveAnterior);

						// Si entre la vela anterior y la actual faltan filas, las CREO con precio
						// arrastrado y volumen cero

						String[] completaAnteriorArray = ultimaLineaRellenaCompletaConocida.split("\\|");

						if (velaActual != null && velaAnterior != null) {

							for (int numVelaHueco = (velaAnterior + 1); numVelaHueco < velaActual; numVelaHueco++) {

								String corregida = actualArray[0];// mercado
								corregida += "|" + actualArray[1];// empresa
								corregida += "|" + String.valueOf(numVelaHueco);// antiguedad (vela)
								corregida += "|" + actualArray[2];// anio
								corregida += "|" + actualArray[3];// mes
								corregida += "|" + actualArray[4];// dia
								corregida += "|" + actualArray[5];// hora
								corregida += "|" + actualArray[6];// minuto
								corregida += "|" + "0";// CORRIJO volumen: pongo un CERO
								corregida += "|" + completaAnteriorArray[8];// CORRIJO high
								corregida += "|" + completaAnteriorArray[9];// CORRIJO low
								corregida += "|" + completaAnteriorArray[10];// CORRIJO close
								corregida += "|" + completaAnteriorArray[11];// CORRIJO open

								// FILA CORREGIDA
								lista.add(corregida);
							}

						}

					}
				}

				// En FILAS YA EXISTENTES (no son huecos), que tengan datos null, tambien
				// relleno con precio arrastrado y volumen cero
				if (ultimaLineaRellenaCompletaConocida != null
						&& (filaActualEsCompleta == false || numeroCamposActual >= 12)) {

					String[] completaAnteriorArray = ultimaLineaRellenaCompletaConocida.split("\\|");

					String corregida = actualArray[0];// mercado
					corregida += "|" + actualArray[1];// empresa
					corregida += "|" + String.valueOf(velaActual);// antiguedad (vela)
					corregida += "|" + actualArray[2];// anio
					corregida += "|" + actualArray[3];// mes
					corregida += "|" + actualArray[4];// dia
					corregida += "|" + actualArray[5];// hora
					corregida += "|" + actualArray[6];// minuto
					corregida += "|" + "0";// CORRIJO volumen: pongo un CERO
					corregida += "|" + completaAnteriorArray[8];// CORRIJO high
					corregida += "|" + completaAnteriorArray[9];// CORRIJO low
					corregida += "|" + completaAnteriorArray[10];// CORRIJO close
					corregida += "|" + completaAnteriorArray[11];// CORRIJO open

					// FILA CORREGIDA
					lista.add(corregida);

				} else {

					String[] originalArray = actual.split("\\|");

					String originalconVela = originalArray[0];// mercado
					originalconVela += "|" + originalArray[1];// empresa
					originalconVela += "|" + String.valueOf(velaActual);// antiguedad (vela)
					originalconVela += "|" + originalArray[2];// anio
					originalconVela += "|" + originalArray[3];// mes
					originalconVela += "|" + originalArray[4];// dia
					originalconVela += "|" + originalArray[5];// hora
					originalconVela += "|" + originalArray[6];// minuto
					originalconVela += "|" + originalArray[7];// volumen
					originalconVela += "|" + originalArray[8];// high
					originalconVela += "|" + originalArray[9];// low
					originalconVela += "|" + originalArray[10];// close
					originalconVela += "|" + originalArray[11];// open

					lista.add(originalconVela);
				}

				anterior = actual;// GUARDO PARA LA SIGUIENTE ITERACION (salvo la cabecera, que no la quiero)

			} else {
				// CABECERA INTACTA
				lista.add(actual);
			}

			primeraLinea = false;
		}
		br.close();

		MY_LOGGER.debug("Lineas leidas: " + lista.size());

		// ---------------- ESCRITURA: sustituye al existente ------
		MY_LOGGER.debug("Escritura...");
		File fout = new File(pathFicheroIn);
		PrintWriter writer = new PrintWriter(file);
		writer.print("");// VACIAMOS CONTENIDO
		writer.close();
		FileOutputStream fos = new FileOutputStream(fout, false);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));

		for (String cad : lista) {
			bw.write(cad);
			bw.newLine();
		}
		bw.close();

	}

	/**
	 * @param listaPreciosClose
	 * @param listaPreciosOpen
	 */
	public static void detectarAnomaliasGigantes(String empresa, JSONArray listaPreciosClose,
			JSONArray listaPreciosOpen) {

		if (listaPreciosClose.size() == listaPreciosOpen.size()) {

			for (int i = 0; i < listaPreciosClose.size(); i++) {

				if (i > 0) { // la primera vela no me vale para poder comparar

					Double close1 = listaPreciosOpen.get(i) == null ? null
							: Double.valueOf(listaPreciosOpen.get(i).toString());
					Double close0 = listaPreciosOpen.get(i - 1) == null ? null
							: Double.valueOf(listaPreciosOpen.get(i - 1).toString());

					if (close1 != null && close0 != null) {

						Double ratioDeCambio = (close1 > close0) ? Math.abs(close1 / close0) - 1
								: Math.abs(close0 / close1);

						if (ratioDeCambio > 2) {
							MY_LOGGER.warn("Empresa=" + empresa
									+ " Tiene anomalia gigante (posible split o contraSplit) en vela = " + i
									+ " con ratio de cambio = " + ratioDeCambio);
						}

					}

				}

			}

		} else {
			MY_LOGGER.error("Listas de distinto tamanio! Ejemplo de una fila: " + listaPreciosClose.get(0).toString());
		}

	}

}
