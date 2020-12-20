package c10X.brutos;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

/**
 * Juntar ESTATICOS + DINAMICOS
 *
 */
public class JuntarEstaticosYDinamicosCSVunico {

	static Logger MY_LOGGER = Logger.getLogger(JuntarEstaticosYDinamicosCSVunico.class);

	final static String CABECERA_OPS_INSIDERS = "flagOperacionesInsiderUltimos90dias|flagOperacionesInsiderUltimos30dias|flagOperacionesInsiderUltimos15dias|flagOperacionesInsiderUltimos5dias";

	private static JuntarEstaticosYDinamicosCSVunico instancia = null;
	public static final SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd");

	private static final int CINCO = 5, QUINCE = 15, TREINTA = 30, NOVENTA = 90;

	private JuntarEstaticosYDinamicosCSVunico() {
		super();
	}

	public static JuntarEstaticosYDinamicosCSVunico getInstance() {
		if (instancia == null)
			instancia = new JuntarEstaticosYDinamicosCSVunico();

		return instancia;
	}

	/**
	 * @param args
	 * @throws IOException
	 * @throws ParseException
	 */
	public static void main(String[] args) throws IOException, ParseException {

		Object appendersAcumulados = Logger.getRootLogger().getAllAppenders();
		if (appendersAcumulados instanceof NullEnumeration) {
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(PatternLayout.TTCC_CONVERSION_PATTERN)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

		// DEFAULT
		String dirBrutoCsv = BrutosUtils.DIR_BRUTOS_CSV;
		Integer desplazamientoAntiguedad = BrutosUtils.DESPLAZAMIENTO_ANTIGUEDAD;
		Integer entornoDeValidacion = BrutosUtils.ES_ENTORNO_VALIDACION;// DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");

		} else if (args.length != 3) {
			MY_LOGGER.error("Parametros de entrada incorrectos!! --> " + args.length);
			int numParams = args.length;
			MY_LOGGER.error("Numero de parametros: " + numParams);
			for (String param : args) {
				MY_LOGGER.error("Param: " + param);
			}

			System.exit(-1);

		} else {
			dirBrutoCsv = args[0];
			desplazamientoAntiguedad = Integer.valueOf(args[1]);
			entornoDeValidacion = Integer.valueOf(args[2]);
			MY_LOGGER.info("PARAMS -> " + dirBrutoCsv);
			MY_LOGGER.info("PARAMS -> " + desplazamientoAntiguedad);
			MY_LOGGER.info("PARAMS -> " + entornoDeValidacion);
		}

		nucleo(dirBrutoCsv, desplazamientoAntiguedad, entornoDeValidacion);
		MY_LOGGER.info("FIN");
	}

	/**
	 * @param dirBrutoCsv
	 * @param desplazamientoAntiguedad
	 * @param entornoDeValidacion
	 * @throws IOException
	 * @throws ParseException
	 */
	public static void nucleo(String dirBrutoCsv, Integer desplazamientoAntiguedad, final Integer entornoDeValidacion)
			throws IOException, ParseException {

		List<EstaticoNasdaqModelo> nasdaqEstaticos1 = EstaticosNasdaqDescargarYParsear
				.descargarNasdaqEstaticosSoloLocal1(entornoDeValidacion);

		int numCasosSinNingunFichero = 0;

		for (EstaticoNasdaqModelo enm : nasdaqEstaticos1) {

			// OBLIGATORIOS (deben haberse descargado)
			String finvizEstaticos = dirBrutoCsv + BrutosUtils.FINVIZ_ESTATICOS + "_" + BrutosUtils.MERCADO_NQ + "_"
					+ enm.symbol + ".csv";
			File fileEstat = new File(finvizEstaticos);

			String yahooFinanceDinamicos = dirBrutoCsv + BrutosUtils.YAHOOFINANCE + "_" + BrutosUtils.MERCADO_NQ + "_"
					+ enm.symbol + ".csv";
			File fileDin = new File(yahooFinanceDinamicos);

			// OPCIONAL: se conocen pocas operaciones de insiders
			String finvizInsiders = dirBrutoCsv + BrutosUtils.FINVIZ_INSIDERS + "_" + BrutosUtils.MERCADO_NQ + "_"
					+ enm.symbol + ".csv";
			File fileInsiders = new File(finvizInsiders); // OPCIONAL

			boolean hay0FicherosObligatorios = !fileEstat.exists() && !fileDin.exists();
			boolean hay1FicherosObligatorio = (fileEstat.exists() && !fileDin.exists())
					|| (!fileEstat.exists() && fileDin.exists());
			boolean hay2FicherosObligatorioSinElOpcional = fileEstat.exists() && fileDin.exists()
					&& !fileInsiders.exists();
			boolean hay2FicherosObligatorioYElOpcional = fileEstat.exists() && fileDin.exists()
					&& fileInsiders.exists();

			if (hay0FicherosObligatorios) {
				// SOLO LO PINTAMOS EN MODO DEBUG, por no ensuciar el log. Aun asi, lo
				// contabilizamos para abortar al final si todos entran en este caso
				MY_LOGGER.debug("Empresa: " + enm.symbol
						+ " --> No conocemos ningun CSV: FZ (estatico) ni YF (dinamico). No procesamos la empresa, pero seguimos.");
				numCasosSinNingunFichero++;
			}
			if (hay1FicherosObligatorio) {
				MY_LOGGER.warn("Empresa: " + enm.symbol
						+ " --> Solo conocemos uno de estos dos CSV: FZ (estatico) ni YF (dinamico). No procesamos la empresa, pero seguimos.");
			}
			if (hay2FicherosObligatorioSinElOpcional) {
				// Muy habitual
				MY_LOGGER.debug("Empresa: " + enm.symbol
						+ " --> Conocemos ambos CSV: FZ (estatico) ni YF (dinamico). Sin operaciones de insiders conocidas");
			}
			if (hay2FicherosObligatorioYElOpcional) {
				// Habitual
				MY_LOGGER.debug("Empresa: " + enm.symbol
						+ " --> Conocemos ambos CSV: FZ (estatico) ni YF (dinamico). Con operaciones de insiders conocidas");
			}

			if (hay2FicherosObligatorioSinElOpcional || hay2FicherosObligatorioYElOpcional) {
				nucleoEmpresa(dirBrutoCsv, enm, fileEstat, fileDin, fileInsiders, desplazamientoAntiguedad);
			}

		}

		if (nasdaqEstaticos1.size() == numCasosSinNingunFichero) {
			MY_LOGGER.debug(
					"No hay ficheros CSV (estaticos ni dinamicos) para ninguna empresa. Hay algun error previo. Saliendo...");
			System.exit(-1);
		}
	}

	/**
	 * @param dirBrutoCsv
	 * @param enm
	 * @param fileEstat
	 * @param fileDin
	 * @param fileDinInsiders          Fichero de OPERACIONES DE INSIDERS (datos
	 *                                 dinámicos, pero la vela más reciente no es de
	 *                                 hoy, sino de hace 2-3 dias como pronto)
	 * @param desplazamientoAntiguedad
	 * @throws IOException
	 * @throws ParseException
	 */
	public static void nucleoEmpresa(String dirBrutoCsv, EstaticoNasdaqModelo enm, File fileEstat, File fileDin,
			File fileDinInsiders, Integer desplazamientoAntiguedad) throws IOException, ParseException {

		MY_LOGGER.info("Empresa: " + enm.symbol);

		// --------- Variables ESTATICAS -------------
		FileReader fr = new FileReader(fileEstat);
		BufferedReader br = new BufferedReader(fr);
		String actual;
		boolean primeraLinea = true;

		String estaticosCabecera = "industria|Insider Own|Quick Ratio|Current Ratio|P/E|Dividend %|Employees|geo|Debt/Eq|LT Debt/Eq|EPS next Y|Earnings|sector|Inst Own|Market Cap";
		String estaticosDatos = "";

		while ((actual = br.readLine()) != null) {
			if (primeraLinea == false) {

				String[] partes = actual.split("\\|");
				// Descartamos: partes[0]=mercado y partes[1]=empresa
				estaticosDatos += partes[2]; // industria
				estaticosDatos += "|" + partes[3];
				estaticosDatos += "|" + partes[4];
				estaticosDatos += "|" + partes[5];
				estaticosDatos += "|" + partes[6];
				estaticosDatos += "|" + partes[7];// Dividend %
				estaticosDatos += "|" + partes[8];
				estaticosDatos += "|" + partes[9];// geo
				estaticosDatos += "|" + partes[10];
				estaticosDatos += "|" + partes[11];
				estaticosDatos += "|" + partes[12];
				estaticosDatos += "|" + partes[13];// presentacion resultados (Earnings date)
				estaticosDatos += "|" + partes[14];// sector
				estaticosDatos += "|" + partes[15];
				estaticosDatos += "|" + partes[16];
			}
			primeraLinea = false;
		}
		br.close();

		// --------- Variables DINAMICAS -------------

		List<String> dinamicosDatos = new ArrayList<String>();

		fr = new FileReader(fileDin);
		br = new BufferedReader(fr);
		primeraLinea = true;

		String dinamicosCabecera = "empresa|antiguedad|mercado|anio|mes|dia|hora|minuto|volumen|high|low|close|open";

		Integer antiguedadDesplazada;

		String dinamicosFilaExtraida;

		while ((actual = br.readLine()) != null) {
			if (primeraLinea == false) {

				String[] partes = actual.split("\\|");

				if ("null".equals(partes[2])) {
					MY_LOGGER.debug("CASO NULO -> " + actual);
					antiguedadDesplazada = Integer.valueOf(0) - desplazamientoAntiguedad;
				} else {
					antiguedadDesplazada = Integer.valueOf(partes[2]) - desplazamientoAntiguedad;
				}

				// Sólo se añade la antigüedad, cuando la resta con el desplazamiento es mayor o
				// igual que cero. Siempre la antigüedad escrita en el CSV comenzará en 0, con
				// el tiempo desplazado

				if (antiguedadDesplazada >= 0) {

					dinamicosFilaExtraida = partes[1];// empresa
					dinamicosFilaExtraida += "|" + antiguedadDesplazada.toString();// antiguedad desplazada
					dinamicosFilaExtraida += "|" + partes[0];// mercado
					dinamicosFilaExtraida += "|" + partes[3]; // anio
					dinamicosFilaExtraida += "|" + partes[4];
					dinamicosFilaExtraida += "|" + partes[5];
					dinamicosFilaExtraida += "|" + partes[6];
					dinamicosFilaExtraida += "|" + partes[7];
					dinamicosFilaExtraida += "|" + partes[8];
					dinamicosFilaExtraida += "|" + partes[9];
					dinamicosFilaExtraida += "|" + partes[10];
					dinamicosFilaExtraida += "|" + partes[11];
					dinamicosFilaExtraida += "|" + partes[12];

					dinamicosDatos.add(dinamicosFilaExtraida);
				}
			}
			primeraLinea = false;
		}
		br.close();

		// ---------- OPERACIONES DE INSIDERS -----------------------
		List<OperacionInsiderFinvizModelo> insidersDatos = new ArrayList<OperacionInsiderFinvizModelo>();
		if (fileDinInsiders.exists()) {

			FileReader fri = new FileReader(fileDinInsiders);
			BufferedReader bri = new BufferedReader(fri);
			String actuali;
			boolean primeraLineai = true;

			while ((actuali = bri.readLine()) != null) {
				if (primeraLineai == false) {
					String[] partes = actuali.split("\\|");

					// TODO NO consideramos otros tipos de operaciones (Option Exercise) porque
					// habría que descargar la URL de la SEC, que seguramente interprete las
					// descargas masivas como un ataque:
					if (partes[1].equalsIgnoreCase(OperacionInsiderFinvizModelo.COMPRA)
							|| partes[1].equalsIgnoreCase(OperacionInsiderFinvizModelo.VENTA)) {
						insidersDatos.add(new OperacionInsiderFinvizModelo(partes[0], partes[1], partes[2]));
					}

				}
				primeraLineai = false;
			}
			bri.close();

			// Para cada día (VELA), metemos NUEVAS FEATURES interpretando la lista de
			// operaciones de insiders de la empresa estudiada
			dinamicosDatos = anhadirOperacionesInsidersEnDinamicos(dinamicosDatos, dinamicosCabecera, insidersDatos);

		} else {
			// Si no hay fichero FI, añadimos columnas vacías a CADA DIA (fila)
			dinamicosDatos = anhadirOperacionesInsidersEnDinamicos(dinamicosDatos, dinamicosCabecera, null);
		}

		// ---------- JUNTOS -----------------------
		String juntos = dirBrutoCsv + BrutosUtils.MERCADO_NQ + "_" + enm.symbol + ".csv";
		MY_LOGGER.debug("Escribiendo CSV juntos en: " + juntos);
		File fjuntos = new File(juntos);
		if (fjuntos.exists()) {
			PrintWriter writer = new PrintWriter(fjuntos);
			writer.print("");// VACIAMOS CONTENIDO
			writer.close();
		}

		// -------- HACEMOS REVERSE DE LOS DATOS, poniendo primero los datos más
		// recientes (optimiza los pasos siguientes en rendimiento, pero no afecta a los
		// modelos porque son casos independientes) ----

		dinamicosDatos = ordenarAscendentePorAntiguedad(dinamicosDatos, 1);

		// ------- ESCRITURA a fichero
		FileOutputStream fos = new FileOutputStream(fjuntos, false);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));

		bw.write(dinamicosCabecera + "|" + CABECERA_OPS_INSIDERS + "|" + estaticosCabecera);
		bw.newLine();
		for (String cad : dinamicosDatos) {
			bw.write(cad + "|" + estaticosDatos);
			bw.newLine();
		}
		bw.close();

	}

	/**
	 * @param dinamicosDatos
	 * @return El primer elemento será el de menor antiguedad.
	 */
	public static List<String> ordenarAscendentePorAntiguedad(List<String> dinamicosDatos, int indiceAntiguedad) {

		Map<Integer, String> mapa = new HashMap<Integer, String>();
		for (String item : dinamicosDatos) {
			String[] partes = item.split("\\|");
			Integer clave = Integer.valueOf(partes[indiceAntiguedad]);
			mapa.put(clave, item);
		}

		// Claves ordenadas del mapa
		List<Integer> clavesOrdenadas = new ArrayList(mapa.keySet());
		Collections.sort(clavesOrdenadas); // orden directo de menor a mayor

		List<String> out = new ArrayList<String>();

		for (Integer clave : clavesOrdenadas) {
			out.add(mapa.get(clave));
		}

		return out;
	}

	/**
	 * Genera NUEVAS FEATURES "ELABORADAS" a la lista de datos dinamicos, basandose
	 * en la info de insiders que tengamos
	 * 
	 * @param dinamicosDatos    Matriz de datos dinamicos de ENTRADA
	 * @param dinamicosCabecera cabecera de la matriz de datos de ENTRADA
	 * @param insidersDatos     Datos de INSIDERS
	 * @return Lista de dinamicos con el contenido de "dinamicosDatos" y habiendo
	 *         añadido las nuevas columnas.
	 * @throws ParseException
	 */
	public static List<String> anhadirOperacionesInsidersEnDinamicos(List<String> dinamicosDatos,
			String dinamicosCabecera, List<OperacionInsiderFinvizModelo> insidersDatos) throws ParseException {

		List<String> out = new ArrayList<String>();

		int posicionAnio = calcularPosicion(dinamicosCabecera, "anio");
		int posicionMes = calcularPosicion(dinamicosCabecera, "mes");
		int posicionDia = calcularPosicion(dinamicosCabecera, "dia");

		List<OperacionInsiderFinvizModelo> operacionesUltimos90dias = new ArrayList<OperacionInsiderFinvizModelo>();
		List<OperacionInsiderFinvizModelo> operacionesUltimos30dias = new ArrayList<OperacionInsiderFinvizModelo>();
		List<OperacionInsiderFinvizModelo> operacionesUltimos15dias = new ArrayList<OperacionInsiderFinvizModelo>();
		List<OperacionInsiderFinvizModelo> operacionesUltimos5dias = new ArrayList<OperacionInsiderFinvizModelo>();
		String flagOperacionesInsiderUltimos90dias = "";
		String flagOperacionesInsiderUltimos30dias = "";
		String flagOperacionesInsiderUltimos15dias = "";
		String flagOperacionesInsiderUltimos5dias = "";

		for (String din : dinamicosDatos) {

			operacionesUltimos90dias.clear();
			operacionesUltimos30dias.clear();
			operacionesUltimos15dias.clear();
			operacionesUltimos5dias.clear();

			String[] dinPartes = din.split("\\|");
			Calendar calDatoDinamico = Calendar.getInstance();
			Integer amdDatoDinamico = Integer
					.valueOf(dinPartes[posicionAnio] + dinPartes[posicionMes] + dinPartes[posicionDia]);
			calDatoDinamico.setTime(sdf.parse(amdDatoDinamico.toString()));

			if (insidersDatos != null && !insidersDatos.isEmpty()) {

				for (OperacionInsiderFinvizModelo op : insidersDatos) {

					long diffInMillies = Math.abs(op.fecha.getTimeInMillis() - calDatoDinamico.getTimeInMillis());
					long diasDiferencia = TimeUnit.DAYS.convert(diffInMillies, TimeUnit.MILLISECONDS);

					if (diasDiferencia >= 0 && Integer.valueOf(sdf.format(op.fecha.getTime())) <= amdDatoDinamico) {
						if (diasDiferencia >= 0 && diasDiferencia <= NOVENTA) {
							operacionesUltimos90dias.add(op);
						}
						if (diasDiferencia >= 0 && diasDiferencia <= TREINTA) {
							operacionesUltimos30dias.add(op);
						}
						if (diasDiferencia >= 0 && diasDiferencia <= QUINCE) {
//							MY_LOGGER.info("Operacion_insider=" + sdf.format(op.fecha.getTime()) + " Dato_dinamico="
//									+ sdf.format(calDatoDinamico.getTime()) + " -->diasDiferencia=" + diasDiferencia);
							operacionesUltimos15dias.add(op);
						}
						if (diasDiferencia >= 0 && diasDiferencia <= CINCO) {
							operacionesUltimos5dias.add(op);
						}
					}

				}

				flagOperacionesInsiderUltimos90dias = sumarItemsYcalcularFlag(operacionesUltimos90dias, amdDatoDinamico,
						NOVENTA);
				flagOperacionesInsiderUltimos30dias = sumarItemsYcalcularFlag(operacionesUltimos30dias, amdDatoDinamico,
						TREINTA);
				flagOperacionesInsiderUltimos15dias = sumarItemsYcalcularFlag(operacionesUltimos15dias, amdDatoDinamico,
						QUINCE);
				flagOperacionesInsiderUltimos5dias = sumarItemsYcalcularFlag(operacionesUltimos5dias, amdDatoDinamico,
						CINCO);
			}

			// Haya datos de insiders o no, añadimos las columnas dinamicas, rellenas o
			// vacías
			out.add(din + "|" + flagOperacionesInsiderUltimos90dias + "|" + flagOperacionesInsiderUltimos30dias + "|"
					+ flagOperacionesInsiderUltimos15dias + "|" + flagOperacionesInsiderUltimos5dias);
		}

		return out;
	}

	/**
	 * @param cabecera
	 * @param nombreColumnaBuscado
	 * @return
	 */
	public static int calcularPosicion(String cabecera, String nombreColumnaBuscado) {

		String[] partes = cabecera.split("\\|");
		int out = -1;
		int num = 0;
		for (String cad : partes) {
			if (cad.equalsIgnoreCase(nombreColumnaBuscado)) {
				out = num;
				break;
			}
			num++;
		}
		return out;
	}

	/**
	 * Suma los importes de todos los elementos de la lista. Con ello decide el flag
	 * de salida, indicando si la suma son ventas o compras.
	 * 
	 * @param lista
	 * @return Numero que indica "-1"=ventas, "1"=compras. En caso de que sumen 0 o
	 *         que sea desconocido, devuelve cadena vacía.
	 */
	public static String sumarItemsYcalcularFlag(List<OperacionInsiderFinvizModelo> lista, Integer amdReferencia,
			int diasPeriodo) {
		Long suma = 0L;
		String out = "";

		for (OperacionInsiderFinvizModelo op : lista) {
			suma += op.importe;
		}

		if (suma > 0L) {
			out = "1"; // COMPRAS
		}
//		else if (suma < 0L) {
//			out = "-1";// VENTAS
//		}
		return out;
	}

}
