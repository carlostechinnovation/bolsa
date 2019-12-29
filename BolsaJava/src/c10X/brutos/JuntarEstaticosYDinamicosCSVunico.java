package c10X.brutos;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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

	private static JuntarEstaticosYDinamicosCSVunico instancia = null;

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
	 */
	public static void main(String[] args) throws IOException {

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
	 * @throws IOException
	 */
	public static void nucleo(String dirBrutoCsv, Integer desplazamientoAntiguedad, final Integer entornoDeValidacion)
			throws IOException {

		List<EstaticoNasdaqModelo> nasdaqEstaticos1 = EstaticosNasdaqDescargarYParsear
				.descargarNasdaqEstaticosSoloLocal1(entornoDeValidacion);

		for (EstaticoNasdaqModelo enm : nasdaqEstaticos1) {

			String finvizEstaticos = dirBrutoCsv + BrutosUtils.FINVIZ + "_" + BrutosUtils.MERCADO_NQ + "_" + enm.symbol
					+ ".csv";
			File fileEstat = new File(finvizEstaticos);

			String yahooFinanceDinamicos = dirBrutoCsv + BrutosUtils.YAHOOFINANCE + "_" + BrutosUtils.MERCADO_NQ + "_"
					+ enm.symbol + ".csv";
			File fileDin = new File(yahooFinanceDinamicos);

			if (fileEstat.exists() && fileDin.exists()) {
				nucleoEmpresa(dirBrutoCsv, enm, fileEstat, fileDin, desplazamientoAntiguedad);
			}

		}
	}

	/**
	 * @param enm
	 * @param fileEstat
	 * @param fileDin
	 * @throws IOException
	 */
	public static void nucleoEmpresa(String dirBrutoCsv, EstaticoNasdaqModelo enm, File fileEstat, File fileDin,
			Integer desplazamientoAntiguedad) throws IOException {

		// --------- Variables ESTATICAS -------------
		FileReader fr = new FileReader(fileEstat);
		BufferedReader br = new BufferedReader(fr);
		String actual;
		boolean primeraLinea = true;

		String estaticosCabecera = "Insider Own|Debt/Eq|P/E|Dividend %|Employees|Inst Own|Market Cap";
		String estaticosDatos = "";

		if (enm.symbol.equals("AAPL")) {
			int x = 0;
		}

		while ((actual = br.readLine()) != null) {
			if (primeraLinea == false) {

				String[] partes = actual.split("\\|");
				estaticosDatos += partes[2];
				estaticosDatos += "|" + partes[3];
				estaticosDatos += "|" + partes[4];
				estaticosDatos += "|" + partes[5];
				estaticosDatos += "|" + partes[6];
				estaticosDatos += "|" + partes[7];
				estaticosDatos += "|" + partes[8];
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

				antiguedadDesplazada = Integer.valueOf(partes[2]) - desplazamientoAntiguedad;

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

		// ---------- JUNTOS -----------------------
		String juntos = dirBrutoCsv + BrutosUtils.MERCADO_NQ + "_" + enm.symbol + ".csv";
		File fjuntos = new File(juntos);
		if (fjuntos.exists()) {
			PrintWriter writer = new PrintWriter(fjuntos);
			writer.print("");// VACIAMOS CONTENIDO
			writer.close();
		}

		// -------- HACEMOS REVERSE DE LOS DATOS, poniendo primero los datos más
		// recientes (optimiza los pasos siguientes en rendimiento, pero no afecta a los
		// modelos porque son casos independientes) ----
		Collections.reverse(dinamicosDatos);

//------- ESCRITURA a fichero
		FileOutputStream fos = new FileOutputStream(fjuntos, false);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));

		bw.write(dinamicosCabecera + "|" + estaticosCabecera);
		bw.newLine();
		for (String cad : dinamicosDatos) {
			bw.write(cad + "|" + estaticosDatos);
			bw.newLine();
		}
		bw.close();

	}

}
