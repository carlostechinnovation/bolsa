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

		nucleo();
		MY_LOGGER.info("FIN");

	}

	/**
	 * @throws IOException
	 */
	public static void nucleo() throws IOException {

		List<EstaticoNasdaqModelo> nasdaqEstaticos1 = EstaticosNasdaqDescargarYParsear
				.descargarNasdaqEstaticosSoloLocal1();

		for (EstaticoNasdaqModelo enm : nasdaqEstaticos1) {

			String finvizEstaticos = BrutosUtils.DIR_BRUTOS_CSV + BrutosUtils.FINVIZ + "_" + BrutosUtils.MERCADO_NQ
					+ "_" + enm.symbol + ".csv";
			File fileEstat = new File(finvizEstaticos);

			String yahooFinanceDinamicos = BrutosUtils.DIR_BRUTOS_CSV + BrutosUtils.YAHOOFINANCE + "_"
					+ BrutosUtils.MERCADO_NQ + "_" + enm.symbol + ".csv";
			File fileDin = new File(yahooFinanceDinamicos);

			if (fileEstat.exists() && fileDin.exists()) {
				nucleoEmpresa(enm, fileEstat, fileDin);
			}

		}
	}

	/**
	 * @param enm
	 * @param fileEstat
	 * @param fileDin
	 * @throws IOException
	 */
	public static void nucleoEmpresa(EstaticoNasdaqModelo enm, File fileEstat, File fileDin) throws IOException {

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

		while ((actual = br.readLine()) != null) {
			if (primeraLinea == false) {

				String[] partes = actual.split("\\|");
				String dinamicosFilaExtraida = partes[1];// empresa
				dinamicosFilaExtraida += "|" + partes[2];// antiguedad
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
			primeraLinea = false;
		}
		br.close();

		// ---------- JUNTOS -----------------------
		String juntos = BrutosUtils.DIR_BRUTOS_CSV + BrutosUtils.MERCADO_NQ + "_" + enm.symbol + ".csv";
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
