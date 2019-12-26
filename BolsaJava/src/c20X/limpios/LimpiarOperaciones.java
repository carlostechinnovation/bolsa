package c20X.limpios;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c10X.brutos.BrutosUtils;
import c30x.elaborados.construir.Estadisticas;

public class LimpiarOperaciones implements Serializable {

	static Logger MY_LOGGER = Logger.getLogger(LimpiarOperaciones.class);
	private static LimpiarOperaciones instancia = null;

	private LimpiarOperaciones() {
		super();
	}

	public static LimpiarOperaciones getInstance() {
		if (instancia == null)
			instancia = new LimpiarOperaciones();

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

		String directorioIn = BrutosUtils.DIR_BRUTOS_CSV; // DEFAULT
		String directorioOut = LimpiosUtils.DIR_LIMPIOS; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 2) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			directorioIn = args[0];
			directorioOut = args[1];
		}

		operacionesLimpieza(directorioIn, directorioOut);

		MY_LOGGER.info("FIN");
	}

	/**
	 * @param directorioIn
	 * @param directorioOut
	 * @throws IOException
	 */
	public static void operacionesLimpieza(String directorioIn, String directorioOut) throws IOException {

		List<String> ficherosEntrada = new ArrayList<String>();
		BrutosUtils.encontrarFicherosEnCarpeta(".*csv", new File(directorioIn), ficherosEntrada);

		for (String pathFichero : ficherosEntrada) {
			operacionesLimpiezaFicheroYGuardar(pathFichero, directorioOut);
		}

	}

	/**
	 * @param pathFicheroIn
	 * @param directorioOut
	 * @throws IOException
	 */
	public static void operacionesLimpiezaFicheroYGuardar(String pathFicheroIn, String directorioOut)
			throws IOException {

		List<List<String>> datos = LimpiosUtils.leerFicheroHaciaListasDeColumnas(pathFicheroIn);

		Estadisticas estad;
		boolean primeraFila = true;

		Map<String, String> columnaAnomaliaCorreccion = new HashMap<String, String>();

		for (List<String> columna : datos) {

			estad = new Estadisticas();

			for (String item : columna) {
				if (primeraFila) {
					MY_LOGGER.info("Analizando columna: " + item);
					primeraFila = false;
				} else {

					try {
						estad.addValue(Double.parseDouble(item));

					} catch (Exception e) {
						// MY_LOGGER.warn("Problema al castear dato a double ->" + item + " -->
						// Sigo...");
					}

				}
			}

			double mediana = estad.getPercentile(50);
			double medianaPor7 = mediana * 7.0F;
			double medianaEntre20 = mediana / 20.0F;
			for (double v : estad.getValues()) {

				// CORREGIMOS LOS OUTLIERS QUE SUBEN/BAJAN MUCHISIMO, dándoles un valor
				// armonizado (no tan alto ni tan bajo), para que el modelo matematico no se
				// vuelva loco
				if (v > medianaPor7) {
					columnaAnomaliaCorreccion.put(columna.get(0) + "#" + String.valueOf(v),
							String.valueOf(medianaPor7)); // COLUMNA-DatoAnomalo
				} else if (v < medianaEntre20) {
					columnaAnomaliaCorreccion.put(columna.get(0) + "#" + String.valueOf(v),
							String.valueOf(medianaEntre20)); // COLUMNA-DatoAnomalo
				}
			}

			// TODO Valores vacios (missing values): borramos esa fila o imputamos valores
			List<Integer> indicesMV = new ArrayList<Integer>();

			// TODO Temporalmente copio los limpios en elaborados

		}

		// TODO para cada columna, corregir el dato de las anomalías detectadas
		int y = 0;
	}

}
