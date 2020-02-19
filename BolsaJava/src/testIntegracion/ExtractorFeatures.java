/**
 * 
 */
package testIntegracion;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c20X.limpios.LimpiosUtils;
import c30x.elaborados.construir.GestorFicheros;

/**
 * Dado un fichero y una lista de features (columnas), muestra solo las columnas
 * seleccionadas
 *
 */
public class ExtractorFeatures implements Serializable {

	private static final long serialVersionUID = 1L;

	static Logger MY_LOGGER = Logger.getLogger(ExtractorFeatures.class);
	private static ExtractorFeatures instancia = null;

	private ExtractorFeatures() {
		super();
	}

	public static ExtractorFeatures getInstance() {
		if (instancia == null)
			instancia = new ExtractorFeatures();

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

		String featuresElegidas = ""; // DEFAULT
		String fichero_entrada = ""; // DEFAULT
		String fichero_salida = ""; // DEFAULT
		Long num_filas = 10L; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 4) {
			MY_LOGGER.error("Parametros de entrada incorrectos (se estan recibiendo " + args.length + "parametros)!!");
			System.exit(-1);
		} else {
			featuresElegidas = args[0];
			fichero_entrada = args[1];
			fichero_salida = args[2];
			num_filas = Long.valueOf(args[3]);
		}

		extraer(featuresElegidas, fichero_entrada, fichero_salida, num_filas);

		MY_LOGGER.info("FIN");
	}

	/**
	 * @param featuresElegidas
	 * @param fichero_entrada
	 * @param fichero_salida
	 * @param num_filas
	 * @throws IOException
	 */
	public static void extraer(String featuresElegidas, String fichero_entrada, String fichero_salida, Long num_filas)
			throws IOException {

		MY_LOGGER.info("featuresElegidas=" + featuresElegidas);
		MY_LOGGER.info("fichero_entrada=" + fichero_entrada);
		MY_LOGGER.info("fichero_salida=" + fichero_salida);
		MY_LOGGER.info("num_filas=" + num_filas);

		List<List<String>> datos = LimpiosUtils.leerFicheroHaciaListasDeColumnas(fichero_entrada, num_filas);

		// Saber las features elegidas
		List<String> featuresElegidasList = new ArrayList<String>();
		featuresElegidasList.addAll(Arrays.asList(featuresElegidas.split("\\|")));

		// Localizar los indices de las features en la matriz de datos
		List<String> listaIndiceFeature = new ArrayList<String>();

		for (String featureIterada : featuresElegidasList) {

			for (int i = 0; i < datos.size(); i++) {
				if (datos.get(i).get(0).equals(featureIterada)) {
					listaIndiceFeature.add(i + "|" + featureIterada);
					MY_LOGGER.info(featureIterada);
					break;
				}
			}

		}

		// Extraer datos de las features deseadas
		List<List<String>> datosExtraidos = new ArrayList<List<String>>();
		for (String indiceFeature : listaIndiceFeature) {

			String[] partes = indiceFeature.split("\\|");

			int sadf = 0;
			List<String> todosLosDatosDeFeature = datos.get(Integer.valueOf(partes[0]));
			datosExtraidos.add(todosLosDatosDeFeature);
		}

		Map<Integer, String> datosPorFilasEscribibles = new HashMap<Integer, String>();

		boolean primeraColumna = true;
		for (List<String> columna : datosExtraidos) {

			if (primeraColumna) {
				for (int j = 0; j < columna.size(); j++) {
					datosPorFilasEscribibles.put(j, "");
				}
			}

			for (int i = 0; i < columna.size(); i++) {
				String separador = primeraColumna ? "" : "|";
				datosPorFilasEscribibles.put(i, datosPorFilasEscribibles.get(i) + separador + columna.get(i));
			}

			primeraColumna = false;
		}

		// ESCRIBIR a FICHERO de salida
		File fileOut = new File(fichero_salida);
		Collection<String> filasAEscribir = datosPorFilasEscribibles.values();
		GestorFicheros.crearFicheroyEscribirCadenas(filasAEscribir, fileOut.getAbsolutePath());

	}

}
