package c20X.limpios;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c10X.brutos.BrutosUtils;
import c30x.elaborados.construir.GestorFicheros;

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
		String p_inicio = LimpiosUtils.P_INICIO; // DEFAULT
		String p_fin = LimpiosUtils.P_FIN; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 4) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			directorioIn = args[0];
			directorioOut = args[1];
			p_inicio = args[2];
			p_fin = args[3];
		}

		operacionesLimpieza(directorioIn, directorioOut, p_inicio, p_fin);

		MY_LOGGER.info("FIN");
	}

	/**
	 * @param directorioIn
	 * @param directorioOut
	 * @param p_inicio      Inicio del periodo. Sólo aplicable al pasado.
	 * @param p_fin         Fin del periodo de entrenamiento. Sólo aplicable al
	 *                      pasado.
	 * @throws IOException
	 */
	public static void operacionesLimpieza(String directorioIn, String directorioOut, String p_inicio, String p_fin)
			throws IOException {

		List<String> ficherosEntrada = new ArrayList<String>();
		BrutosUtils.encontrarFicherosEnCarpeta(".*csv", new File(directorioIn), ficherosEntrada);

		for (String pathFichero : ficherosEntrada) {
			operacionesLimpiezaFicheroYGuardar(pathFichero, directorioOut, p_inicio, p_fin);
		}

	}

	/**
	 * @param pathFicheroIn
	 * @param directorioOut
	 * @param p_inicio
	 * @param p_fin
	 * @throws IOException
	 */
	public static void operacionesLimpiezaFicheroYGuardar(String pathFicheroIn, String directorioOut, String p_inicio,
			String p_fin) throws IOException {

		List<List<String>> datos = LimpiosUtils.leerFicheroHaciaListasDeColumnas(pathFicheroIn);

		Integer p_inicio_int = Integer.valueOf(p_inicio);
		Integer p_fin_int = Integer.valueOf(p_fin);

		// ------------- EXTRAER ANIO-MES-DIA de los datos -----
		List<String> anios = new ArrayList<String>();
		List<String> meses = new ArrayList<String>();
		List<String> dias = new ArrayList<String>();

		for (List<String> columna : datos) {

			if (columna.get(0).equals(BrutosUtils.COL_ANIO)) {
				anios = columna;
			} else if (columna.get(0).equals(BrutosUtils.COL_MES)) {
				meses = columna;
			} else if (columna.get(0).equals(BrutosUtils.COL_DIA)) {
				dias = columna;
			}
		}

		// ------------ APLICAR PERIODO -------------
		List<Integer> indicesFilasSeleccionadas = new ArrayList<Integer>();

		for (int indice = 0; indice < anios.size(); indice++) {

			if (indice == 0) {
				indicesFilasSeleccionadas.add(0);

			} else if (indice > 0) {
				String aniomesdia = anios.get(indice) + meses.get(indice) + dias.get(indice);

				Integer aniomesdiaInt = Integer.parseInt(aniomesdia);

				if (p_inicio_int <= aniomesdiaInt.intValue() && aniomesdiaInt.intValue() <= p_fin_int) {
					indicesFilasSeleccionadas.add(indice);
				}

			}

		}

		// --------- EXTRAER LAS FILAS que están dentro del periodo --
		Map<Integer, String> datosDentroDePeriodo = new HashMap<Integer, String>();
		for (int i : indicesFilasSeleccionadas) {
			datosDentroDePeriodo.put(i, "");
		}

		boolean primeraColumna = true;
		for (List<String> columna : datos) {

			for (int i : indicesFilasSeleccionadas) {

				String separador = primeraColumna ? "" : "|";

				datosDentroDePeriodo.put(i, datosDentroDePeriodo.get(i) + separador + columna.get(i));

			}

			primeraColumna = false;
		}

		// ---------- ESCRIBIR a FICHERO ----
		File fileIn = new File(pathFicheroIn);
		String fileNameIn = fileIn.getName();
		Collection<String> filasAEscribir = datosDentroDePeriodo.values();
		GestorFicheros.crearFicheroyEscribirCadenas(filasAEscribir, directorioOut + fileNameIn);

	}

}
