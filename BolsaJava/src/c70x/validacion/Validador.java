package c70x.validacion;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c30x.elaborados.construir.ConstructorElaborados;
import c30x.elaborados.construir.ElaboradosUtils;
import c40X.subgrupos.SubgruposUtils;

public class Validador implements Serializable {

	private static final long serialVersionUID = 1L;

	static Logger MY_LOGGER = Logger.getLogger(Validador.class);

	private static Validador instancia = null;

	private static String DEFINICION_PREDICCION = "COMPLETO_PREDICCION.csv";

	private Validador() {
		super();
	}

	public static Validador getInstance() {
		if (instancia == null)
			instancia = new Validador();

		return instancia;
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws Exception {

		Object appendersAcumulados = Logger.getRootLogger().getAllAppenders();
		if (appendersAcumulados instanceof NullEnumeration) {
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(PatternLayout.TTCC_CONVERSION_PATTERN)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

		Integer velasRetroceso = ValidadorUtils.VELAS_RETROCESO;
		String pathValidacion = ValidadorUtils.PATH_VALIDACION; // DEFAULT
		Integer S = ElaboradosUtils.S; // DEFAULT
		Integer X = ElaboradosUtils.X; // DEFAULT
		Integer R = ElaboradosUtils.R; // DEFAULT
		Integer M = ElaboradosUtils.M; // DEFAULT
		Integer F = ElaboradosUtils.F; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 7) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			velasRetroceso = Integer.valueOf(args[0]);
			pathValidacion = args[1];
			S = Integer.valueOf(args[2]);
			X = Integer.valueOf(args[3]);
			R = Integer.valueOf(args[4]);
			M = Integer.valueOf(args[5]);
			F = Integer.valueOf(args[6]);
		}

		analizarPrediccion(velasRetroceso, pathValidacion, S, X, R, M, F);

		MY_LOGGER.info("FIN");
	}

	/**
	 * Tanto la predicción desfasada como la actual, estarán en la misma carpeta, y
	 * se compararán por subgrupos. Se obtienen las estadísticas de aciertos/fallos.
	 * Para diferenciar cada fichero por subgrupo, comienzan por la cantidad de
	 * desfase (que será 0 o velasRetroceso).
	 * 
	 * @param velasRetroceso Tiempo en velas donde simulamos la predicción.
	 * @param pathValidacion Contiene los datos a tiempo actual, con los targets
	 *                       rellenos, con los que se validará (comenzará por )
	 * @param S              Subida mínima
	 * @param X              Dentro de X velas
	 * @param R              Caída máxima
	 * @param M              En todas las M velas posteriores
	 * @throws IOException
	 */
	public static void analizarPrediccion(final Integer velasRetroceso, final String pathValidacion, final Integer S,
			final Integer X, final Integer R, final Integer M, final Integer F) throws IOException {

		Predicate<Path> filtroFicheroPrediccion = s -> s.getFileName().toString().contains(DEFINICION_PREDICCION);

		Predicate<Path> filtroPredichos = p -> p.getFileName().toString().startsWith(velasRetroceso + "_");
		Predicate<Path> filtroValidaciones = p -> p.getFileName().toString().startsWith("0_");

		// Buscar ficheros: PREDICHO (días atrás) y REAL (hoy)
		List<Path> ficherosPredichos = Files.walk(Paths.get(pathValidacion), 2)
				.filter(filtroFicheroPrediccion.and(filtroPredichos)).collect(Collectors.toList());
		List<Path> ficherosValidacion = Files.walk(Paths.get(pathValidacion), 2)
				.filter(filtroFicheroPrediccion.and(filtroValidaciones)).collect(Collectors.toList());

		// Recorro cada subgrupo predicho (a validar). Para cada uno, cogeré el fichero
		// equivalente en el otro bloque de carpetas de de validación, dentro de su
		// subgrupo
		List<String> filasPredichas, filasValidacion;

		ValidadorComparacionSubgrupo vc = new ValidadorComparacionSubgrupo();

		for (Path predicho : ficherosPredichos) {

			MY_LOGGER.info("Path predicho: " + predicho.toString());

			// Reinicio contadores de Subgrupo
			vc = new ValidadorComparacionSubgrupo();

			// Resultados subgrupo
			HashMap<String, Integer> resultadosSubgrupo = new HashMap<String, Integer>();

			for (Path validacion : ficherosValidacion) {

				MY_LOGGER.info("Path validacion: " + validacion.toString());

				// Se asume que la estructura del nombre de cada fichero es:
				// <retroceso>_SG_<numeroSubgrupo>_COMPLETO_PREDICCION.csv

				String nombrePredicho = predicho.getFileName().toString();
				String finalnombrePredicho = nombrePredicho.substring(nombrePredicho.indexOf("_"));

				String nombreValidacion = validacion.getFileName().toString();
				String finalnombreValidacion = nombreValidacion.substring(nombreValidacion.indexOf("_"));

				if (finalnombrePredicho.equals(finalnombreValidacion)) {

					MY_LOGGER.info("Se compara el fichero de PREDICCIÓN (" + nombrePredicho
							+ ") con el que tiene la info REAL de contraste/validación (" + nombreValidacion + "): ");

					// Aquí tenemos ya el fichero predicho y el de validación, dentro de un mismo
					// subgrupo.
					// 1.1. Para cada fila 0 predicha (que sólo habrá una para cada empresa), busco
					// en el fichero de validación la fila igual en fecha y en empresa.
					// AQUÍ comparo si el target coincide, y lo apunto. Esto es RENTABILIDAD
					// ACIERTOS, y se mediría sólo el número de aciertos, pero no el rendimiento en
					// precios.
					// RENTABILIDAD PRECIOS:
					// Busco la fila desplazada según lo definido por el target (S, X, R, M) dentro
					// de las filas de validación.
					// Saco la rentabilidad REAL de subida, todo obtenido de las filas de
					// validación. Lo apunto.

					filasPredichas = Files.readAllLines(predicho);
					filasValidacion = Files.readAllLines(validacion);

					resultadosSubgrupo = compararPredichasYRealesDeUnSubgrupo(filasPredichas, filasValidacion, vc, S, X,
							R, M, F);

				}
			}

			if (resultadosSubgrupo != null && !resultadosSubgrupo.isEmpty()) {

				int numAciertos = resultadosSubgrupo.get("aciertos");
				int numFallos = resultadosSubgrupo.get("fallos");
				int numNulos = resultadosSubgrupo.get("nulos");

				// PORCENTAJE ACIERTOS por SUBGRUPO
				MY_LOGGER.info("COBERTURA - Porcentaje aciertos subgrupo " + predicho.getFileName() + ": " +

						String.valueOf(numAciertos) + " / " + String.valueOf(numAciertos + numFallos + numNulos) + " = "
						+ (numAciertos / (numAciertos + numFallos + numNulos)) * 100
						+ "%, siendo acierto si PREDICHO=1 cumplió las condiciones en la realidad en la fecha predicha");

//				// RENTABILIDAD PRECIOS por SUBGRUPO
//				MY_LOGGER.info("RENTABILIDAD - Porcentaje medio de SUBIDA del precio en subgrupo "
//						+ predicho.getFileName() + ": " + vc.performanceClose.getMean() * 100 + " % de "
//						+ vc.totalTargetUnoEnSubgrupo + " elementos PREDICHOS a TARGET=1");
//				MY_LOGGER.info("RENTABILIDAD - Desviación estándar de SUBIDA del precio en subgrupo "
//						+ predicho.getFileName() + ": " + vc.performanceClose.getStandardDeviation() + " para "
//						+ vc.totalTargetUnoEnSubgrupo + " elementos PREDICHOS a TARGET=1");
//
//				// ANALISIS
//				MY_LOGGER.info("ANALISIS - ACERTADOS - Porcentaje medio de SUBIDA del precio en subgrupo "
//						+ predicho.getFileName() + ": " + vc.performanceCloseAcertados.getMean() * 100 + " % de "
//						+ vc.totalTargetUnoEnSubgrupo + " elementos PREDICHOS a TARGET=1");
//				MY_LOGGER.info("ANALISIS - ACERTADOS - Desviación estándar de SUBIDA del precio en subgrupo "
//						+ predicho.getFileName() + ": " + vc.performanceCloseAcertados.getStandardDeviation() + " para "
//						+ vc.totalTargetUnoEnSubgrupo + " elementos PREDICHOS a TARGET=1");
//
//				MY_LOGGER.info("ANALISIS - FALLADOS - Porcentaje medio de SUBIDA del precio en subgrupo "
//						+ predicho.getFileName() + ": " + vc.performanceCloseFallados.getMean() * 100 + " % de "
//						+ vc.totalTargetUnoEnSubgrupo + " elementos PREDICHOS a TARGET=1");
//				MY_LOGGER.info("ANALISIS - FALLADOS - Desviación estándar de SUBIDA del precio en subgrupo "
//						+ predicho.getFileName() + ": " + vc.performanceCloseFallados.getStandardDeviation() + " para "
//						+ vc.totalTargetUnoEnSubgrupo + " elementos PREDICHOS a TARGET=1");

			} else {
				MY_LOGGER.error(
						"No hemos podido calcular la comparativa para el subgrupo cuyo fichero de prediccion fue: "
								+ predicho.toString());
			}

		}

	}

	/**
	 * 
	 * @param dir
	 * @return
	 * @throws IOException
	 */
	public List<String> listFilesUsingDirectoryStream(String dir) throws IOException {
		List<String> fileList = new ArrayList<String>();
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(Paths.get(dir))) {
			for (Path path : stream) {
				if (!Files.isDirectory(path)) {
					fileList.add(path.getFileName().toString());
				}
			}
		}
		return fileList;
	}

	/**
	 * @param filasPredichas
	 * @param filasValidacion
	 * @param vc
	 * @param S
	 * @param X
	 * @param R
	 * @param M
	 */
	public static HashMap<String, Integer> compararPredichasYRealesDeUnSubgrupo(List<String> filasPredichas,
			List<String> filasValidacion, ValidadorComparacionSubgrupo vc, final Integer S, final Integer X,
			final Integer R, final Integer M, final Integer F) {

		int numAciertos = 0;
		int numFallos = 0;
		int numNulos = 0;

		if (filasPredichas != null && !filasPredichas.isEmpty() && filasPredichas != null
				&& !filasValidacion.isEmpty()) {

			String cabeceraValidacion = filasValidacion.get(0);

			// Elimino las filas de cabecera
			filasPredichas.remove(0);
			filasValidacion.remove(0);

			for (String filaPredicha : filasPredichas) {

				// VELA 0 en la lista PREDICHAS
				String antiguedadPredicha = filaPredicha.substring(
						SubgruposUtils.indiceDeAparicion("|".charAt(0), 1, filaPredicha) + 1,
						SubgruposUtils.indiceDeAparicion("|".charAt(0), 2, filaPredicha));
				boolean antiguedadPredichaVelaCero = antiguedadPredicha.equals("0");
				vc.indicePredicha = SubgruposUtils.indiceDeAparicion("|".charAt(0), 8, filaPredicha);
				vc.fechaPredicha = filaPredicha.substring(0, vc.indicePredicha);
				vc.empresaPredicha = vc.fechaPredicha.substring(0, vc.fechaPredicha.indexOf("|"));
				vc.fechaPredicha = SubgruposUtils.recortaPrimeraParteDeString("|".charAt(0), 3, vc.fechaPredicha);

				// Sólo se analizarán las predicciones. Es decir, las filas con antigüedad=0 en
				// predichas
				if (antiguedadPredichaVelaCero) {

					Boolean outCaso = compararFilaPredichaYListaFilasValidacion(filaPredicha, filasValidacion,
							cabeceraValidacion, vc, S, X, R, M, F);

					if (outCaso == null) {
						numNulos++;
					} else if (outCaso.booleanValue()) {
						numAciertos++;
					} else {
						numFallos++;
					}

				}

			}

		} else {
			MY_LOGGER.warn("El fichero de PREDICCIÓN o el fichero contraste/validación es nulo o está vacío.");
		}

		HashMap<String, Integer> out = new HashMap<String, Integer>();
		out.put("aciertos", numAciertos);
		out.put("fallos", numFallos);
		out.put("nulos", numNulos);

		return out;

	}

	/**
	 * @param filaPredicha       Fila en fichero PREDICHAS en la FECHA en la que
	 *                           hicimos la prediccion para adivinar el instante con
	 *                           antigüedad=(FECHA - X)
	 * @param filasValidacion    Lista de velas de validacion, donde aparece el
	 *                           precio real para ese instante con antigüedad=(FECHA
	 *                           - X) y también el resto de velas de alrededor, que
	 *                           permiten COMPROBAR SI SE CUMPLEN EL RESTO DE
	 *                           CONDICIONES PARA CALCULAR EL TARGET (no solo basta
	 *                           mirar el precio de cierre de ese día).
	 * @param cabeceraValidacion
	 * @param vc
	 * @param S
	 * @param X
	 * @param R
	 * @param M
	 * @param F
	 * @return TRUE si acierto. FALSE si fallo. NULL si no se puede comparar.
	 */
	public static Boolean compararFilaPredichaYListaFilasValidacion(String filaPredicha, List<String> filasValidacion,
			String cabeceraValidacion, ValidadorComparacionSubgrupo vc, final Integer S, final Integer X,
			final Integer R, final Integer M, final Integer F) {

		String empresa = null;
		List<String> filasValidacionEmpresa = new ArrayList<String>();

		for (String filaValidacion : filasValidacion) {

			vc.antiguedadValidacion = filaValidacion.substring(
					SubgruposUtils.indiceDeAparicion("|".charAt(0), 1, filaValidacion) + 1,
					SubgruposUtils.indiceDeAparicion("|".charAt(0), 2, filaValidacion));
			vc.indiceValidacion = SubgruposUtils.indiceDeAparicion("|".charAt(0), 8, filaValidacion);
			vc.fechaValidacion = filaValidacion.substring(0, vc.indiceValidacion);
			vc.empresaValidacion = vc.fechaValidacion.substring(0, vc.fechaValidacion.indexOf("|"));
			vc.fechaValidacion = SubgruposUtils.recortaPrimeraParteDeString("|".charAt(0), 3, vc.fechaValidacion);

			boolean mismaEmpresa = vc.empresaPredicha.equals(vc.empresaValidacion);

			if (mismaEmpresa) {
				empresa = vc.empresaPredicha;
				filasValidacionEmpresa.add(filaValidacion);
			}
		}

		Boolean out = null;
		if (empresa != null && !empresa.isEmpty()) {
			out = compararFilaPredichaYListaFilasValidacionMismaEmpresa(empresa, filaPredicha, filasValidacionEmpresa,
					cabeceraValidacion, vc, S, X, R, M, F);
		}

		return out;
	}

	/**
	 * @param empresa         Empresa analizada
	 * @param filaPredicha    Vela en tiempo T2 predicha para la empresa pasada como
	 *                        parámetro
	 * @param filasValidacion Velas de validacion (donde ya se sabe el precio real
	 *                        que ha sucedido) que servirán para comprobar si se
	 *                        cumplieron realmente las condiciones fijadas por el
	 *                        target para el instante futuro T3.
	 * @param vc
	 * @param S
	 * @param X
	 * @param R
	 * @param M               Duracion del periodo [t2,t3]
	 * @param F
	 * @return TRUE si acierto. FALSE si fallo. NULL si no se puede comparar.
	 */
	public static Boolean compararFilaPredichaYListaFilasValidacionMismaEmpresa(String empresa, String filaPredicha,
			List<String> filasValidacion, String cabeceraValidacion, ValidadorComparacionSubgrupo vc, final Integer S,
			final Integer X, final Integer R, final Integer M, final Integer F) {

		String[] nombresColumnas = cabeceraValidacion.split("\\|");

		String[] partesPredicha = filaPredicha.split("\\|");

		if (nombresColumnas.length != partesPredicha.length) {
			// Caso especial en el que el ultimo elemento está vacío. Lo relleno con un
			// guion para poder cortar
			partesPredicha = filaPredicha.concat("-").split("\\|");
		} else {
			partesPredicha = filaPredicha.split("\\|");
		}

		String tiempoVelaT2Predicha = partesPredicha[3] + "|" + partesPredicha[4] + "|" + partesPredicha[5] + "|"
				+ partesPredicha[6] + "|" + partesPredicha[7];
		String targetPredicho = null;
		for (int j = 0; j < nombresColumnas.length; j++) {
			if (nombresColumnas[j].equals("TARGET_PREDICHO")) {
				targetPredicho = partesPredicha[j];
			}

		}

		HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada = new HashMap<Integer, HashMap<String, String>>();
		Integer indiceVelaT2Validacion = null;
		int i = 0;
		String[] partesFilaValidacion;
		HashMap<String, String> aux;

		for (String filaValidacion : filasValidacion) {

			partesFilaValidacion = filaValidacion.split("\\|");

			if (nombresColumnas.length != partesFilaValidacion.length) {
				// Caso especial en el que el ultimo elemento está vacío. Lo relleno con un
				// guion para poder cortar
				partesFilaValidacion = filaValidacion.concat("-").split("\\|");
			} else {
				partesFilaValidacion = filaValidacion.split("\\|");
			}

			aux = new HashMap<String, String>();

			for (int j = 0; j < nombresColumnas.length; j++) {
				aux.put(nombresColumnas[j], partesFilaValidacion[j]);

			}
			datosEmpresaEntrada.put(i, aux);

			if (filaValidacion.contains(tiempoVelaT2Predicha)) {
				indiceVelaT2Validacion = i;
			}

			i++;

		}

		String targetReal = ConstructorElaborados.calcularTarget(empresa, datosEmpresaEntrada, indiceVelaT2Validacion,
				S, X, R, M, F);

		Boolean out = null;
		if (targetPredicho != null && targetReal != null) {
			out = (targetReal.equals("1") && targetPredicho.equals(targetReal)); // ACIERTO: si lo real fue un uno y
																					// predijimos un uno.
		}

		MY_LOGGER.info("empresa=" + empresa + " targetPredicho=" + targetPredicho + " targetReal=" + targetReal
				+ " ==> out=" + out);

		return out;
	}

//	/**
//	 * @param filaPredicha    Fila en fichero PREDICHAS en la FECHA en la que
//	 *                        hicimos la prediccion para adivinar el instante con
//	 *                        antigüedad=(FECHA - X)
//	 * @param filaValidacion  Fila en fichero VALIDACION en la FECHA ANALIZADA (en
//	 *                        la que hemos predicho)
//	 * @param filasValidacion
//	 * @param S
//	 * @param X
//	 * @param R
//	 * @param M
//	 */
//	public static void compararFilaPredichaYReal(String filaPredicha, String filaValidacion,
//			List<String> filasValidacion, final Integer S, final Integer X, final Integer R, final Integer M,
//			ValidadorComparacionSubgrupo vc) {
//
//		// Tengo ya las posiciones en empresa+fecha en predicha y en validación.
//
//		// RENTABILIDAD ACIERTOS/FALLOS para TARGET=1
//		// Cogeré el target predicho y el de validación, y los compararé
//		// Las dos últimas columnas son: TARGET_REAL (VALIDACIÓN), TARGET_PREDICHO
//		String targetPredicho = filaPredicha.substring(filaPredicha.lastIndexOf("|") + 1);
//		Integer indiceValidacionAux = SubgruposUtils.indiceDeAparicion("|".charAt(0),
//				(int) filaValidacion.chars().filter(ch -> ch == "|".charAt(0)).count(), filaValidacion);
//
//		String targetValidacion = filaValidacion.substring(indiceValidacionAux - 3, indiceValidacionAux - 2);
//
//		if (targetPredicho.equals("1")) {
//
//			vc.totalTargetUnoEnSubgrupo++;
//			if (targetPredicho.equals(targetValidacion)) {
//				vc.aciertosTargetUnoSubgrupo++;
//				vc.acertado = Boolean.TRUE;
//			} else {
//				vc.fallosTargetUnoSubgrupo++;
//				vc.acertado = Boolean.FALSE;
//			}
//
//			// RENTABILIDAD PRECIOS para TARGET = 1
//			// Para esto, debo buscar la fila de validación del futuro, X velas más allá
//			// (menor en antigüedad) que la actual
//			vc.antiguedadFutura = Integer.valueOf(vc.antiguedadValidacion) - X;
//
//			for (String filaValidacionFutura : filasValidacion) {
//				if (filaValidacionFutura.startsWith(vc.empresaValidacion + "|" + vc.antiguedadFutura.toString())) {
//
//					vc.closeValidacionFutura = Double.valueOf(filaValidacionFutura.substring(
//							SubgruposUtils.indiceDeAparicion("|".charAt(0), 11, filaValidacionFutura) + 1,
//							SubgruposUtils.indiceDeAparicion("|".charAt(0), 12, filaValidacionFutura)));
//
//					vc.closeValidacionActual = Double.valueOf(filaValidacion.substring(
//							SubgruposUtils.indiceDeAparicion("|".charAt(0), 11, filaValidacion) + 1,
//							SubgruposUtils.indiceDeAparicion("|".charAt(0), 12, filaValidacion)));
//
//					Double performance = vc.calcularRentabilidad();
//					vc.performanceClose.addValue(performance);
//
//					if (vc.acertado) {
//						vc.performanceCloseAcertados.addValue(performance);
//					} else {
//						vc.performanceCloseFallados.addValue(performance);
//					}
//
//					MY_LOGGER.info("----------------NUEVO CASO---------------");
//					MY_LOGGER.info("filaPredicha:                             " + filaPredicha);
//					MY_LOGGER.info("filaValidacion:                           " + filaValidacion);
//					MY_LOGGER.info("filaValidacionFutura (para rentabilidad): " + filaValidacionFutura);
//					MY_LOGGER.info("closeValidacionActual: " + vc.closeValidacionActual);
//					MY_LOGGER.info("closeValidacionFutura: " + vc.closeValidacionFutura);
//
//					if (vc.acertado) {
//						MY_LOGGER.info("EL SISTEMA HA ACERTADO, con rentabilidad: " + performance * 100 + " %");
//					} else {
//						MY_LOGGER.info("EL SISTEMA HA FALLADO, con rentabilidad: " + performance * 100 + " %");
//					}
//					MY_LOGGER.info("----------------FIN NUEVO CASO---------------");
//
//					// No itero más
//					break;
//				}
//
//			}
//		}
//
//	}

}
