package c70x.validacion;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c30x.elaborados.construir.Estadisticas;
import c40X.subgrupos.SubgruposUtils;

public class Validador implements Serializable {

	/**
	 * 
	 */
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
		Integer S = ValidadorUtils.S; // DEFAULT
		Integer X = ValidadorUtils.X; // DEFAULT
		Integer R = ValidadorUtils.R; // DEFAULT
		Integer M = ValidadorUtils.M; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 6) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			velasRetroceso = Integer.valueOf(args[0]);
			pathValidacion = args[1];
			S = Integer.valueOf(args[2]);
			X = Integer.valueOf(args[3]);
			R = Integer.valueOf(args[4]);
			M = Integer.valueOf(args[5]);
		}

		analizarPrediccion(velasRetroceso, pathValidacion, S, X, R, M);

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
			final Integer X, final Integer R, final Integer M) throws IOException {

		Predicate<Path> filtroFicheroPrediccion = s -> s.getFileName().toString().contains(DEFINICION_PREDICCION);
		Predicate<Path> filtroPredichos = p -> p.getFileName().toString().startsWith(velasRetroceso + "_");
		Predicate<Path> filtroValidaciones = p -> p.getFileName().toString().startsWith("0_");

		List<Path> ficherosPredichos = Files.walk(Paths.get(pathValidacion), 2)
				.filter(filtroFicheroPrediccion.and(filtroPredichos)).collect(Collectors.toList());
		List<Path> ficherosValidacion = Files.walk(Paths.get(pathValidacion), 2)
				.filter(filtroFicheroPrediccion.and(filtroValidaciones)).collect(Collectors.toList());

		// Recorro cada subgrupo predicho (a validar). Para cada uno, cogeré el fichero
		// equivalente
		// en el otro bloque de carpetas de de validación, dentro de su subgrupo
		List<String> filasPredichas, filasValidacion;

		String empresaPredicha, empresaValidacion, antiguedadPredicha, antiguedadValidacion, fechaPredicha,
				fechaValidacion;
		Integer indicePredicha, indiceValidacion;
		String targetPredicho, targetValidacion;

		Integer aciertosTargetUnoSubgrupo, fallosTargetUnoSubgrupo, totalTargetUnoEnSubgrupo;
		Integer antiguedadFutura;
		Estadisticas performanceClose;

		for (Path predicho : ficherosPredichos) {

			// Reinicio contadores de Subgrupo
			aciertosTargetUnoSubgrupo = 0;
			fallosTargetUnoSubgrupo = 0;
			totalTargetUnoEnSubgrupo = 0;
			performanceClose = new Estadisticas();

			for (Path validacion : ficherosValidacion) {
				// Se asume que la estructura del nombre de cada fichero es:
				// <retroceso>_SG_<numeroSubgrupo>_COMPLETO_PREDICCION.csv

				String nombrePredicho = predicho.getFileName().toString();
				String finalnombrePredicho = nombrePredicho.substring(nombrePredicho.indexOf("_"));

				String nombreValidacion = validacion.getFileName().toString();
				String finalnombreValidacion = nombreValidacion.substring(nombreValidacion.indexOf("_"));

				if (finalnombrePredicho.compareTo(finalnombreValidacion) == 0) {

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

					// Elimino las filas de cabecera
					filasPredichas.remove(0);
					filasValidacion.remove(0);

					for (String filaPredicha : filasPredichas) {

						antiguedadPredicha = filaPredicha.substring(
								SubgruposUtils.indiceDeAparicion("|".charAt(0), 1, filaPredicha) + 1,
								SubgruposUtils.indiceDeAparicion("|".charAt(0), 2, filaPredicha));
						indicePredicha = SubgruposUtils.indiceDeAparicion("|".charAt(0), 8, filaPredicha);
						fechaPredicha = filaPredicha.substring(0, indicePredicha);
						empresaPredicha = fechaPredicha.substring(0, fechaPredicha.indexOf("|"));
						fechaPredicha = SubgruposUtils.recortaPrimeraParteDeString("|".charAt(0), 3, fechaPredicha);

						// Sólo se analizarán las predicciones. Es decir, las filas con antigüedad=0 en
						// predichas
						if (antiguedadPredicha.compareTo("0") == 0) {
							for (String filaValidacion : filasValidacion) {
								antiguedadValidacion = filaValidacion.substring(
										SubgruposUtils.indiceDeAparicion("|".charAt(0), 1, filaValidacion) + 1,
										SubgruposUtils.indiceDeAparicion("|".charAt(0), 2, filaValidacion));
								indiceValidacion = SubgruposUtils.indiceDeAparicion("|".charAt(0), 8, filaValidacion);
								fechaValidacion = filaValidacion.substring(0, indiceValidacion);
								empresaValidacion = fechaValidacion.substring(0, fechaValidacion.indexOf("|"));
								fechaValidacion = SubgruposUtils.recortaPrimeraParteDeString("|".charAt(0), 3,
										fechaValidacion);

								if (empresaPredicha.compareTo(empresaValidacion) == 0
										&& fechaPredicha.compareTo(fechaValidacion) == 0) {
									// Tengo ya las posiciones en empresa+fecha en predicha y en validación.

									// RENTABILIDAD ACIERTOS/FALLOS para TARGET=1
									// Cogeré el target predicho y el de validación, y los compararé
									// Las dos últimas columnas son: TARGET_REAL (VALIDACIÓN), TARGET_PREDICHO
									targetPredicho = filaPredicha.substring(filaPredicha.lastIndexOf("|") + 1);
									Integer indice = SubgruposUtils.indiceDeAparicion("|".charAt(0),
											(int) filaValidacion.chars().filter(ch -> ch == "|".charAt(0)).count(),
											filaValidacion);
									targetValidacion = filaValidacion.substring(indice - 3, indice - 2);

									if (targetPredicho.compareTo("1") == 0) {
										totalTargetUnoEnSubgrupo++;
										if (targetPredicho.compareTo(targetValidacion) == 0) {
											aciertosTargetUnoSubgrupo++;
										} else {
											fallosTargetUnoSubgrupo++;
										}

										// RENTABILIDAD PRECIOS para TARGET = 1
										// Para esto, debo buscar la fila de validación del futuro, X velas más allá
										// (menor en antigüedad) que la actual
										antiguedadFutura = Integer.valueOf(antiguedadValidacion) - X;
										for (String filaValidacionFutura : filasValidacion) {
											if (filaValidacionFutura.startsWith(
													empresaValidacion + "|" + antiguedadFutura.toString())) {

												Double closeValidacionFutura = Double
														.valueOf(filaValidacionFutura.substring(
																SubgruposUtils.indiceDeAparicion("|".charAt(0), 11,
																		filaValidacionFutura) + 1,
																SubgruposUtils.indiceDeAparicion("|".charAt(0), 12,
																		filaValidacionFutura)));

												Double closeValidacionActual = Double.valueOf(filaValidacion.substring(
														SubgruposUtils.indiceDeAparicion("|".charAt(0), 11,
																filaValidacion) + 1,
														SubgruposUtils.indiceDeAparicion("|".charAt(0), 12,
																filaValidacion)));

												performanceClose
														.addValue((closeValidacionFutura - closeValidacionActual)
																/ closeValidacionActual);

												MY_LOGGER.info("----------------NUEVO CASO---------------");
												MY_LOGGER.info(
														"filaPredicha:                             " + filaPredicha);
												MY_LOGGER.info(
														"filaValidacion:                           " + filaValidacion);
												MY_LOGGER.info("filaValidacionFutura (para rentabilidad): "
														+ filaValidacionFutura);
												MY_LOGGER.info("closeValidacionActual: " + closeValidacionActual);
												MY_LOGGER.info("closeValidacionFutura: " + closeValidacionFutura);
												MY_LOGGER.info("----------------FIN NUEVO CASO---------------");

												// No itero más
												break;
											}

										}
									}

									break; // No sigo iterando en este for interno, porque ya tengo la info
								}
							}

						}

					}

				}
			}

			// PORCENTAJE ACIERTOS por SUBGRUPO
			MY_LOGGER.info("COBERTURA - Porcentaje aciertos subgrupo " + predicho.getFileName() + ": "
					+ (aciertosTargetUnoSubgrupo / totalTargetUnoEnSubgrupo) * 100 + "% de " + totalTargetUnoEnSubgrupo
					+ " elementos PREDICHOS a TARGET=1");

			// RENTABILIDAD PRECIOS por SUBGRUPO
			Double mediaRendimientoClose = performanceClose.getMean();
			Double stdRendimientoClose = performanceClose.getStandardDeviation();
			MY_LOGGER.info("RENTABILIDAD - porcentaje medio de SUBIDA del precio en subgrupo " + predicho.getFileName()
					+ ": " + mediaRendimientoClose * 100 + " % de " + totalTargetUnoEnSubgrupo
					+ " elementos PREDICHOS a TARGET=1");
			MY_LOGGER.info("RENTABILIDAD - Desviación estándar de SUBIDA del precio en subgrupo "
					+ predicho.getFileName() + ": " + stdRendimientoClose + " para " + totalTargetUnoEnSubgrupo
					+ " elementos PREDICHOS a TARGET=1");

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
}