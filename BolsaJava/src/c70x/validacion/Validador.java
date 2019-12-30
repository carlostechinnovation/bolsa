package c70x.validacion;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
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

	// TODO: el final del fichero de predicciones debe llamarse
	// COMPLETO_PREDICCION.csv, no lo que indicamos a continuación. Acordarlo con
	// Carlos y cambiar esta línea.
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

		String pathPredicho = ValidadorUtils.PATH_PREDICHO; // DEFAULT
		String pathValidacion = ValidadorUtils.PATH_VALIDACION; // DEFAULT
		Integer S = ValidadorUtils.S; // DEFAULT
		Integer X = ValidadorUtils.X; // DEFAULT
		Integer R = ValidadorUtils.R; // DEFAULT
		Integer M = ValidadorUtils.M; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 2) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			pathPredicho = args[0];
			pathValidacion = args[1];
			S = Integer.valueOf(args[2]);
			X = Integer.valueOf(args[3]);
			R = Integer.valueOf(args[4]);
			M = Integer.valueOf(args[5]);
		}

		analizarPrediccion(pathPredicho, pathValidacion, S, X, R, M);

		MY_LOGGER.info("FIN");
	}

	/**
	 * * Tanto en la predicción desfasada como en la actual, estarán en el mismo
	 * esquema de subcarpetas, así que se recorre cada subgrupo en paralelo, y se
	 * comparan en parejas los ficheros actual y desfasado. Se obtienen las
	 * estadísticas de aciertos/fallos.
	 * 
	 * @param pathPredicho   Predicción, en tiempo desfasado.
	 * @param pathValidacion Datos a tiempo actual, con los targets rellenos, con
	 *                       los que se validará.
	 * @param S              Subida mínima
	 * @param X              Dentro de X velas
	 * @param R              Caída máxima
	 * @param M              En todas las M velas posteriores
	 * @throws IOException
	 */
	public static void analizarPrediccion(final String pathPredicho, final String pathValidacion, Integer S, Integer X,
			Integer R, Integer M) throws IOException {

		List<Path> ficherosPredichos = Files.walk(Paths.get(pathPredicho), 2)
				.filter(s -> s.getFileName().toString().contains(DEFINICION_PREDICCION)).collect(Collectors.toList());
		List<Path> ficherosValidacion = Files.walk(Paths.get(pathValidacion), 2)
				.filter(s -> s.getFileName().toString().contains(DEFINICION_PREDICCION)).collect(Collectors.toList());

		// Recorro cada subgrupo predicho (a validar). Para cada uno, cogeré el fichero
		// equivalente
		// en el otro bloque de carpetas de de validación, dentro de su subgrupo
		List<String> filasPredichas, filasValidacion;

		String empresaPredicha, empresaValidacion, antiguedadPredicha, antiguedadValidacion, fechaPredicha,
				fechaValidacion;
		Integer indicePredicha, indiceValidacion;
		String targetPredicho, targetValidacion;

		Integer aciertosTargetUnoSubgrupo, fallosTargetUnoSubgrupo, totalTargetUnoEnSubgrupo;
		Double precioActualTargetUnoPorSubgrupo, precioFuturoTargetUnoPorSubgrupo, rendimientoMedioTargetUnoPorSubgrupo;
		Integer antiguedadFutura;
		Estadisticas performanceClose;

		for (Path predicho : ficherosPredichos) {

			// Reinicio contadores de Subgrupo
			aciertosTargetUnoSubgrupo = 0;
			fallosTargetUnoSubgrupo = 0;
			totalTargetUnoEnSubgrupo = 0;
			precioActualTargetUnoPorSubgrupo = 0D;
			precioFuturoTargetUnoPorSubgrupo = 0D;
			rendimientoMedioTargetUnoPorSubgrupo = 0D;
			performanceClose = new Estadisticas();

			for (Path validacion : ficherosValidacion) {
				if (predicho.getFileName().toString().compareTo(validacion.getFileName().toString()) == 0) {
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
									targetPredicho = filaPredicha.substring(filaPredicha.lastIndexOf("|") + 1);
									targetValidacion = filaValidacion.substring(filaValidacion.lastIndexOf("|") + 1);

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
			System.out.println("Porcentaje aciertos subgrupo " + predicho.getFileName() + ": "
					+ (aciertosTargetUnoSubgrupo / totalTargetUnoEnSubgrupo) * 100 + "% de " + totalTargetUnoEnSubgrupo
					+ " elementos PREDICHOS a TARGET=1");
			MY_LOGGER.info("Porcentaje aciertos subgrupo " + predicho.getFileName() + ": "
					+ (aciertosTargetUnoSubgrupo / totalTargetUnoEnSubgrupo) * 100 + "% de " + totalTargetUnoEnSubgrupo
					+ " elementos PREDICHOS a TARGET=1");

			// RENTABILIDAD PRECIOS por SUBGRUPO
			Double mediaRendimientoClose = performanceClose.getMean();
			Double stdRendimientoClose = performanceClose.getStandardDeviation();
			System.out.println("Porcentaje medio de rendimiento en precios CLOSE subgrupo " + predicho.getFileName()
					+ ": " + mediaRendimientoClose * 100 + " % de " + totalTargetUnoEnSubgrupo
					+ " elementos PREDICHOS a TARGET=1");
			MY_LOGGER.info("Porcentaje medio de rendimiento en precios CLOSE subgrupo " + predicho.getFileName() + ": "
					+ mediaRendimientoClose * 100 + " % de " + totalTargetUnoEnSubgrupo
					+ " elementos PREDICHOS a TARGET=1");
			System.out.println("Desviación estándar de rendimiento en precios CLOSE subgrupo " + predicho.getFileName()
					+ ": " + stdRendimientoClose + " de " + totalTargetUnoEnSubgrupo
					+ " elementos PREDICHOS a TARGET=1");
			MY_LOGGER.info("Desviación estándar de rendimiento en precios CLOSE subgrupo " + predicho.getFileName()
					+ ": " + stdRendimientoClose + " de " + totalTargetUnoEnSubgrupo
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
