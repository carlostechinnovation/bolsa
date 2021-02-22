package c70x.validacion;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c30x.elaborados.construir.ElaboradosUtils;
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
	private static String MODO_VALIDAR = "VALIDAR";
	private static String MODO_MEDIR_OVERFITTING = "MEDIR_OVERFITTING";

	public static final Integer UMBRAL_SUFICIENTES_ITEMS_OVERFITTING = 5;

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
		String MODO = MODO_VALIDAR; // DEFAULT
		String PATH_VALIDADOR_LOG = "/bolsa/logs/validador.log"; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 8) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			velasRetroceso = Integer.valueOf(args[0]);
			pathValidacion = args[1];
			S = Integer.valueOf(args[2]);
			X = Integer.valueOf(args[3]);
			R = Integer.valueOf(args[4]);
			M = Integer.valueOf(args[5]);
			MODO = args[6];
			PATH_VALIDADOR_LOG = args[7];
		}

		if (MODO != null && MODO_VALIDAR.equalsIgnoreCase(MODO)) {
			analizarPrediccion(velasRetroceso, pathValidacion, S, X, R, M);

		} else if (MODO != null && MODO_MEDIR_OVERFITTING.equalsIgnoreCase(MODO)) {
			medirOverfitting(PATH_VALIDADOR_LOG);

		} else {
			MY_LOGGER.error("Parametro de entrada SOLO_MEDIR_SOBREENTRENAMIENTO es incorrecto! Saliendo...");
			System.exit(-1);
		}

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

		int antiguedadAnalizada = velasRetroceso;
		int prefijoFicheroValidacion = velasRetroceso - X - M;

		MY_LOGGER.info("analizarPrediccion: velasRetroceso|S|X|R|M|antiguedadAnalizada|prefijoFicheroValidacion --> "
				+ velasRetroceso + "|" + S + "|" + X + "|" + R + "|" + M + "|" + antiguedadAnalizada + "|"
				+ prefijoFicheroValidacion);

		Predicate<Path> filtroFicheroPrediccion = s -> s.getFileName().toString().contains(DEFINICION_PREDICCION);
		Predicate<Path> filtroPredichos = p -> p.getFileName().toString().startsWith(antiguedadAnalizada + "_");
		Predicate<Path> filtroValidaciones = p -> p.getFileName().toString().startsWith(prefijoFicheroValidacion + "_");

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

		Float aciertosTargetUnoSubgrupo, fallosTargetUnoSubgrupo, totalTargetUnoEnSubgrupo;
		Integer antiguedadFutura;
		Estadisticas performanceClose, performanceCloseAcertados, performanceCloseFallados;
		Double closeValidacionFutura, closeValidacionActual;
		Double performance;
		Double mediaRendimientoClose, mediaRendimientoCloseAcertados, mediaRendimientoCloseFallados;
		Double stdRendimientoClose, stdRendimientoCloseAcertados, stdRendimientoCloseFallados;
		Boolean acertado;

		for (Path predicho : ficherosPredichos) {

			// Reinicio contadores de Subgrupo
			aciertosTargetUnoSubgrupo = 0F;
			fallosTargetUnoSubgrupo = 0F;
			totalTargetUnoEnSubgrupo = 0F;
			performanceClose = new Estadisticas();
			performanceCloseAcertados = new Estadisticas();
			performanceCloseFallados = new Estadisticas();

			for (Path validacion : ficherosValidacion) {
				// Se asume que la estructura del nombre de cada fichero es:
				// <retroceso>_SG_<numeroSubgrupo>_COMPLETO_PREDICCION.csv

				String nombrePredicho = predicho.getFileName().toString();
				String finalnombrePredicho = nombrePredicho.substring(nombrePredicho.indexOf("_"));

				String nombreValidacion = validacion.getFileName().toString();
				String finalnombreValidacion = nombreValidacion.substring(nombreValidacion.indexOf("_"));

				if (finalnombrePredicho.compareTo(finalnombreValidacion) == 0) {

					MY_LOGGER.info("Se compara el fichero de PREDICCIÓN (" + nombrePredicho
							+ ") con el que tiene la info REAL de contraste/validación (" + nombreValidacion + "): \n");

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
					MY_LOGGER.debug("Header predicha -> " + filasPredichas.get(0));
					MY_LOGGER.debug("Header validacion -> " + filasValidacion.get(0));
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

									String[] filaPredichaPartes = filaPredicha.split("\\|", -1); // los vacios son
																									// importantes
									targetPredicho = filaPredichaPartes[filaPredichaPartes.length - 2]; // penultimo

									String[] filaValidacionPartes = filaValidacion.split("\\|", -1); // los vacios son
																										// importantes
									targetValidacion = filaValidacionPartes[filaValidacionPartes.length - 3];

									if (targetPredicho != null && !targetPredicho.isEmpty()
											&& Integer.valueOf(targetPredicho).intValue() == 1) {

										totalTargetUnoEnSubgrupo++;

										if (targetValidacion != null && !targetValidacion.isEmpty()
												&& Float.valueOf(targetValidacion).intValue() == 1) {
											aciertosTargetUnoSubgrupo++;
											acertado = Boolean.TRUE;
										} else {
											fallosTargetUnoSubgrupo++;
											acertado = Boolean.FALSE;
										}

										// RENTABILIDAD PRECIOS para TARGET = 1
										// Para esto, debo buscar la fila de validación del futuro, X velas más allá
										// (menor en antigüedad) que la actual
										antiguedadFutura = Integer.valueOf(antiguedadValidacion) - X;

										for (String filaValidacionFutura : filasValidacion) {

											if (filaValidacionFutura.startsWith(
													empresaValidacion + "|" + antiguedadFutura.toString())) {

												closeValidacionFutura = Double.valueOf(filaValidacionFutura.substring(
														SubgruposUtils.indiceDeAparicion("|".charAt(0), 11,
																filaValidacionFutura) + 1,
														SubgruposUtils.indiceDeAparicion("|".charAt(0), 12,
																filaValidacionFutura)));

												closeValidacionActual = Double.valueOf(filaValidacion.substring(
														SubgruposUtils.indiceDeAparicion("|".charAt(0), 11,
																filaValidacion) + 1,
														SubgruposUtils.indiceDeAparicion("|".charAt(0), 12,
																filaValidacion)));

												performance = (closeValidacionFutura - closeValidacionActual)
														/ closeValidacionActual;
												performanceClose.addValue(performance);

												if (acertado) {
													performanceCloseAcertados.addValue(performance);
												} else {
													performanceCloseFallados.addValue(performance);
												}

												MY_LOGGER.info("----------------NUEVO CASO---------------");
												MY_LOGGER.info(
														"----ATENCION, IMPORTANTE: Se predice TARGET=1 para la empresa: "
																+ empresaPredicha);
												MY_LOGGER.info(
														"filaPredicha:     							" + filaPredicha);
												MY_LOGGER.info(
														"filaValidacion:   							" + filaValidacion);
												MY_LOGGER.info("filaValidacionFutura (para rentabilidad): 	"
														+ filaValidacionFutura);
												MY_LOGGER.info("closeValidacionActual:	" + closeValidacionActual);
												MY_LOGGER.info("closeValidacionFutura:	" + closeValidacionFutura);
												if (acertado) {
													MY_LOGGER.info("EL SISTEMA HA ACERTADO, con rentabilidad: "
															+ performance * 100 + " %");
												} else {
													MY_LOGGER.info("EL SISTEMA HA FALLADO, con rentabilidad: "
															+ performance * 100 + " %");
												}
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
			MY_LOGGER.info("FUTURO -> PRECISION - aciertosTargetUnoSubgrupo (PREDICHOS)=" + aciertosTargetUnoSubgrupo);
			MY_LOGGER.info("FUTURO -> PRECISION - totalTargetUnoEnSubgrupo (REALES)=" + totalTargetUnoEnSubgrupo);
			MY_LOGGER.info("FUTURO -> PRECISION - Porcentaje aciertos subgrupo " + predicho.getFileName() + ": "
					+ aciertosTargetUnoSubgrupo + "/" + totalTargetUnoEnSubgrupo + " = "
					+ Math.round((aciertosTargetUnoSubgrupo / totalTargetUnoEnSubgrupo) * 100)
					+ "% de elementos PREDICHOS con TARGET=1");

			// RENTABILIDAD PRECIOS por SUBGRUPO
			mediaRendimientoClose = performanceClose.getMean();
			stdRendimientoClose = performanceClose.getStandardDeviation();
			MY_LOGGER.info("RENTABILIDAD - Porcentaje medio de SUBIDA del precio en subgrupo " + predicho.getFileName()
					+ ": " + Math.round(mediaRendimientoClose * 100) + " % de " + Math.round(performanceClose.getN())
					+ " elementos PREDICHOS a TARGET=1");
			MY_LOGGER.info("RENTABILIDAD - Desviación estándar de SUBIDA del precio en subgrupo "
					+ predicho.getFileName() + ": " + stdRendimientoClose + " para "
					+ Math.round(performanceClose.getN()) + " elementos PREDICHOS a TARGET=1");

			// ANALISIS
			mediaRendimientoCloseAcertados = performanceCloseAcertados.getMean();
			stdRendimientoCloseAcertados = performanceCloseAcertados.getStandardDeviation();
			MY_LOGGER.info("ANALISIS - ACERTADOS - Porcentaje medio de SUBIDA del precio en subgrupo "
					+ predicho.getFileName() + ": " + mediaRendimientoCloseAcertados * 100 + " % de "
					+ Math.round(performanceCloseAcertados.getN()) + " elementos PREDICHOS a TARGET=1");
			MY_LOGGER.info("ANALISIS - ACERTADOS - Desviación estándar de SUBIDA del precio en subgrupo "
					+ predicho.getFileName() + ": " + stdRendimientoCloseAcertados + " para "
					+ Math.round(performanceCloseAcertados.getN()) + " elementos PREDICHOS a TARGET=1");

			mediaRendimientoCloseFallados = performanceCloseFallados.getMean();
			stdRendimientoCloseFallados = performanceCloseFallados.getStandardDeviation();
			MY_LOGGER.info("ANALISIS - FALLADOS - Porcentaje medio de SUBIDA del precio en subgrupo "
					+ predicho.getFileName() + ": " + mediaRendimientoCloseFallados * 100 + " % de "
					+ Math.round(performanceCloseFallados.getN())
					+ " elementos PREDICHOS (y además ACERTADOS) a TARGET=1");
			MY_LOGGER.info("ANALISIS - FALLADOS - Desviación estándar de SUBIDA del precio en subgrupo "
					+ predicho.getFileName() + ": " + stdRendimientoCloseFallados + " para "
					+ Math.round(performanceCloseFallados.getN())
					+ " elementos PREDICHOS (y además FALLADOS) a TARGET=1");

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
	 * Asumiendo que ya se ha calculado la validacion (PASADO vs FUT1-FUT2), esta
	 * función analiza el log para obtener una MEDIDA del OVERFITTING. Lo ideal es
	 * que sea 0. Se calculará mirando sólo los subgrupos con población suficiente
	 * en fut1-fut2.
	 * 
	 * @param validadorLog Path absoluto del log de validador.
	 * @throws IOException
	 */
	public static void medirOverfitting(final String validadorLog) throws IOException {

		List<String> datosPasadoStr = new ArrayList<String>();
		List<String> datosFut1Fut2Str = new ArrayList<String>();
		Map<Integer, Float> mapaSubgrupoMetricaPasado = new HashMap<Integer, Float>();
		Map<Integer, Float> mapaSubgrupoMetricaFuturo = new HashMap<Integer, Float>();

		// ---------------------------- FICHERO--------------------------------------
		File file = new File(validadorLog);
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		String linea = "";
		while ((linea = br.readLine()) != null) {
			if (linea != null && !linea.isEmpty()) {

				if (linea.contains("PASADO") && linea.contains("METRICA")) {
					datosPasadoStr.add(linea);
				} else if (linea.contains("FUTURO") && linea.contains("aciertos")) {
					datosFut1Fut2Str.add(linea);
				}
			}
		}
		fr.close();
		// ------------------------------------------------------------------

		// PASADO
		for (String cad : datosPasadoStr) {
			// MY_LOGGER.info(cad);
			if (cad != null && cad.contains("SG_") && cad.contains("METRICA")) {
				Integer subgrupo = Integer.valueOf((cad.split("SG_"))[1].split(" ")[0]);
				Float metrica = Float.valueOf((cad.split("avg_precision = "))[1].split(" ")[0].replace(")", "").trim());

				mapaSubgrupoMetricaPasado.put(subgrupo, 100.0F * metrica);
			}
		}

		// FUTURO
		for (String cad : datosFut1Fut2Str) {
			// MY_LOGGER.info(cad);

			if (cad != null && cad.contains("Porcentaje") && cad.contains("SG") && cad.contains("%")) {

				String[] partes = (cad.split("Porcentaje aciertos subgrupo "))[1].split("_");
				Integer subgrupo = Integer.valueOf(partes[2]);
				String[] partes2 = partes[4].split(" = ");
				String[] partes3 = (partes2[0].split(" "))[1].split("/");
				Float aciertos = Float.valueOf(partes3[0]);
				Float totales = Float.valueOf(partes3[1]);

				if (totales.intValue() >= UMBRAL_SUFICIENTES_ITEMS_OVERFITTING) {
					mapaSubgrupoMetricaFuturo.put(subgrupo, 100.0F * aciertos / totales);
				}

			}

		}

		// OVERFITTING
		// Para cada subgrupo, se compara la métrica del pasado (train) y la real del
		// fut1fut2. El overfitting será su diferencia.
		// Idealmente debería ser 0
		Map<Integer, Float> mapaOverfitting = new HashMap<Integer, Float>();
		for (Integer subgrupo : mapaSubgrupoMetricaPasado.keySet()) {
			if (mapaSubgrupoMetricaFuturo.containsKey(subgrupo) && !mapaSubgrupoMetricaPasado.get(subgrupo).isNaN()
					&& !mapaSubgrupoMetricaFuturo.get(subgrupo).isNaN()) {

				mapaOverfitting.put(subgrupo,
						mapaSubgrupoMetricaPasado.get(subgrupo) - mapaSubgrupoMetricaFuturo.get(subgrupo));
			}
		}

		DecimalFormat df = new DecimalFormat("0.00");

		MY_LOGGER.info("*******************");
		double suma = mapaOverfitting.values().stream().mapToDouble(a -> a).sum();
		double overfittingMedio = suma / mapaOverfitting.size();
		MY_LOGGER.info("OVERFITTING (sobreentrenamiento) MEDIO = " + df.format(overfittingMedio));
		MY_LOGGER.info("*******************");

		MY_LOGGER.info("*******************");
		MY_LOGGER.info(
				"MEDIDA DEL SOBREENTRENAMIENTO (overfitting) de todos los subgrupos (UMBRAL_SUFICIENTES_ITEMS_OVERFITTING = "
						+ UMBRAL_SUFICIENTES_ITEMS_OVERFITTING + "):");
		List<Integer> clavesOrdenadas = new ArrayList(mapaOverfitting.keySet());
		Collections.sort(clavesOrdenadas);
		for (Integer sg : clavesOrdenadas) {
			boolean mejorQueLaMedia = mapaOverfitting.get(sg) < overfittingMedio;
			String mejorQueLaMediaStr = mejorQueLaMedia ? " --> POCO OVERFITTING" : "";
			MY_LOGGER.info("Overfitting del subgrupo=" + sg + " -> " + df.format(mapaOverfitting.get(sg))
					+ mejorQueLaMediaStr);
		}

	}

}
