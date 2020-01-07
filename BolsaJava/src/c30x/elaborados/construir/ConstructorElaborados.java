package c30x.elaborados.construir;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c20X.limpios.LimpiosUtils;
import c30x.elaborados.construir.Estadisticas.FINAL_NOMBRES_PARAMETROS_ELABORADOS;

public class ConstructorElaborados implements Serializable {

	static Logger MY_LOGGER = Logger.getLogger(ConstructorElaborados.class);

	private static ConstructorElaborados instancia = null;

	private ConstructorElaborados() {
		super();
	}

	public static ConstructorElaborados getInstance() {
		if (instancia == null)
			instancia = new ConstructorElaborados();

		return instancia;
	}

	// META-PARAMETRIZACION
	// Periodo de la vela de entrada
	public final static String T_velaEntrada = "H";
	// x dias

	// Se usan los periodos típicos que suelen usar los robots: 3, 7, 20, 50 días
	public final static Integer[] periodosHParaParametros = new Integer[] { 3 * ElaboradosUtils.HORAS_AL_DIA,
			7 * ElaboradosUtils.HORAS_AL_DIA, 20 * ElaboradosUtils.HORAS_AL_DIA, 50 * ElaboradosUtils.HORAS_AL_DIA };

	// IMPORTANTE: se asume que los datos estan ordenados de menor a mayor
	// antiguedad, y agrupados por empresa

	final static String TARGET_INVALIDO = "null";

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		Object appendersAcumulados = Logger.getRootLogger().getAllAppenders();
		if (appendersAcumulados instanceof NullEnumeration) {
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(PatternLayout.TTCC_CONVERSION_PATTERN)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

		String directorioIn = LimpiosUtils.DIR_LIMPIOS; // DEFAULT
		String directorioOut = ElaboradosUtils.DIR_ELABORADOS; // DEFAULT
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
			directorioIn = args[0];
			directorioOut = args[1];
			S = Integer.valueOf(args[2]);
			X = Integer.valueOf(args[3]);
			R = Integer.valueOf(args[4]);
			M = Integer.valueOf(args[5]);
			F = Integer.valueOf(args[6]);
		}

		File directorioEntrada = new File(directorioIn);
		File directorioSalida = new File(directorioOut);
		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosEntrada;
		HashMap<Integer, String> ordenNombresParametros;
		GestorFicheros gestorFicheros = new GestorFicheros();
		ArrayList<File> ficherosEntradaEmpresas = gestorFicheros.listaFicherosDeDirectorio(directorioEntrada);

		MY_LOGGER.info(crearDefinicionTarget(S, X, R, M, F));

		String destino = "";
		Iterator<File> iterator = ficherosEntradaEmpresas.iterator();
		File ficheroGestionado;

		int i = 1;

		while (iterator.hasNext()) {

			if (i % 10 == 1) {
				MY_LOGGER.info("Empresa numero = " + i);
			}
			i++;

			gestorFicheros = new GestorFicheros();
			datosEntrada = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
			ficheroGestionado = iterator.next();
			datosEntrada = gestorFicheros
					.leeSoloParametrosNoElaboradosFicheroDeSoloUnaEmpresa(ficheroGestionado.getPath(), Boolean.FALSE);
			destino = directorioSalida + "/" + ficheroGestionado.getName();
			MY_LOGGER.debug("Ficheros entrada|salida -> " + ficheroGestionado.getAbsolutePath() + " | " + destino);
			ordenNombresParametros = gestorFicheros.getOrdenNombresParametrosLeidos();
			anadirParametrosElaboradosDeSoloUnaEmpresa(datosEntrada, ordenNombresParametros, S, X, R, M, F);
			gestorFicheros.creaFicheroDeSoloUnaEmpresa(datosEntrada, ordenNombresParametros, destino);
		}

		MY_LOGGER.info("FIN");
	}

	/**
	 * 
	 * @param datos
	 * @param ordenNombresParametros
	 * @param S
	 * @param X
	 * @param R
	 * @param M
	 * @throws Exception
	 */
	public static void anadirParametrosElaboradosDeSoloUnaEmpresa(
			HashMap<String, HashMap<Integer, HashMap<String, String>>> datos,
			HashMap<Integer, String> ordenNombresParametros, Integer S, Integer X, Integer R, Integer M, Integer F)
			throws Exception {

		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosSalida = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
		HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada = new HashMap<Integer, HashMap<String, String>>();
		HashMap<Integer, HashMap<String, String>> datosEmpresaFinales = new HashMap<Integer, HashMap<String, String>>();

		// ORDEN DE PARÁMETROS DE ENTRADA
		HashMap<Integer, String> ordenNombresParametrosSalida = new HashMap<Integer, String>();
		Integer numeroParametrosEntrada = ordenNombresParametros.size();
		for (int i = 0; i < numeroParametrosEntrada; i++) {
			ordenNombresParametrosSalida.put(i, ordenNombresParametros.get(i));
		}

		// CÁLCULOS DE PARÁMETROS ELABORADOS
		Integer antiguedad;
		String empresa = "";
		Set<String> empresas = datos.keySet();
		Iterator<String> itEmpresas = datos.keySet().iterator();
		if (empresas.size() != 1) {
			throw new Exception("Se están calculando parámetros elaborados de más de una empresa");
		} else {
			while (itEmpresas.hasNext())
				empresa = itEmpresas.next();
		}

		// EXTRACCIÓN DE DATOS DE LA EMPRESA
		datosEmpresaEntrada = datos.get(empresa);
		MY_LOGGER.debug("anadirParametrosElaboradosDeSoloUnaEmpresa() -> Empresa: " + empresa);

		HashMap<String, String> parametros = new HashMap<String, String>();
		Iterator<Integer> itAntiguedad;
		Set<Integer> periodos, antiguedades;
		HashMap<Integer, Estadisticas> estadisticasPrecioPorAntiguedad = new HashMap<Integer, Estadisticas>();
		HashMap<Integer, Estadisticas> estadisticasVolumenPorAntiguedad = new HashMap<Integer, Estadisticas>();
		Estadisticas estadisticasPrecio = new Estadisticas();
		Estadisticas estadisticasVolumen = new Estadisticas();
		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasPrecioPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>();
		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasVolumenPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>();
		HashMap<Integer, String> ordenPrecioNombresParametrosElaborados = estadisticasPrecio
				.getOrdenNombresParametrosElaborados();
		HashMap<Integer, String> ordenVolumenNombresParametrosElaborados = estadisticasPrecio
				.getOrdenNombresParametrosElaborados();
		Integer parametrosAcumulados = numeroParametrosEntrada;
		String auxPrecio, auxVolumen;
		Integer antiguedadHistoricaMaxima;

		for (Integer periodo : periodosHParaParametros) {

			// Se guarda el orden de los datos elaborados
			for (int i = 0; i < ordenPrecioNombresParametrosElaborados.size(); i++) {
				// PRECIO
				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
						ordenPrecioNombresParametrosElaborados.get(i + 1) + periodo
								+ FINAL_NOMBRES_PARAMETROS_ELABORADOS._PRECIO.toString());
			}
			parametrosAcumulados += ordenPrecioNombresParametrosElaborados.size();

			for (int i = 0; i < ordenVolumenNombresParametrosElaborados.size(); i++) {
				// VOLUMEN
				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
						ordenVolumenNombresParametrosElaborados.get(i + 1) + periodo
								+ FINAL_NOMBRES_PARAMETROS_ELABORADOS._VOLUMEN.toString());
			}
			parametrosAcumulados += ordenVolumenNombresParametrosElaborados.size();
			Iterator<Integer> itAntiguedadTarget = datosEmpresaEntrada.keySet().iterator();

			while (itAntiguedadTarget.hasNext()) {
				antiguedad = itAntiguedadTarget.next();

				// PARA CADA PERIODO DE CALCULO DE PARAMETROS ELABORADOS y cada antiguedad, que
				// será un GRUPO de COLUMNAS...

				// Deben existir datos de una antiguedadHistorica = (antiguedad + periodo)
				antiguedadHistoricaMaxima = antiguedad + periodo;
				MY_LOGGER.debug("datosEmpresaEntrada.size(): " + datosEmpresaEntrada.size());
				MY_LOGGER.debug("Antiguedad: " + antiguedad);
				if (antiguedadHistoricaMaxima < datosEmpresaEntrada.size()) {

					for (int i = 0; i < periodo; i++) {

						parametros = datosEmpresaEntrada.get(i + antiguedad);
						MY_LOGGER.debug("i + antiguedad: " + (i + antiguedad));

						if (parametros == null) {

							MY_LOGGER.error("Empresa=" + empresa + " No hay datos para la vela --> " + (i + antiguedad)
									+ " Posible causa: el mercado estaba abierto cuando hemos ejecutado la descarga de datos");

						} else {
							// Se toma el parámetro "close" para las estadisticas de precio
							// Se toma el parámetro "volumen" para las estadisticas de volumen
							auxPrecio = parametros.get("close");
							auxVolumen = parametros.get("volumen");
							estadisticasPrecio.addValue(new Double(auxPrecio));
							estadisticasVolumen.addValue(new Double(auxVolumen));
							MY_LOGGER.debug("(antiguedad: " + antiguedad + ", periodo: " + periodo
									+ ") Metido para estadísticas: " + auxPrecio);
						}

					}

				} else {
					// Para los datos de antiguuedad excesiva, se sale del bucle
					break;
				}

				// VALIDACIÓN DE ESTADíSTICAS
				// La empresa y la antigüedad no las usamos

				MY_LOGGER.debug("------------------>>>>>>> Periodo: " + periodo + ", n: " + estadisticasPrecio.getN());
				estadisticasPrecioPorAntiguedad.put(antiguedad, estadisticasPrecio);
				estadisticasVolumenPorAntiguedad.put(antiguedad, estadisticasVolumen);
				// Se limpia este almacén temporal
				estadisticasPrecio = new Estadisticas();
				estadisticasVolumen = new Estadisticas();
			}

			estadisticasPrecioPorAntiguedadYPeriodo.put(periodo, estadisticasPrecioPorAntiguedad);
			estadisticasVolumenPorAntiguedadYPeriodo.put(periodo, estadisticasVolumenPorAntiguedad);
			// Se limpia este almacén temporal
			estadisticasPrecioPorAntiguedad = new HashMap<Integer, Estadisticas>();
			estadisticasVolumenPorAntiguedad = new HashMap<Integer, Estadisticas>();
		}

		// ESTADÍSTICA --> A la vez: CALCULA y RELLENA
		periodos = estadisticasPrecioPorAntiguedadYPeriodo.keySet();
		Integer periodoActual;
		Iterator<Integer> itPeriodo = periodos.iterator();

		HashMap<String, String> mapaParamsPrecio, mapaParamsVolumen;

		while (itPeriodo.hasNext()) {
			periodoActual = itPeriodo.next();
			estadisticasPrecioPorAntiguedad = estadisticasPrecioPorAntiguedadYPeriodo.get(periodoActual);
			estadisticasVolumenPorAntiguedad = estadisticasVolumenPorAntiguedadYPeriodo.get(periodoActual);
			antiguedades = estadisticasPrecioPorAntiguedad.keySet();
			itAntiguedad = antiguedades.iterator();

			while (itAntiguedad.hasNext()) {
				antiguedad = itAntiguedad.next();
				estadisticasPrecio = estadisticasPrecioPorAntiguedad.get(antiguedad);
				estadisticasVolumen = estadisticasVolumenPorAntiguedad.get(antiguedad);
				antiguedadHistoricaMaxima = antiguedad + periodoActual;
				// Se cogen sólo los datos con la antiguedad dentro del rango a analizar
				if (antiguedadHistoricaMaxima < datosEmpresaEntrada.size()) {
					parametros = datosEmpresaEntrada.get(antiguedad);
					// COSTE DE COMPUTACION
					// <<<<<<<<-------

					mapaParamsPrecio = estadisticasPrecio.getParametros(periodoActual,
							FINAL_NOMBRES_PARAMETROS_ELABORADOS._PRECIO.toString(), Boolean.FALSE);

					mapaParamsVolumen = estadisticasVolumen.getParametros(periodoActual,
							FINAL_NOMBRES_PARAMETROS_ELABORADOS._VOLUMEN.toString(), Boolean.FALSE);

					parametros.putAll(mapaParamsPrecio);
					parametros.putAll(mapaParamsVolumen);

					// <<<<<<<------
				} else {
					// Para los datos de antiguedad excesiva, salgo del bucle
					break;
				}
				// ADICION DE PARAMETROS ELABORADOS AL HASHMAP
				MY_LOGGER.debug("Anhadimos " + parametros.size() + " items a la antiguedad (vela) " + antiguedad
						+ " de la empresa " + empresa);

				datosEmpresaFinales.put(antiguedad, parametros);
			}
		}

		// Aniado el TARGET
		Integer antiguedadX;
		Double subidaSPrecioTantoPorUno = (100 + S) / 100.0;
		Double caidaRPrecioTantoPorUno = (100 - R) / 100.0;
		Double caidaFPrecioTantoPorUno = (100 - F) / 100.0;
		// Target=0 es que no se cumple. 1 es que sí. TARGET_INVALIDO es que no se puede
		// calcular
		String target = TARGET_INVALIDO;
		Boolean mCumplida = Boolean.FALSE;

		antiguedades = datosEmpresaEntrada.keySet();
		Integer antiguedadMaxima = Collections.max(antiguedades);
		HashMap<String, String> datosAntiguedad, datosAntiguedadX, datosAntiguedadM;
		Iterator<Integer> itAntiguedadTarget = datosEmpresaEntrada.keySet().iterator();
		HashMap<Integer, String> antiguedadYTarget = new HashMap<Integer, String>();

		while (itAntiguedadTarget.hasNext()) {

			antiguedad = itAntiguedadTarget.next();

			if (antiguedad >= M) {
				antiguedadX = antiguedad + X;

				if (antiguedadMaxima < antiguedadX) {
					// Estamos analizando un punto en el tiempo X datos anteriores
					target = TARGET_INVALIDO;
					break;

				} else {

					datosAntiguedad = datosEmpresaEntrada.get(antiguedad);
					datosAntiguedadX = datosEmpresaEntrada.get(antiguedadX);

					if (!datosAntiguedad.containsKey("close")) {
						MY_LOGGER.error("Empresa=" + empresa + " -> Falta dato close en datosAntiguedad");
					}
					if (!datosAntiguedadX.containsKey("close")) {
						MY_LOGGER.error("Empresa=" + empresa + " -> Falta dato close en datosAntiguedadX");
					}

					Double closeAntiguedad = Double.valueOf(datosAntiguedad.get("close"));
					Double closeAntiguedadX = Double.valueOf(datosAntiguedadX.get("close"));

					boolean closeActualSuperaCloseXConSubidaS = closeAntiguedad >= closeAntiguedadX
							* subidaSPrecioTantoPorUno;

					if (closeActualSuperaCloseXConSubidaS) {

						Integer antiguedadM = antiguedad - M;// Última vela M futura, más allá de la antigüedad actual
						datosAntiguedadM = datosEmpresaEntrada.get(antiguedadM);

						if (datosAntiguedadM == null) {
							MY_LOGGER.error("Empresa=" + empresa + " -> datosAntiguedadM es NULO para antiguedad="
									+ antiguedad + " y M=" + M + " -> antiguedadM=" + antiguedadM
									+ " Posible causa: el mercado estaba abierto cuando hemos ejecutado la descarga de datos");
						} else {

							Double closeAntiguedadM = Double.valueOf(datosAntiguedadM.get("close"));

							// En la vela M el precio debe ser un F% mejor que en la vela actual
							boolean closeMSuperaCloseActualMayorQueF = closeAntiguedad < caidaFPrecioTantoPorUno
									* closeAntiguedadM;
							if (closeMSuperaCloseActualMayorQueF) {

								for (int i = 1; i <= M; i++) {
									Integer antiguedadI = antiguedad - i; // Voy hacia el futuro

									Double closeAntiguedadI = Double
											.valueOf(datosEmpresaEntrada.get(antiguedadI).get("close"));
									if (closeAntiguedad * caidaRPrecioTantoPorUno < closeAntiguedadI) {
										// El precio puede haber caido, pero nunca más de R
										mCumplida = Boolean.TRUE;
									} else {
										// Se ha encontrado AL MENOS una vela posterior, en las M siguientes, con el
										// precio por debajo de la caída mínima R
										// TODAS LAS VELAS FUTURAS TIENEN QUE ESTAR POR ENCIMA DE ESE UMBRAL DE CAIDA
										mCumplida = Boolean.FALSE;
										break;
									}
								}
							}

							if (mCumplida) {
								// La S sí se cumple, y la M tambien en todo el rango
								target = "1";
							} else {
								target = "0";
							}
						}

					} else {
						// La S no se cumple
						target = "0";
					}
				}

			} else {
				// La antiguedad es demasiado reciente para ver si es estable en M
				target = TARGET_INVALIDO;
			}
			antiguedadYTarget.put(antiguedad, target);
		}

		// Se rellena el target en los datos de entrada tras el analisis, al final de
		// todos los parametros
		Iterator<Integer> itAntiguedadDatos = datosEmpresaFinales.keySet().iterator();
		ordenNombresParametrosSalida.put(ordenNombresParametrosSalida.size(), "TARGET");
		while (itAntiguedadDatos.hasNext()) {
			antiguedad = itAntiguedadDatos.next();
			parametros = datosEmpresaFinales.get(antiguedad);
			parametros.put("TARGET", String.valueOf(antiguedadYTarget.get(antiguedad)));
			datosEmpresaFinales.replace(antiguedad, parametros);
		}

		// Vuelco todos los parámetros
		datosSalida.put(empresa, datosEmpresaFinales);
		datos = datosSalida;
		ordenNombresParametros.clear();
		ordenNombresParametros.putAll(ordenNombresParametrosSalida);
	}

	/**
	 * Traducción a horas (hábiles, con Bolsa abierta)
	 * 
	 * @param T
	 * @return
	 * @throws Exception
	 */
	public static Integer tiempoEnHoras(String T) throws Exception {

		Integer horas = 0;
		if (T == "H") {
			horas = 1;
		} else if (T == "D") {
			// HAY DÍAS QUE LA BOLSA ABRE SÓLO MEDIA JORNADA, ASÍ QUE ESTO NO ES TOTALMENTE
			// CORRECTO. Normalmente son 7h al día
			horas = ElaboradosUtils.HORAS_AL_DIA;
		} else if (T == "H") {
			throw new Exception("Tiempo erróneo");
		}
		return horas;
	}

	/**
	 * @param S
	 * @param X
	 * @param R
	 * @param M
	 * @param F
	 * @return
	 */
	public static String crearDefinicionTarget(Integer S, Integer X, Integer R, Integer M, Integer F) {

		String definicion = "DEFINICION DE TARGET --> Vemos los instantes [t1,t2,t3]. Ahora estamos en t2. El periodo [t1,t2] duró "
				+ X + " velas, en el que el precio subió >= " + S + "% . El periodo [t2,t3] duró " + M
				+ " velas; durante TODAS ellas, el precio de cierre puede haber caido un poco, pero nunca por debajo de un "
				+ R + "% respecto del precio de t2. El precio en el instante t3 (es decir, tras " + M
				+ " velas desde ahora) debe ser >= " + F + "% respecto del t2. Entonces Target=1";

		return definicion;

	}

}
