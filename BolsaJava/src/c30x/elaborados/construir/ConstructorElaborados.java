package c30x.elaborados.construir;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
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
import c30x.elaborados.construir.Estadisticas.OTROS_PARAMS_ELAB;

public class ConstructorElaborados implements Serializable {

	private static final long serialVersionUID = 1L;

	static Logger MY_LOGGER = Logger.getLogger(ConstructorElaborados.class);

	private static ConstructorElaborados instancia = null;

	// Se toman los parámetros de los grupos 11, 33 y 42 como los únicos autorizados
	// en el sistema. Así evitamos que la matriz de datos se muy grande
	private static String LISTA_PARAMETROS_AUTORIZADOS = "MEDIA_SMA_1_VOLUMEN,RATIO_MINRELATIVO_4_HIGH,RATIO_MAXRELATIVO_4_LOW,RATIO_U_MAXRELATIVO_20_HIGH,RATIO_U_MINRELATIVO_20_LOW,SKEWNESS_20_LOW,SKEWNESS_20_OPEN,RATIO_SMA_20_VOLUMEN,PENDIENTE_2M_SMA_50_HIGH,RATIO_U_MINRELATIVO_50_HIGH,PENDIENTE_SMA_50_OPEN,CURTOSIS_50_CLOSEOPEN,RATIO_MINRELATIVO_50_HIGHLOW,RATIO_U_SMA_50_HIGHLOW,SG_HIGH_PENDIENTE7D,MEDIA_SMA_1_LOW,CURTOSIS_1_OPEN,MEDIA_SMA_1_HIGHLOW,RATIO_U_MAXRELATIVO_1_VOLUMEN,RATIO_U_MINRELATIVO_1_VOLUMEN,CURTOSIS_1_VOLUMEN,MEDIA_SMA_4_CLOSE,RATIO_SMA_4_CLOSE,RATIO_MAXRELATIVO_4_CLOSE,RATIO_MINRELATIVO_SEGUNDO_4_HIGH,RATIO_U_MINRELATIVO_4_HIGH,PENDIENTE_SMA_4_LOW,RATIO_MINRELATIVO_SEGUNDO_4_LOW,RATIO_U_MAXRELATIVO_4_LOW,RATIO_MAXRELATIVO_4_OPEN,SKEWNESS_4_VOLUMEN,RATIO_MINRELATIVO_20_CLOSE,RATIO_U_MINRELATIVO_20_CLOSE,RATIO_MAXRELATIVO_20_HIGH,RATIO_U_MINRELATIVO_20_HIGH,MEDIA_SMA_20_LOW,RATIO_MINRELATIVO_20_LOW,RATIO_U_MAXRELATIVO_20_LOW,SKEWNESS_20_CLOSEOPEN,RATIO_MINRELATIVO_20_HIGHLOW,RATIO_MINRELATIVO_SEGUNDO_20_HIGHLOW,PENDIENTE_SMA_20_VOLUMEN,RATIO_MAXRELATIVO_20_VOLUMEN,RATIO_MINRELATIVO_SEGUNDO_20_VOLUMEN,RATIO_U_MAXRELATIVO_20_VOLUMEN,CURTOSIS_20_VOLUMEN,SKEWNESS_20_VOLUMEN,RATIO_MINRELATIVO_SEGUNDO_50_CLOSE,RATIO_U_SMA_50_CLOSE,RATIO_U_MAXRELATIVO_50_HIGH,CURTOSIS_50_HIGH,SKEWNESS_50_HIGH,PENDIENTE_SMA_50_LOW,RATIO_MINRELATIVO_50_LOW,RATIO_U_SMA_50_LOW,PENDIENTE_2M_SMA_50_OPEN,RATIO_U_SMA_50_OPEN,SKEWNESS_50_CLOSEOPEN,MEDIA_SMA_50_OPENHIGH,RATIO_U_SMA_50_OPENHIGH,PENDIENTE_2M_SMA_50_HIGHLOW,RATIO_MINRELATIVO_SEGUNDO_50_HIGHLOW,RATIO_MINRELATIVO_50_VOLUMEN,RATIO_MINRELATIVO_SEGUNDO_50_VOLUMEN,CURTOSIS_50_VOLUMEN,SKEWNESS_50_VOLUMEN,SG_LOW_PENDIENTE3D,RATIO_MAXRELATIVO_20_CLOSE,RATIO_MAXRELATIVO_50_HIGH,SG_HIGH_PENDIENTE3D";

	private ConstructorElaborados() {
		super();
	}

	public static ConstructorElaborados getInstance() {
		if (instancia == null)
			instancia = new ConstructorElaborados();

		return instancia;
	}

	// Se usan los periodos típicos que suelen usar los robots
	// (consideraremos velas)
	// Se añade el periodo 1, para obtener el tamaño de vela, etc.
	public final static Integer[] periodosDParaParametros = new Integer[] { 1, 4, 20, 50 };

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
		Integer B = ElaboradosUtils.B; // DEFAULT
		Double umbralMaximo = ElaboradosUtils.SUBIDA_MAXIMA_POR_VELA; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 9) {
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
			B = Integer.valueOf(args[7]);
			umbralMaximo = Double.valueOf(args[8]);
		}

		File directorioEntrada = new File(directorioIn);
		File directorioSalida = new File(directorioOut);
		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosEntrada;
		HashMap<Integer, String> ordenNombresParametros;
		GestorFicheros gestorFicheros = new GestorFicheros();
		ArrayList<File> ficherosEntradaEmpresas = gestorFicheros.listaFicherosDeDirectorio(directorioEntrada);

		MY_LOGGER.info(crearDefinicionTarget(S, X, R, M, F, B));

		String destino = "";
		Iterator<File> iterator = ficherosEntradaEmpresas.iterator();
		File ficheroGestionado;

		int i = 1;

		while (iterator.hasNext()) {

			gestorFicheros = new GestorFicheros();
			datosEntrada = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
			ficheroGestionado = iterator.next();

			if (i % 10 == 1) {
				MY_LOGGER.info("Empresa numero = " + i + " (" + ficheroGestionado.getName() + ")");
			}
			i++;

			datosEntrada = gestorFicheros
					.leeSoloParametrosNoElaboradosFicheroDeSoloUnaEmpresa(ficheroGestionado.getPath(), Boolean.FALSE);
			destino = directorioSalida + "/" + ficheroGestionado.getName();
			MY_LOGGER.debug("Ficheros entrada|salida -> " + ficheroGestionado.getAbsolutePath() + " | " + destino);
			ordenNombresParametros = gestorFicheros.getOrdenNombresParametrosLeidos();
			anadirParametrosElaboradosDeSoloUnaEmpresa(datosEntrada, ordenNombresParametros, S, X, R, M, F, B,
					umbralMaximo);
			gestorFicheros.creaFicheroDeSoloUnaEmpresa(datosEntrada, ordenNombresParametros, destino);
		}

		MY_LOGGER.info("FIN");
	}

	/**
	 * Calcula columnas ELABORADAS (incluido el TARGET) y las añade a la MATRIZ de
	 * datos pasada como parámetro.
	 * 
	 * @param datos                  MATRIZ de datos no elaborados. A ella se
	 *                               añadiran columnas elaboradas
	 * @param ordenNombresParametros Mapa de nombres de columnas y su orden de
	 *                               aparicion en la MATRIZ.
	 * @param S
	 * @param X
	 * @param R
	 * @param M
	 * @param B
	 * @param umbralMaximo
	 * @throws Exception
	 */
	public static void anadirParametrosElaboradosDeSoloUnaEmpresa(
			HashMap<String, HashMap<Integer, HashMap<String, String>>> datos,
			HashMap<Integer, String> ordenNombresParametros, Integer S, Integer X, Integer R, Integer M, Integer F,
			Integer B, Double umbralMaximo) throws Exception {

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

		HashMap<String, String> parametros = new HashMap<String, String>(1);
		Iterator<Integer> itAntiguedad;
		Set<Integer> periodos, antiguedades;
		HashMap<Integer, Estadisticas> estadisticasVolumenPorAntiguedad = new HashMap<Integer, Estadisticas>(1);
		HashMap<Integer, Estadisticas> estadisticasClosePorAntiguedad = new HashMap<Integer, Estadisticas>(1);
		HashMap<Integer, Estadisticas> estadisticasHighPorAntiguedad = new HashMap<Integer, Estadisticas>(1);
		HashMap<Integer, Estadisticas> estadisticasLowPorAntiguedad = new HashMap<Integer, Estadisticas>(1);
		HashMap<Integer, Estadisticas> estadisticasOpenPorAntiguedad = new HashMap<Integer, Estadisticas>(1);
		HashMap<Integer, Estadisticas> estadisticasCloseOpenPorAntiguedad = new HashMap<Integer, Estadisticas>(1);
//		HashMap<Integer, Estadisticas> estadisticasCloseHighPorAntiguedad = new HashMap<Integer, Estadisticas>(1);
//		HashMap<Integer, Estadisticas> estadisticasCloseLowPorAntiguedad = new HashMap<Integer, Estadisticas>(1);
		HashMap<Integer, Estadisticas> estadisticasOpenHighPorAntiguedad = new HashMap<Integer, Estadisticas>(1);
//		HashMap<Integer, Estadisticas> estadisticasOpenLowPorAntiguedad = new HashMap<Integer, Estadisticas>(1);
		HashMap<Integer, Estadisticas> estadisticasHighLowPorAntiguedad = new HashMap<Integer, Estadisticas>(1);
		Estadisticas estadisticasVolumen = new Estadisticas();
		Estadisticas estadisticasClose = new Estadisticas();
		Estadisticas estadisticasHigh = new Estadisticas();
		Estadisticas estadisticasLow = new Estadisticas();
		Estadisticas estadisticasOpen = new Estadisticas();
		Estadisticas estadisticasCloseOpen = new Estadisticas();
//		Estadisticas estadisticasCloseHigh = new Estadisticas();
//		Estadisticas estadisticasCloseLow = new Estadisticas();
		Estadisticas estadisticasOpenHigh = new Estadisticas();
//		Estadisticas estadisticasOpenLow = new Estadisticas();
		Estadisticas estadisticasHighLow = new Estadisticas();
		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasVolumenPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>(
				1);
		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasClosePorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>(
				1);
		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasHighPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>(
				1);
		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasLowPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>(
				1);
		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasOpenPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>(
				1);
		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasCloseOpenPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>(
				1);
//		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasCloseHighPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>(
//				1);
//		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasCloseLowPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>(
//				1);
		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasOpenHighPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>(
				1);
//		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasOpenLowPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>(
//				1);
		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasHighLowPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>(
				1);
		HashMap<Integer, String> ordenVolumenNombresParametrosElaborados = estadisticasVolumen
				.getOrdenNombresParametrosElaborados();
		HashMap<Integer, String> ordenCloseNombresParametrosElaborados = estadisticasClose
				.getOrdenNombresParametrosElaborados();
		HashMap<Integer, String> ordenHighNombresParametrosElaborados = estadisticasHigh
				.getOrdenNombresParametrosElaborados();
		HashMap<Integer, String> ordenLowNombresParametrosElaborados = estadisticasLow
				.getOrdenNombresParametrosElaborados();
		HashMap<Integer, String> ordenOpenNombresParametrosElaborados = estadisticasOpen
				.getOrdenNombresParametrosElaborados();
		HashMap<Integer, String> ordenCloseOpenNombresParametrosElaborados = estadisticasCloseOpen
				.getOrdenNombresParametrosElaborados();
//		HashMap<Integer, String> ordenCloseHighNombresParametrosElaborados = estadisticasCloseHigh
//				.getOrdenNombresParametrosElaborados();
//		HashMap<Integer, String> ordenCloseLowNombresParametrosElaborados = estadisticasCloseLow
//				.getOrdenNombresParametrosElaborados();
		HashMap<Integer, String> ordenOpenHighNombresParametrosElaborados = estadisticasOpenHigh
				.getOrdenNombresParametrosElaborados();
//		HashMap<Integer, String> ordenOpenLowNombresParametrosElaborados = estadisticasOpenLow
//				.getOrdenNombresParametrosElaborados();
		HashMap<Integer, String> ordenHighLowNombresParametrosElaborados = estadisticasHighLow
				.getOrdenNombresParametrosElaborados();
		Integer parametrosAcumulados = numeroParametrosEntrada;
		String auxClose, auxVolumen, auxHigh, auxLow, auxOpen;
		Integer antiguedadHistoricaMaxima;

		for (Integer periodo : periodosDParaParametros) {

			// Se guarda el orden de los datos elaborados
			for (int i = 0; i < ordenCloseNombresParametrosElaborados.size(); i++) {
				// CLOSE
				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
						ordenCloseNombresParametrosElaborados.get(i + 1) + periodo
								+ FINAL_NOMBRES_PARAMETROS_ELABORADOS._CLOSE.toString());
			}
			parametrosAcumulados += ordenCloseNombresParametrosElaborados.size();

			for (int i = 0; i < ordenHighNombresParametrosElaborados.size(); i++) {
				// HIGH
				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
						ordenHighNombresParametrosElaborados.get(i + 1) + periodo
								+ FINAL_NOMBRES_PARAMETROS_ELABORADOS._HIGH.toString());
			}
			parametrosAcumulados += ordenHighNombresParametrosElaborados.size();

			for (int i = 0; i < ordenLowNombresParametrosElaborados.size(); i++) {
				// LOW
				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
						ordenLowNombresParametrosElaborados.get(i + 1) + periodo
								+ FINAL_NOMBRES_PARAMETROS_ELABORADOS._LOW.toString());
			}
			parametrosAcumulados += ordenLowNombresParametrosElaborados.size();

			for (int i = 0; i < ordenOpenNombresParametrosElaborados.size(); i++) {
				// OPEN
				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
						ordenOpenNombresParametrosElaborados.get(i + 1) + periodo
								+ FINAL_NOMBRES_PARAMETROS_ELABORADOS._OPEN.toString());
			}
			parametrosAcumulados += ordenOpenNombresParametrosElaborados.size();

			for (int i = 0; i < ordenCloseOpenNombresParametrosElaborados.size(); i++) {
				// CLOSEOPEN
				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
						ordenCloseOpenNombresParametrosElaborados.get(i + 1) + periodo
								+ FINAL_NOMBRES_PARAMETROS_ELABORADOS._CLOSEOPEN.toString());
			}
			parametrosAcumulados += ordenCloseOpenNombresParametrosElaborados.size();

//			for (int i = 0; i < ordenCloseHighNombresParametrosElaborados.size(); i++) {
//				// CLOSEHIGH
//				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
//						ordenCloseHighNombresParametrosElaborados.get(i + 1) + periodo
//								+ FINAL_NOMBRES_PARAMETROS_ELABORADOS._CLOSEHIGH.toString());
//			}
//			parametrosAcumulados += ordenCloseHighNombresParametrosElaborados.size();

//			for (int i = 0; i < ordenCloseLowNombresParametrosElaborados.size(); i++) {
//				// CLOSELOW
//				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
//						ordenCloseLowNombresParametrosElaborados.get(i + 1) + periodo
//								+ FINAL_NOMBRES_PARAMETROS_ELABORADOS._CLOSELOW.toString());
//			}
//			parametrosAcumulados += ordenCloseLowNombresParametrosElaborados.size();

			for (int i = 0; i < ordenOpenHighNombresParametrosElaborados.size(); i++) {
				// OPENHIGH
				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
						ordenOpenHighNombresParametrosElaborados.get(i + 1) + periodo
								+ FINAL_NOMBRES_PARAMETROS_ELABORADOS._OPENHIGH.toString());
			}
			parametrosAcumulados += ordenOpenHighNombresParametrosElaborados.size();

//			for (int i = 0; i < ordenOpenLowNombresParametrosElaborados.size(); i++) {
//				// OPENLOW
//				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
//						ordenOpenLowNombresParametrosElaborados.get(i + 1) + periodo
//								+ FINAL_NOMBRES_PARAMETROS_ELABORADOS._OPENLOW.toString());
//			}
//			parametrosAcumulados += ordenOpenLowNombresParametrosElaborados.size();

			for (int i = 0; i < ordenHighLowNombresParametrosElaborados.size(); i++) {
				// HIGHLOW
				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
						ordenHighLowNombresParametrosElaborados.get(i + 1) + periodo
								+ FINAL_NOMBRES_PARAMETROS_ELABORADOS._HIGHLOW.toString());
			}
			parametrosAcumulados += ordenHighLowNombresParametrosElaborados.size();

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

							MY_LOGGER.debug("Empresa=" + empresa + " No hay datos para la vela --> " + (i + antiguedad)
									+ " Posible causa: el mercado estaba abierto cuando hemos ejecutado la descarga de datos");

						} else {
							auxVolumen = parametros.get("volumen");
							auxClose = parametros.get("close");
							auxHigh = parametros.get("high");
							auxLow = parametros.get("low");
							auxOpen = parametros.get("open");

							// Precio simple
							estadisticasClose.addValue(new Double(auxClose));
							estadisticasHigh.addValue(new Double(auxHigh));
							estadisticasLow.addValue(new Double(auxLow));
							estadisticasOpen.addValue(new Double(auxOpen));

							// Precio complejo
							estadisticasCloseOpen.addValue(new Double(auxClose) - new Double(auxOpen));
//							estadisticasCloseHigh.addValue(new Double(auxClose) - new Double(auxHigh));
//							estadisticasCloseLow.addValue(new Double(auxClose) - new Double(auxLow));
							estadisticasOpenHigh.addValue(new Double(auxOpen) - new Double(auxHigh));
//							estadisticasOpenLow.addValue(new Double(auxOpen) - new Double(auxLow));
							estadisticasHighLow.addValue(new Double(auxHigh) - new Double(auxLow));

							// Volumen
							estadisticasVolumen.addValue(new Double(auxVolumen));

							MY_LOGGER.debug("(antiguedad: " + antiguedad + ", periodo: " + periodo
									+ ") Metido para estadísticas: " + auxClose);
						}
					}

				} else {
					// Para los datos de antiguedad excesiva, se sale del bucle
					break;
				}

				// VALIDACIÓN DE ESTADíSTICAS
				// La empresa y la antigüedad no las usamos

				MY_LOGGER.debug("------------------>>>>>>> Periodo: " + periodo + ", n: " + estadisticasClose.getN());
				estadisticasVolumenPorAntiguedad.put(antiguedad, estadisticasVolumen);
				estadisticasClosePorAntiguedad.put(antiguedad, estadisticasClose);
				estadisticasHighPorAntiguedad.put(antiguedad, estadisticasHigh);
				estadisticasLowPorAntiguedad.put(antiguedad, estadisticasLow);
				estadisticasOpenPorAntiguedad.put(antiguedad, estadisticasOpen);
				estadisticasCloseOpenPorAntiguedad.put(antiguedad, estadisticasCloseOpen);
//				estadisticasCloseHighPorAntiguedad.put(antiguedad, estadisticasCloseHigh);
//				estadisticasCloseLowPorAntiguedad.put(antiguedad, estadisticasCloseLow);
				estadisticasOpenHighPorAntiguedad.put(antiguedad, estadisticasOpenHigh);
//				estadisticasOpenLowPorAntiguedad.put(antiguedad, estadisticasOpenLow);
				estadisticasHighLowPorAntiguedad.put(antiguedad, estadisticasHighLow);

				// Se limpia este almacén temporal
				estadisticasVolumen = new Estadisticas();
				estadisticasClose = new Estadisticas();
				estadisticasHigh = new Estadisticas();
				estadisticasLow = new Estadisticas();
				estadisticasOpen = new Estadisticas();
				estadisticasCloseOpen = new Estadisticas();
//				estadisticasCloseHigh = new Estadisticas();
//				estadisticasCloseLow = new Estadisticas();
				estadisticasOpenHigh = new Estadisticas();
//				estadisticasOpenLow = new Estadisticas();
				estadisticasHighLow = new Estadisticas();

			}

			estadisticasVolumenPorAntiguedadYPeriodo.put(periodo, estadisticasVolumenPorAntiguedad);
			estadisticasClosePorAntiguedadYPeriodo.put(periodo, estadisticasClosePorAntiguedad);
			estadisticasHighPorAntiguedadYPeriodo.put(periodo, estadisticasHighPorAntiguedad);
			estadisticasLowPorAntiguedadYPeriodo.put(periodo, estadisticasLowPorAntiguedad);
			estadisticasOpenPorAntiguedadYPeriodo.put(periodo, estadisticasOpenPorAntiguedad);
			estadisticasCloseOpenPorAntiguedadYPeriodo.put(periodo, estadisticasCloseOpenPorAntiguedad);
//			estadisticasCloseHighPorAntiguedadYPeriodo.put(periodo, estadisticasCloseHighPorAntiguedad);
//			estadisticasCloseLowPorAntiguedadYPeriodo.put(periodo, estadisticasCloseLowPorAntiguedad);
			estadisticasOpenHighPorAntiguedadYPeriodo.put(periodo, estadisticasOpenHighPorAntiguedad);
//			estadisticasOpenLowPorAntiguedadYPeriodo.put(periodo, estadisticasOpenLowPorAntiguedad);
			estadisticasHighLowPorAntiguedadYPeriodo.put(periodo, estadisticasHighLowPorAntiguedad);

			// Se limpia este almacén temporal
			estadisticasVolumenPorAntiguedad = new HashMap<Integer, Estadisticas>();
			estadisticasClosePorAntiguedad = new HashMap<Integer, Estadisticas>();
			estadisticasHighPorAntiguedad = new HashMap<Integer, Estadisticas>();
			estadisticasLowPorAntiguedad = new HashMap<Integer, Estadisticas>();
			estadisticasOpenPorAntiguedad = new HashMap<Integer, Estadisticas>();
			estadisticasCloseOpenPorAntiguedad = new HashMap<Integer, Estadisticas>();
//			estadisticasCloseHighPorAntiguedad = new HashMap<Integer, Estadisticas>();
//			estadisticasCloseLowPorAntiguedad = new HashMap<Integer, Estadisticas>();
			estadisticasOpenHighPorAntiguedad = new HashMap<Integer, Estadisticas>();
//			estadisticasOpenLowPorAntiguedad = new HashMap<Integer, Estadisticas>();
			estadisticasHighLowPorAntiguedad = new HashMap<Integer, Estadisticas>();
		}

		// ESTADÍSTICA --> A la vez: CALCULA y RELLENA
		periodos = estadisticasClosePorAntiguedadYPeriodo.keySet();
		Integer periodoActual;
		Iterator<Integer> itPeriodo = periodos.iterator();

		HashMap<String, String> mapaParamsVolumen, mapaParamsClose, mapaParamsHigh, mapaParamsLow, mapaParamsOpen,
				mapaParamsCloseOpen, mapaParamsCloseHigh, mapaParamsCloseLow, mapaParamsOpenHigh, mapaParamsOpenLow,
				mapaParamsHighLow;

		while (itPeriodo.hasNext()) { // periodo analizado
			periodoActual = itPeriodo.next();
			estadisticasVolumenPorAntiguedad = estadisticasVolumenPorAntiguedadYPeriodo.get(periodoActual);
			estadisticasClosePorAntiguedad = estadisticasClosePorAntiguedadYPeriodo.get(periodoActual);
			estadisticasHighPorAntiguedad = estadisticasHighPorAntiguedadYPeriodo.get(periodoActual);
			estadisticasLowPorAntiguedad = estadisticasLowPorAntiguedadYPeriodo.get(periodoActual);
			estadisticasOpenPorAntiguedad = estadisticasOpenPorAntiguedadYPeriodo.get(periodoActual);
			estadisticasCloseOpenPorAntiguedad = estadisticasCloseOpenPorAntiguedadYPeriodo.get(periodoActual);
//			estadisticasCloseHighPorAntiguedad = estadisticasCloseHighPorAntiguedadYPeriodo.get(periodoActual);
//			estadisticasCloseLowPorAntiguedad = estadisticasCloseLowPorAntiguedadYPeriodo.get(periodoActual);
			estadisticasOpenHighPorAntiguedad = estadisticasOpenHighPorAntiguedadYPeriodo.get(periodoActual);
//			estadisticasOpenLowPorAntiguedad = estadisticasOpenLowPorAntiguedadYPeriodo.get(periodoActual);
			estadisticasHighLowPorAntiguedad = estadisticasHighLowPorAntiguedadYPeriodo.get(periodoActual);

			antiguedades = estadisticasClosePorAntiguedad.keySet();
			itAntiguedad = antiguedades.iterator();

			while (itAntiguedad.hasNext()) { // antigüedad de la vela analizada: 0 1 2 3... (días hacia atrás)

				antiguedad = itAntiguedad.next();

				estadisticasVolumen = estadisticasVolumenPorAntiguedad.get(antiguedad);
				estadisticasClose = estadisticasClosePorAntiguedad.get(antiguedad);
				estadisticasHigh = estadisticasHighPorAntiguedad.get(antiguedad);
				estadisticasLow = estadisticasLowPorAntiguedad.get(antiguedad);
				estadisticasOpen = estadisticasOpenPorAntiguedad.get(antiguedad);
				estadisticasCloseOpen = estadisticasCloseOpenPorAntiguedad.get(antiguedad);
//				estadisticasCloseHigh = estadisticasCloseHighPorAntiguedad.get(antiguedad);
//				estadisticasCloseLow = estadisticasCloseLowPorAntiguedad.get(antiguedad);
				estadisticasOpenHigh = estadisticasOpenHighPorAntiguedad.get(antiguedad);
//				estadisticasOpenLow = estadisticasOpenLowPorAntiguedad.get(antiguedad);
				estadisticasHighLow = estadisticasHighLowPorAntiguedad.get(antiguedad);

				antiguedadHistoricaMaxima = antiguedad + periodoActual; // se analiza el periodo desde la vela analizada
																		// hacia atrás en el tiempo

				// Se cogen sólo los datos con la antiguedad dentro del rango a analizar
				if (antiguedadHistoricaMaxima < datosEmpresaEntrada.size()) {
					parametros = datosEmpresaEntrada.get(antiguedad);
					// COSTE DE COMPUTACION
					// <<<<<<<<-------

					mapaParamsVolumen = estadisticasVolumen.getParametros(periodoActual,
							FINAL_NOMBRES_PARAMETROS_ELABORADOS._VOLUMEN.toString(), Boolean.FALSE);
					mapaParamsClose = estadisticasClose.getParametros(periodoActual,
							FINAL_NOMBRES_PARAMETROS_ELABORADOS._CLOSE.toString(), Boolean.FALSE);
					mapaParamsHigh = estadisticasHigh.getParametros(periodoActual,
							FINAL_NOMBRES_PARAMETROS_ELABORADOS._HIGH.toString(), Boolean.FALSE);
					mapaParamsLow = estadisticasLow.getParametros(periodoActual,
							FINAL_NOMBRES_PARAMETROS_ELABORADOS._LOW.toString(), Boolean.FALSE);
					mapaParamsOpen = estadisticasOpen.getParametros(periodoActual,
							FINAL_NOMBRES_PARAMETROS_ELABORADOS._OPEN.toString(), Boolean.FALSE);
					mapaParamsCloseOpen = estadisticasCloseOpen.getParametros(periodoActual,
							FINAL_NOMBRES_PARAMETROS_ELABORADOS._CLOSEOPEN.toString(), Boolean.FALSE);
//					mapaParamsCloseHigh = estadisticasCloseHigh.getParametros(periodoActual,
//							FINAL_NOMBRES_PARAMETROS_ELABORADOS._CLOSEHIGH.toString(), Boolean.FALSE);
//					mapaParamsCloseLow = estadisticasCloseLow.getParametros(periodoActual,
//							FINAL_NOMBRES_PARAMETROS_ELABORADOS._CLOSELOW.toString(), Boolean.FALSE);
					mapaParamsOpenHigh = estadisticasOpenHigh.getParametros(periodoActual,
							FINAL_NOMBRES_PARAMETROS_ELABORADOS._OPENHIGH.toString(), Boolean.FALSE);
//					mapaParamsOpenLow = estadisticasOpenLow.getParametros(periodoActual,
//							FINAL_NOMBRES_PARAMETROS_ELABORADOS._OPENLOW.toString(), Boolean.FALSE);
					mapaParamsHighLow = estadisticasHighLow.getParametros(periodoActual,
							FINAL_NOMBRES_PARAMETROS_ELABORADOS._HIGHLOW.toString(), Boolean.FALSE);

					parametros.putAll(mapaParamsVolumen);
					parametros.putAll(mapaParamsClose);
					parametros.putAll(mapaParamsHigh);
					parametros.putAll(mapaParamsLow);
					parametros.putAll(mapaParamsOpen);
					parametros.putAll(mapaParamsCloseOpen);
//					parametros.putAll(mapaParamsCloseHigh);
//					parametros.putAll(mapaParamsCloseLow);
					parametros.putAll(mapaParamsOpenHigh);
//					parametros.putAll(mapaParamsOpenLow);
					parametros.putAll(mapaParamsHighLow);

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

		// FEATURES RESPECTO AL FIN DE SEMANA, MES, TRIMESTRE
		meterParametrosFinDeEtapaTemporal(datosEmpresaFinales, ordenNombresParametrosSalida);

//		// Se calculan parámetros elaborados ESTÁTICOS (por eso se coge sólo la vela 0).
//		// Parámetro SCREENER1: basado en el screener que hemos visto que
//		// funciona:
//		// EPSgrowthNextYear > 0, CurrentRatio > 2,QuickRatio > 2, LongTermDebt <
//		// 0.0, InstitutionalOwnership > 10(%)
//		// Valores: Si no se tienen los datos o no se cumplen las condiciones, será 0.
//		// Si sí se cumple todo, será 1
//		String SCREENER1 = "0";
//
//		try {
//			parametros = datosEmpresaEntrada.get(0);
//			String EPSgrowthNextYear = parametros.get("EPS next Y");
//			String CurrentRatio = parametros.get("Current Ratio");
//			String QuickRatio = parametros.get("Quick Ratio");
//			String LongTermDebt = parametros.get("LT Debt/Eq");
//			String InstitutionalOwnership = parametros.get("Inst Own");
//
//			Float EPSgrowthNextYearF = Float.valueOf(EPSgrowthNextYear);
//			Float CurrentRatioF = Float.valueOf(CurrentRatio);
//			Float QuickRatioF = Float.valueOf(QuickRatio);
//			Float LongTermDebtF = Float.valueOf(LongTermDebt);
//			Float InstitutionalOwnershipF = Float.valueOf(InstitutionalOwnership);
//
//			// Cálculo del parámetro
//			if (EPSgrowthNextYearF > 0F && CurrentRatioF > 2F && QuickRatioF > 2F && LongTermDebtF > 0F
//					&& InstitutionalOwnershipF > 10F) {
//				SCREENER1 = "1";
//			}
//
//		} catch (Exception e) {
//			MY_LOGGER.debug("La empresa " + empresa + " no tiene alguno de los parámetros necesarios para calcular "
//					+ "el parámetro elaborado SCREENER1, o directamente no tiene ningún parámetro para la primera vela. Se pone SCREENER1=0");
//		}
//
//		MY_LOGGER.debug("SCREENER1: " + SCREENER1 + " para la empresa " + empresa);
//
//		// Se añade el parámetro SCREENER1
//		ordenNombresParametrosSalida.put(ordenNombresParametrosSalida.size(), "SCREENER1");
//		parametrosAcumulados++;
//
//		// Se añade el parámetro HYPE1
//		ordenNombresParametrosSalida.put(ordenNombresParametrosSalida.size(), "HYPE1");
//		parametrosAcumulados++;
//
//		// Se añade el parámetro HYPE2
//		ordenNombresParametrosSalida.put(ordenNombresParametrosSalida.size(), "HYPE2");
//		parametrosAcumulados++;
//
//		// Se añade el parámetro HYPE3
//		ordenNombresParametrosSalida.put(ordenNombresParametrosSalida.size(), "HYPE3");
//		parametrosAcumulados++;
//
//		// Se añade el parámetro HYPE4
//		ordenNombresParametrosSalida.put(ordenNombresParametrosSalida.size(), "HYPE4");
//		parametrosAcumulados++;

		// Aniado el TARGET
		// Target=0 es que no se cumple. 1 es que sí. TARGET_INVALIDO es que no se puede
		// calcular
		String target = TARGET_INVALIDO;

		antiguedades = datosEmpresaEntrada.keySet();

		if (antiguedades != null && !antiguedades.isEmpty()) {

			Integer antiguedadMaxima = Collections.max(antiguedades);
			Iterator<Integer> itAntiguedadTarget = datosEmpresaEntrada.keySet().iterator();
			HashMap<Integer, String> antiguedadYTarget = new HashMap<Integer, String>();

			while (itAntiguedadTarget.hasNext()) {

				antiguedad = itAntiguedadTarget.next();

				if (antiguedad >= X + M) {

					if (antiguedadMaxima < antiguedad + X) {
						// El periodo hacia atrás en el tiempo son X velas, desde el instante analizado
						target = TARGET_INVALIDO;
						break;

					} else {
						target = calcularTarget(empresa, datosEmpresaEntrada, antiguedad, S, X, R, M, F, B,
								umbralMaximo);
					}
				} else {
					// La antiguedad es demasiado reciente para ver si es estable en X+M
					target = TARGET_INVALIDO;
				}
				antiguedadYTarget.put(antiguedad, target);
			}

			// Auxiliares
//			String Sprecio, Svolumen, Ssma20Precio, Ssma50Precio, Ssma7Precio, Ssma4Precio, Ssma20Volumen, Ssma7Volumen,
//					Ssma4Volumen, Smaximo4Precio, Smaximo7Precio, Sstdsma4Precio, Sstdsma20Precio;
//			Float precio, volumen, sma20Precio, sma50Precio, sma7Precio, sma4Precio, sma20Volumen, sma7Volumen,
//					sma4Volumen, maximo4Precio, maximo7Precio, stdsma4Precio, stdsma20Precio;
//			String HYPE1 = "0", HYPE2 = "0", HYPE3 = "0", HYPE4 = "0";

			// Se rellena el target en los datos de entrada tras el analisis, al final de
			// todos los parametros
			Iterator<Integer> itAntiguedadDatos = datosEmpresaFinales.keySet().iterator();
			int indiceNuevoItem = ordenNombresParametrosSalida.size();
			ordenNombresParametrosSalida.put(indiceNuevoItem, "TARGET");
			while (itAntiguedadDatos.hasNext()) {
				antiguedad = itAntiguedadDatos.next();
				parametros = datosEmpresaFinales.get(antiguedad);

//				// Se AÑADEN parámetros elaborados ESTÁTICOS
//				parametros.put("SCREENER1", SCREENER1);
//
//				// Se AÑADEN parámetros elaborados DINÁMICOS. Para ello deben estar ya guardados
//				// el resto de parámetros que se referencien.
//
//				// PARÁMETRO HYPE1:
//				// PATRÓN DIP BUY
//				// PRECIO -> Menor que media 4, 7 y 20
//				// VOLUMEN -> Mayor que media 4, 7 y 20
//				// Valores: Si no se tienen los datos, será null. Si no se cumplen las
//				// condiciones, será 0.
//				// Si sí se cumple todo, será 1
//				Sprecio = parametros.get("close");
//				Svolumen = parametros.get("volumen");
//				Ssma20Precio = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"20_PRECIO");
//				Ssma7Precio = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"7_PRECIO");
//				Ssma4Precio = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"4_PRECIO");
//				Ssma20Volumen = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"20_VOLUMEN");
//				Ssma7Volumen = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"7_VOLUMEN");
//				Ssma4Volumen = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"4_VOLUMEN");
//				if (Sprecio != null && Svolumen != null && Ssma4Precio != null && Ssma7Precio != null
//						&& Ssma20Precio != null && Ssma4Volumen != null && Ssma7Volumen != null
//						&& Ssma20Volumen != null) {
//					precio = Float.valueOf(Sprecio);
//					volumen = Float.valueOf(Svolumen);
//					sma4Precio = Float.valueOf(Ssma4Precio);
//					sma7Precio = Float.valueOf(Ssma7Precio);
//					sma20Precio = Float.valueOf(Ssma20Precio);
//					sma4Volumen = Float.valueOf(Ssma4Volumen);
//					sma7Volumen = Float.valueOf(Ssma7Volumen);
//					sma20Volumen = Float.valueOf(Ssma20Volumen);
//					if (precio < sma4Precio && precio < sma7Precio && precio < sma20Precio && volumen > sma4Volumen
//							&& volumen > sma7Volumen && volumen > sma20Volumen) {
//						HYPE1 = "1";
//					} else {
//						HYPE1 = "0";
//					}
//				} else {
//					HYPE1 = "null";
//				}
//
//				// PARÁMETRO HYPE2:
//				// PRECIO -> Mayor que sma7 y SMA20
//				// PRECIO -> mayor (o igual, ya que será el máximo xD) que el máximo de 3 velas,
//				// en positivo
//				// VOLUMEN -> 3 veces mayor que SMA7
//				// Valores: Si no se tienen los datos, será null. Si no se cumplen las
//				// condiciones, será 0.
//				// Si sí se cumple todo, será 1
//				Sprecio = parametros.get("close");
//				Svolumen = parametros.get("volumen");
//				Ssma20Precio = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"20_PRECIO");
//				Ssma7Precio = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"7_PRECIO");
//				Ssma7Volumen = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"7_VOLUMEN");
//				Smaximo4Precio = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"4_PRECIO");
//				if (Sprecio != null && Svolumen != null && Ssma20Precio != null && Ssma7Precio != null
//						&& Ssma7Volumen != null && Smaximo4Precio != null) {
//					precio = Float.valueOf(Sprecio);
//					volumen = Float.valueOf(Svolumen);
//					sma20Precio = Float.valueOf(Ssma20Precio);
//					sma7Precio = Float.valueOf(Ssma7Precio);
//					sma7Volumen = Float.valueOf(Ssma7Volumen);
//					maximo4Precio = Float.valueOf(Smaximo4Precio);
//					if (precio > sma7Precio && precio > sma20Precio && precio >= maximo4Precio
//							&& volumen > 3 * sma7Volumen) {
//						HYPE2 = "1";
//					} else {
//						HYPE2 = "0";
//					}
//				} else {
//					HYPE2 = "null";
//				}
//
//				// PARÁMETRO HYPE3:
//				// PATRÓN subida VBLT 22/1/2020
//				// Precio: mayor que SMA20, pero menor que SMA4, SMA7
//				// Volumen: mayor que SMA4, pero menor que SMA7, SMA20
//				// Valores: Si no se tienen los datos, será null. Si no se cumplen las
//				// condiciones, será 0.
//				// Si sí se cumple todo, será 1
//				Sprecio = parametros.get("close");
//				Svolumen = parametros.get("volumen");
//				Ssma20Precio = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"20_PRECIO");
//				Ssma7Precio = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"7_PRECIO");
//				Ssma4Precio = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"4_PRECIO");
//				Ssma20Volumen = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"20_VOLUMEN");
//				Ssma7Volumen = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"7_VOLUMEN");
//				Ssma4Volumen = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_+"4_VOLUMEN");
//				if (Sprecio != null && Svolumen != null && Ssma4Precio != null && Ssma7Precio != null
//						&& Ssma20Precio != null && Ssma4Volumen != null && Ssma7Volumen != null
//						&& Ssma20Volumen != null) {
//					precio = Float.valueOf(Sprecio);
//					volumen = Float.valueOf(Svolumen);
//					sma4Precio = Float.valueOf(Ssma4Precio);
//					sma7Precio = Float.valueOf(Ssma7Precio);
//					sma20Precio = Float.valueOf(Ssma20Precio);
//					sma4Volumen = Float.valueOf(Ssma4Volumen);
//					sma7Volumen = Float.valueOf(Ssma7Volumen);
//					sma20Volumen = Float.valueOf(Ssma20Volumen);
//					if (precio < sma4Precio && precio < sma7Precio && precio > sma20Precio && volumen > sma4Volumen
//							&& volumen < sma7Volumen && volumen < sma20Volumen) {
//						HYPE3 = "1";
//					} else {
//						HYPE3 = "0";
//					}
//				} else {
//					HYPE3 = "null";
//				}
//
//				// PARÁMETRO HYPE4:
//				// Sacado a ojo, leyendo las combinaciones de resultados, para eliminar los falsos positivos
//				// STD_SMA_4_PRECIO > 1.5
//				// STD_SMA_20_PRECIO > 0.5
//				// Valores: Si no se tienen los datos, será null. Si no se cumplen las
//				// condiciones, será 0.
//				// Si sí se cumple todo, será 1
//				Sstdsma4Precio = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.STD_SMA_+"4_PRECIO");
//				Sstdsma20Precio = parametros.get(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.STD_SMA_+"20_PRECIO");
//
//				if (Sstdsma4Precio != null && Sstdsma20Precio != null) {
//					stdsma4Precio = Float.valueOf(Sstdsma4Precio);
//					stdsma20Precio = Float.valueOf(Sstdsma20Precio);
//					if (stdsma4Precio > 1.5 && stdsma20Precio > 0.5) {
//						HYPE4 = "1";
//					} else {
//						HYPE4 = "0";
//					}
//				} else {
//					HYPE4 = "null";
//				}
//
//				// AÑADO PARÁMETROS. TAMBIÉN HAY QUE AÑADIRLO EN LA LÍNEA 358 (VER OTROS
//				// EJEMPLOS)
//				// Se añade HYPE1
//				parametros.put("HYPE1", HYPE1);
//
//				// Se añade HYPE2
//				parametros.put("HYPE2", HYPE2);
//
//				// Se añade HYPE3
//				parametros.put("HYPE3", HYPE3);
//
//				// Se añade HYPE4
//				parametros.put("HYPE4", HYPE4);

				// SE AÑADE EL TARGET
				parametros.put("TARGET", String.valueOf(antiguedadYTarget.get(antiguedad)));
				datosEmpresaFinales.replace(antiguedad, parametros);
			}

			// Se dejan pasar sólo los parámetros autorizados
			datosEmpresaFinales = filtrarParametrosAutorizados(datosEmpresaFinales);
			ordenNombresParametrosSalida = filtrarNombresParametrosAutorizados(ordenNombresParametrosSalida);

			// Vuelco todos los parámetros
			datosSalida.put(empresa, datosEmpresaFinales);
			datos = datosSalida;
			ordenNombresParametros.clear();
			ordenNombresParametros.putAll(ordenNombresParametrosSalida);
		}

	}

	/**
	 * @param S
	 * @param X
	 * @param R
	 * @param M
	 * @param F
	 * @return
	 */
	public static String crearDefinicionTarget(Integer S, Integer X, Integer R, Integer M, Integer F, Integer B) {

		String definicion = "DEFINICION DE TARGET --> Vemos los instantes [t1,t2,t3]. Ahora estamos en t1. El periodo [t1,t2] dura "
				+ X + " velas, en el que el precio debe subir >= " + S
				+ "% . Durante todas ellas, el precio de cierre nunca puede estar debajo del de t1*(1-B%), con B=" + B
				+ "% . El periodo [t2,t3] dura " + M
				+ " velas; durante TODAS ellas, el precio de cierre puede caer un poco, pero nunca por debajo de un "
				+ R + "% respecto del precio de t2. El precio en el instante t3 (es decir, tras "
				+ Integer.valueOf(X + M) + " velas desde ahora) debe ser >= " + F
				+ "% respecto del t2. Sólo entonces, Target=1";

		return definicion;

	}

	/**
	 * Calcula el TARGET. Analiza los periodos [t1,t2] y [t2,t3], donde el foco está
	 * en el tiempo t1. El precio debe subir durante [t1,t2] y no haber bajado
	 * demasiado en [t2,t3].
	 * 
	 * @param datosEmpresa Entrada MATRIZ de datos.
	 * @param antiguedad   Antigüedad (índice en el tiempo t2) de la vela analizada,
	 *                     dentro de la MATRIZ pasada como parámetro.
	 * @param S            Subida del precio de cierre durante [t1,t2]
	 * @param X            Duración del periodo [t1,t2]
	 * @param R            Caida ligera máxima permitida durante [t2,t3], en TODAS
	 *                     esas velas.
	 * @param M            Duración del periodo [t2,t3]
	 * @param F            Caida ligera permitida durante [t2,t3], en la ÚLTIMA
	 *                     vela.
	 * @param B            Caida ligera permitida durante [t1,t2], en TODAS esas
	 *                     velas.
	 * @param umbralMaximo Porcentaje máximo aceptado para la subida de cada vela
	 *                     respecto del total de 1 a X.
	 * 
	 * @return
	 */
	public static String calcularTarget(String empresa, HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada,
			Integer antiguedad, Integer S, Integer X, Integer R, Integer M, Integer F, Integer B, Double umbralMaximo) {

		String targetOut = TARGET_INVALIDO; // default

		Double subidaSPrecioTantoPorUno = (100 + S) / 100.0;
		Double subidaSmenosRPrecioTantoPorUno = (100 + S - R) / 100.0;
		Double subidaSmenosFPrecioTantoPorUno = (100 + S - F) / 100.0;
		Double bajadaBPrecioTantoPorUno = (100 - B) / 100.0;

		Boolean mCumplida = Boolean.FALSE;

		HashMap<String, String> datosAntiguedad = datosEmpresaEntrada.get(antiguedad); // vela analizada
		HashMap<String, String> datosAntiguedadX = datosEmpresaEntrada.get(antiguedad - X); // velas siguientes (precio
																							// debe subir)
		HashMap<String, String> datosAntiguedadXyM = datosEmpresaEntrada.get(antiguedad - X - M); // velas muy futuras

		if (datosAntiguedad == null || !datosAntiguedad.containsKey("close")) {
			MY_LOGGER.debug("Empresa=" + empresa + " -> Falta dato close en datosAntiguedad");
			return TARGET_INVALIDO;
		}
		if (datosAntiguedadX == null || !datosAntiguedadX.containsKey("close")) {
			MY_LOGGER.debug("Empresa=" + empresa + " -> Falta dato close en datosAntiguedadX");
			return TARGET_INVALIDO;
		}

		Double closeAntiguedad = Double.valueOf(datosAntiguedad.get("close"));
		Double closeAntiguedadX = Double.valueOf(datosAntiguedadX.get("close"));

		boolean cumpleSubidaS = closeAntiguedadX > closeAntiguedad * subidaSPrecioTantoPorUno;
		Double closeAntiguedadXyM, closeAntiguedadI, closeAntiguedadZ;
		if (cumpleSubidaS) {

			if (datosAntiguedadXyM == null) {
				mCumplida = Boolean.FALSE;
				MY_LOGGER.error("Empresa=" + empresa + " -> datosAntiguedadM es NULO para antiguedad=" + antiguedad
						+ ", M=" + M + " -> antiguedadM=" + (antiguedad - M) + ", X+M=" + Integer.valueOf(X + M)
						+ " -> antiguedadXyM=" + (antiguedad - X - M)
						+ " Posible causa: el mercado estaba abierto cuando hemos ejecutado la descarga de datos");
			} else {

				closeAntiguedadXyM = Double.valueOf(datosAntiguedadXyM.get("close"));

				// En la vela X+M el precio no debe caer más de un (X-F)% respecto del precio
				// actual
				boolean cumpleSubidaSmenosF = closeAntiguedadXyM > subidaSmenosFPrecioTantoPorUno * closeAntiguedad;
				if (cumpleSubidaSmenosF) {

					for (int i = 1; i <= X; i++) {
						Integer antiguedadI = antiguedad - i; // Voy hacia el futuro
						closeAntiguedadI = Double.valueOf(datosEmpresaEntrada.get(antiguedadI).get("close"));
						boolean cumpleEncimaDeB = closeAntiguedadI > bajadaBPrecioTantoPorUno * closeAntiguedad;
						if (cumpleEncimaDeB) {
							// El precio no debe bajar más de B
							MY_LOGGER.debug("---ATENCION ENCONTRADO TARGET 1---> EMPRESA: " + empresa
									+ " y antigüedad: " + antiguedad + " (Mes: " + datosAntiguedad.get("mes") + " Dia: "
									+ datosAntiguedad.get("dia") + " Hora: " + datosAntiguedad.get("hora") + ")");
							mCumplida = Boolean.TRUE;
						} else {
							// Se ha encontrado AL MENOS una vela posterior, en las (1 a X) siguientes,
							// con el precio por debajo de la caída mínima B
							// TODAS LAS VELAS de ese periodo TIENEN QUE ESTAR POR ENCIMA DE ESE UMBRAL DE
							// CAIDA
							mCumplida = Boolean.FALSE;
							break;
						}
					}
					for (int i = 1; i <= X; i++) {
						Integer antiguedadI = antiguedad - i; // Voy hacia el futuro
						closeAntiguedadI = Double.valueOf(datosEmpresaEntrada.get(antiguedadI).get("close"));
						if (mCumplida) {
							for (int z = X + 1; z <= X + M; z++) {
								Integer antiguedadZ = antiguedad - z; // Voy hacia el muy futuro

								closeAntiguedadZ = Double.valueOf(datosEmpresaEntrada.get(antiguedadZ).get("close"));
								boolean cumpleUmbralVelaZ = closeAntiguedadZ > subidaSmenosRPrecioTantoPorUno
										* closeAntiguedad;
								if (cumpleUmbralVelaZ) {
									// El precio no debe bajar más de X-R
									mCumplida = Boolean.TRUE;
								} else {
									// Se ha encontrado AL MENOS una vela posterior, en las (X+1 a X+M) siguientes,
									// con el precio por debajo de la caída mínima (S-R)
									// TODAS LAS VELAS de ese periodo TIENEN QUE ESTAR POR ENCIMA DE ESE UMBRAL DE
									// CAIDA
									mCumplida = Boolean.FALSE;
									break;
								}
							}
						}
					}

				} else {
					mCumplida = Boolean.FALSE;
				}

			}

		} else {
			// La S no se cumple
			mCumplida = Boolean.FALSE;
		}

		// Se descartan los targets=1 que no cumplan esta condición
		if (mCumplida) {
			Estadisticas e = new Estadisticas();
			for (int i = 1; i <= X; i++) {
				Integer antiguedadI = antiguedad - i; // Voy hacia el futuro
				closeAntiguedadI = Double.valueOf(datosEmpresaEntrada.get(antiguedadI).get("close"));
				e.addValue(closeAntiguedadI);
				MY_LOGGER.debug(closeAntiguedadI + ", ");
			}
			MY_LOGGER.debug("ANÁLISIS VARIABILIDAD--> EMPRESA: " + empresa + " -> Antigüedad: " + antiguedad + " (Mes: "
					+ datosAntiguedad.get("mes") + " Dia: " + datosAntiguedad.get("dia") + " Hora: "
					+ datosAntiguedad.get("hora") + ". Variabilidad: " + e.getVariacionRelativaMaxima()
					+ " y umbral máximo: " + umbralMaximo);
			// Ninguna vela puede superar a la media en una cantidad igual a un umbral
			if (e.getVariacionRelativaMaxima() > umbralMaximo) {
				mCumplida = Boolean.FALSE;
			}
			// La vela de mayor variación debe oscilar más que un umbral
//			Double umbralMinimo=3D;
//			if (e.getVariacionRelativaMaxima() < umbralMinimo) {
//				mCumplida = Boolean.FALSE;
//			}
		}

		if (mCumplida) {
			// La S sí se cumple, y la M también en todo el rango
			targetOut = "1";
			MY_LOGGER.debug("---ATENCION ENCONTRADO TARGET 1 --> EMPRESA: " + empresa + " -> Antigüedad: " + antiguedad
					+ " (Mes: " + datosAntiguedad.get("mes") + " Dia: " + datosAntiguedad.get("dia") + " Hora: "
					+ datosAntiguedad.get("hora") + ")");
		} else {
			targetOut = "0";
		}
		return targetOut;
	}

	/**
	 * Añade las features de FIN DE MES y FIN DE TRIMESTRE
	 * 
	 * @param datosEmpresaFinales          Mapa de empresas y sus parametros
	 * @param ordenNombresParametrosSalida Mapa de parametros acumulados, a los que
	 *                                     añadir estos nuevos
	 */
	public static void meterParametrosFinDeEtapaTemporal(HashMap<Integer, HashMap<String, String>> entrada,
			HashMap<Integer, String> ordenNombresParametrosSalida) {

		Calendar ahora = Calendar.getInstance();
		Calendar ultimoDiaMes = Estadisticas.calcularUltimoDiaDelMes(ahora.getTime());
		Calendar ultimoDiaTrimestre = Estadisticas.calcularUltimoDiaDelTrimestre(ahora.getTime());
//		Calendar ultimoDiaAnio = Estadisticas.calcularUltimoDiaDelAnio(ahora.getTime());

		int diasHastaFinMes = Estadisticas.restarTiempos(ahora.getTimeInMillis(), ultimoDiaMes.getTimeInMillis());
		int diasHastaFinTrimestre = Estadisticas.restarTiempos(ahora.getTimeInMillis(),
				ultimoDiaTrimestre.getTimeInMillis());
//		int diasHastaFinAnio = Estadisticas.restarTiempos(ahora.getTimeInMillis(), ultimoDiaAnio.getTimeInMillis());

		for (Integer clave : entrada.keySet()) {
			entrada.get(clave).put(OTROS_PARAMS_ELAB.DIAS_HASTA_FIN_MES.toString(), String.valueOf(diasHastaFinMes));
			entrada.get(clave).put(OTROS_PARAMS_ELAB.DIAS_HASTA_FIN_TRIMESTRE.toString(),
					String.valueOf(diasHastaFinTrimestre));
		}

		// AÑADIR LOS NUEVOS PARAMETROS
		int indiceNuevoItem = ordenNombresParametrosSalida.size();
		ordenNombresParametrosSalida.put(indiceNuevoItem, OTROS_PARAMS_ELAB.DIAS_HASTA_FIN_MES.toString());
		ordenNombresParametrosSalida.put(indiceNuevoItem + 1, OTROS_PARAMS_ELAB.DIAS_HASTA_FIN_TRIMESTRE.toString());

	}

	/**
	 * Se dejan pasar los parámetros que pertenezcan a una lista de autorizados. El
	 * resto, se descartan.
	 * 
	 * @param datosEmpresa
	 * @return
	 */
	public static HashMap<Integer, HashMap<String, String>> filtrarParametrosAutorizados(
			HashMap<Integer, HashMap<String, String>> datosEmpresaIniciales) {

		// Separo los parámetros por espacios o comas
		ArrayList<String> parametrosAutorizados = new ArrayList<String>(
				Arrays.asList(LISTA_PARAMETROS_AUTORIZADOS.split("\\s*,\\s*")));

		// Se añade el target a los autorizados
		parametrosAutorizados.add("TARGET");

		HashMap<Integer, HashMap<String, String>> datosEmpresaFinales = new HashMap<Integer, HashMap<String, String>>();
		Integer antiguedad;
		String parametro;

		Iterator<Integer> itAntiguedadDatos = datosEmpresaIniciales.keySet().iterator();
		while (itAntiguedadDatos.hasNext()) {
			antiguedad = itAntiguedadDatos.next();
			HashMap<String, String> parametrosEntrada = datosEmpresaIniciales.get(antiguedad);
			HashMap<String, String> parametrosSalida = new HashMap<String, String>(1);
			Iterator<String> itParametros = parametrosEntrada.keySet().iterator();
			while (itParametros.hasNext()) {
				parametro = itParametros.next();
				if (parametrosAutorizados.contains(parametro)) {
					// El parámetro está dentro de los autorizados. Se añade a la lista final
					// filtrada
					parametrosSalida.put(parametro, parametrosEntrada.get(parametro));
				}
			}
			// Se guarda el vector de parámetros filtrado
			datosEmpresaFinales.put(antiguedad, parametrosSalida);
		}

		return datosEmpresaFinales;

	}

	/**
	 * Se dejan pasar los parámetros que pertenezcan a una lista de autorizados. El
	 * resto, se descartan.
	 * 
	 * @param ordenNombresParametrosEntrada
	 * @return
	 */
	public static HashMap<Integer, String> filtrarNombresParametrosAutorizados(
			HashMap<Integer, String> ordenNombresParametrosEntrada) {
		HashMap<Integer, String> ordenNombresParametrosSalida = new HashMap<Integer, String>();

		// Separo los parámetros por espacios o comas
		ArrayList<String> parametrosAutorizados = new ArrayList<String>(
				Arrays.asList(LISTA_PARAMETROS_AUTORIZADOS.split("\\s*,\\s*")));

		// Se añade el target a los autorizados
		parametrosAutorizados.add("TARGET");

		Integer numeroParametrosEntrada = ordenNombresParametrosEntrada.size();
		int iSalida = 0;
		String nombreParametro;
		for (int i = 0; i < numeroParametrosEntrada; i++) {
			nombreParametro = ordenNombresParametrosEntrada.get(i);
			if (parametrosAutorizados.contains(nombreParametro)) {
				ordenNombresParametrosSalida.put(iSalida, nombreParametro);
				iSalida++;
			}
		}
		return ordenNombresParametrosSalida;
	}

}
