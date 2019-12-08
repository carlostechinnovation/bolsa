package c30x.elaborados.construir;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import c30x.elaborados.construir.Estadisticas.FINAL_NOMBRES_PARAMETROS_ELABORADOS;

public class ConstructorElaborados {

	static Logger MY_LOGGER = Logger.getLogger(ConstructorElaborados.class);

	// META-PARAMETRIZACIÓN
	// Periodo de la vela de entrada
	public final static String T_velaEntrada = "H";
	// x días
	public final static Integer HORAS_AL_DIA = 4;
	public final static Integer[] periodosHParaParametros = new Integer[] { 1 * HORAS_AL_DIA, 2 * HORAS_AL_DIA };

	// Parámetros del TARGET (subida del S% en precio de close, tras X velas, y no
	// cae más de un R% dentro de las siguientes M velas posteriores)
	public final static Integer S = 20;
	public final static Integer X = 2;
	public final static Integer R = 10;
	public final static Integer M = 2;

	// IMPORTANTE: se asume que los datos están ordenados de menor a mayor
	// antigüedad, y agrupados por empresa

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		MY_LOGGER.info("INICIO");

		BasicConfigurator.configure();
		MY_LOGGER.setLevel(Level.INFO);

		String directorioIn = "/bolsa/pasado/limpios/"; // DEFAULT
		String directorioOut = "/bolsa/pasado/elaborados/"; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 2) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			directorioIn = args[0];
			directorioOut = args[1];
		}

		File directorioEntrada = new File(directorioIn);
		File directorioSalida= new File(directorioOut);
		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosEntrada = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
		HashMap<Integer, String> ordenNombresParametros;
		GestorFicheros gestorFicheros = new GestorFicheros(Boolean.TRUE);
		ArrayList<File> ficherosEntradaEmpresas = gestorFicheros.listaFicherosDeDirectorio(directorioEntrada);

		String destino = "";
		Iterator<File> iterator = ficherosEntradaEmpresas.iterator();
		File ficheroGestionado;
		while (iterator.hasNext()) {
			ficheroGestionado = iterator.next();

			datosEntrada = gestorFicheros.leeFicheroDeSoloUnaEmpresa(ficheroGestionado.getPath());
			destino = directorioSalida+"/" + ficheroGestionado.getName().substring(0, ficheroGestionado.getName().length() - 4)
					+ "elaborada.csv";
			MY_LOGGER.debug("Fichero entrada: " + ficheroGestionado.getAbsolutePath());
			MY_LOGGER.debug("Fichero salida:  " + destino);
			ordenNombresParametros = gestorFicheros.getOrdenNombresParametrosLeidos();
			anadirParametrosElaboradosDeSoloUnaEmpresa(datosEntrada, ordenNombresParametros);
			gestorFicheros.creaFicheroDeSoloUnaEmpresa(datosEntrada, ordenNombresParametros, destino);
		}
	}

	public static void anadirParametrosElaboradosDeSoloUnaEmpresa(
			HashMap<String, HashMap<Integer, HashMap<String, String>>> datos,
			HashMap<Integer, String> ordenNombresParametros) throws Exception {

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
			throw new Exception("Es están calculando parámetros elaborados de más de una empresa");
		} else {
			while (itEmpresas.hasNext())
				empresa = itEmpresas.next();
		}
		// EXTRACCIÓN DE DATOS DE LA EMPRESA
		datosEmpresaEntrada = datos.get(empresa);
		MY_LOGGER.debug("Empresa: " + empresa);
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
				// PARA CADA PERIODO DE CÁLCULO DE PARÁMETROS ELABORADOS y cada antigüedad, que
				// será un GRUPO de COLUMNAS...

				// Deben existir datos de una antiguëdadHistórica = (antigüedad + periodo)
				antiguedadHistoricaMaxima = antiguedad + periodo;
				MY_LOGGER.debug("datosEmpresaEntrada.size(): " + datosEmpresaEntrada.size());
				MY_LOGGER.debug("Antigüedad: " + antiguedad);
				if (antiguedadHistoricaMaxima < datosEmpresaEntrada.size()) {
					for (int i = 0; i < periodo; i++) {
						parametros = datosEmpresaEntrada.get(i + antiguedad);
						MY_LOGGER.debug("i + antigüedad: " + (i + antiguedad));
						// Se toma el parámetro "close" para las estadísticas de precio
						// Se toma el parámetro "volumen" para las estadísticas de volumen
						auxPrecio = parametros.get("close");
						auxVolumen = parametros.get("volumen");
						estadisticasPrecio.addValue(new Double(auxPrecio));
						estadisticasVolumen.addValue(new Double(auxVolumen));
						MY_LOGGER.debug("(antigüedad: " + antiguedad + ", periodo: " + periodo
								+ ") Metido para estadísticas: " + auxPrecio);
					}
				} else {
					// Para los datos de antigüedad excesiva, se sale del bucle
					break;
				}
				// VALIDACIÓN DE ESTADÍSTICAS
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

		// ESTADÍSTICAS: iré calculando y rellenando
		periodos = estadisticasPrecioPorAntiguedadYPeriodo.keySet();
		Integer periodoActual;
		Iterator<Integer> itPeriodo = periodos.iterator();
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
				// Se cogen sólo los datos con la antigüedad dentro del rango a analizar
				if (antiguedadHistoricaMaxima < datosEmpresaEntrada.size()) {
					parametros = datosEmpresaEntrada.get(antiguedad);
					// COSTE DE COMPUTACIÓN
					// <<<<<<<<-------
					parametros.putAll(estadisticasPrecio.getParametros(periodoActual,
							FINAL_NOMBRES_PARAMETROS_ELABORADOS._PRECIO.toString(), Boolean.FALSE));
					parametros.putAll(estadisticasVolumen.getParametros(periodoActual,
							FINAL_NOMBRES_PARAMETROS_ELABORADOS._VOLUMEN.toString(), Boolean.FALSE));
					// <<<<<<<------
				} else {
					// Para los datos de antigüedad excesiva, salgo del bucle
					break;
				}
				// ADICIÓN DE PARÁMETROS ELABORADOS AL HASHMAP
				datosEmpresaFinales.put(antiguedad, parametros);
			}
		}

		// Añado el TARGET
		Integer antiguedadX, antiguedadM;
		Double subidaSPrecioTantoPorUno = (100 + S ) / 100.0;
		Double caidaRPrecioTantoPorUno = (100 - R ) / 100.0;
		// Target=0 es que no se cumple. 1 es que sí. -1 es que no se puede calcular
		Integer target = -1;
		Boolean mCumplida = Boolean.FALSE;

		antiguedades = datosEmpresaEntrada.keySet();
		Integer antiguedadMaxima = Collections.max(antiguedades);
		HashMap<String, String> datosAntiguedad, datosAntiguedadX;
		Iterator<Integer> itAntiguedadTarget = datosEmpresaEntrada.keySet().iterator();
		HashMap<Integer, Integer> antiguedadYTarget = new HashMap<Integer, Integer>();
		while (itAntiguedadTarget.hasNext()) {
			antiguedad = itAntiguedadTarget.next();
			antiguedadM = antiguedad + M;
			if (antiguedad > M) {
				antiguedadX = antiguedad + X;
				if (antiguedadMaxima < antiguedadX) {
					// Estamos analizando un punto en el tiempo X datos anteriores
					target = -1;
					break;
				} else {
					// Si el precio actual ha subido S% tras X velas viejas, y si después, durante
					// todas las M velas nuevas, no ha caído más de R%, entonces Target=1
					datosAntiguedad = datosEmpresaEntrada.get(antiguedad);
					datosAntiguedadX = datosEmpresaEntrada.get(antiguedadX);
					if (Double.valueOf(datosAntiguedad.get("close")) >= (Double.valueOf(datosAntiguedadX.get("close"))
							* subidaSPrecioTantoPorUno)) {
						for (int i = 0; i < M; i++) {
							if (Double.valueOf(datosEmpresaEntrada.get(antiguedad).get("close"))
									* caidaRPrecioTantoPorUno < Double
											.valueOf(datosEmpresaEntrada.get(antiguedadM).get("close"))) {
								// No se cumple el target
								mCumplida = Boolean.FALSE;
								break;
							} else {
								mCumplida = Boolean.TRUE;
							}
						}
						if (mCumplida) {
							// La S sí se cumple, y la M también en todo el rango
							target = 1;
						}
					} else {
						// La S no se cumple
						target = 0;
					}
				}
			} else {
				// La antigüedad es demasiado reciente para ver si es estable en M
				target = -1;
			}
			antiguedadYTarget.put(antiguedad, target);
		}

		// Se rellena el target en los datos de entrada tras el análisis, al final de
		// todos los parámetros
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

	public static Integer tiempoEnHoras(String T) throws Exception {
		// Traducción a horas (hábiles, con Bolsa abierta)
		Integer horas = 0;
		if (T == "H")
			horas = 1;
		else if (T == "D")
			// HAY DÍAS QUE LA BOLSA ABRE SÓLO MEDIA JORNADA, ASÍ QUE ESTO NO ES TOTALMENTE
			// CORRECTO. Normalmente son 7h al día
			horas = HORAS_AL_DIA;
		else if (T == "H")
			throw new Exception("Tiempo erróneo");

		return horas;
	}

}
