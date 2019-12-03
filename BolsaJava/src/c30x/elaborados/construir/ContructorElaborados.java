package c30x.elaborados.construir;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

public class ContructorElaborados {

	public enum NOMBRES_PARAMETROS {
		PRECIO, VOLUMEN;
	}

	// META-PARAMETRIZACIÓN
	// Periodo de la vela de entrada
	public final static String T_velaEntrada = "H";
	// x días
	public final static Integer HORAS_AL_DIA = 4;
	public final static Integer[] periodosHParaParametros = new Integer[] { 1 * HORAS_AL_DIA, 2 * HORAS_AL_DIA };

	// IMPORTANTE: se asume que los datos están ordenados de menor a mayor
	// antigüedad, y agrupados por empresa
	/**
	 * @param args
	 * @throws Exception
	 */
	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		// DATOS DE ENTRADA
		// Asumo que la clave es el ticker de la empresa.
		// Mapa de tickers de empresas (OJO en el orden)
		ArrayList<String> empresas = new ArrayList<String>();
		// El valor es un mapa, de antigüedad (en HORAS) + String de valores de
		// parámetros limpios
		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosEntrada = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();

		HashMap<String, String> datosParametrosEmpresaSNAP8 = new HashMap<String, String>();
		HashMap<String, String> datosParametrosEmpresaSNAP7 = new HashMap<String, String>();
		HashMap<String, String> datosParametrosEmpresaSNAP6 = new HashMap<String, String>();
		HashMap<String, String> datosParametrosEmpresaSNAP5 = new HashMap<String, String>();
		HashMap<String, String> datosParametrosEmpresaSNAP4 = new HashMap<String, String>();
		HashMap<String, String> datosParametrosEmpresaSNAP3 = new HashMap<String, String>();
		HashMap<String, String> datosParametrosEmpresaSNAP2 = new HashMap<String, String>();
		HashMap<String, String> datosParametrosEmpresaSNAP1 = new HashMap<String, String>();
		HashMap<String, String> datosParametrosEmpresaSNAP0 = new HashMap<String, String>();
		HashMap<Integer, HashMap<String, String>> datosEmpresa = new HashMap<Integer, HashMap<String, String>>();

		// Meto datos de EJEMPLO
		empresas.add("SNAP");

		datosParametrosEmpresaSNAP8.put(NOMBRES_PARAMETROS.PRECIO.toString(), "1");
		datosParametrosEmpresaSNAP7.put(NOMBRES_PARAMETROS.PRECIO.toString(), "2");
		datosParametrosEmpresaSNAP6.put(NOMBRES_PARAMETROS.PRECIO.toString(), "3");
		datosParametrosEmpresaSNAP5.put(NOMBRES_PARAMETROS.PRECIO.toString(), "4");
		datosParametrosEmpresaSNAP4.put(NOMBRES_PARAMETROS.PRECIO.toString(), "5");
		datosParametrosEmpresaSNAP3.put(NOMBRES_PARAMETROS.PRECIO.toString(), "6");
		datosParametrosEmpresaSNAP2.put(NOMBRES_PARAMETROS.PRECIO.toString(), "7");
		datosParametrosEmpresaSNAP1.put(NOMBRES_PARAMETROS.PRECIO.toString(), "8");
		datosParametrosEmpresaSNAP0.put(NOMBRES_PARAMETROS.PRECIO.toString(), "9");

		datosParametrosEmpresaSNAP8.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "5000");
		datosParametrosEmpresaSNAP7.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP6.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP5.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP4.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP3.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP2.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP1.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP0.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "2000");

		datosEmpresa.put(8, datosParametrosEmpresaSNAP8);
		datosEmpresa.put(7, datosParametrosEmpresaSNAP7);
		datosEmpresa.put(6, datosParametrosEmpresaSNAP6);
		datosEmpresa.put(5, datosParametrosEmpresaSNAP5);
		datosEmpresa.put(4, datosParametrosEmpresaSNAP4);
		datosEmpresa.put(3, datosParametrosEmpresaSNAP3);
		datosEmpresa.put(2, datosParametrosEmpresaSNAP2);
		datosEmpresa.put(1, datosParametrosEmpresaSNAP1);
		datosEmpresa.put(0, datosParametrosEmpresaSNAP0);
		datosEntrada.put("SNAP", datosEmpresa);

		// CÁLCULOS DE PARÁMETROS ELABORADOS
		Integer antiguedadEnAnalisis;

		for (int x = 0; x < empresas.size(); x++) {
			// EXTRACCIÓN DE DATOS DE LA EMPRESA
			datosEmpresa = datosEntrada.get(empresas.get(x));
			System.out.println("Empresa: " + empresas.get(x));
			for (Integer periodo : periodosHParaParametros) {
				// PARA CADA PERIODO DE CÁLCULO DE PARÁMETROS ELABORADOS...
				Estadisticas estadisticasPrecio = new Estadisticas();
				Estadisticas estadisticasVolumen = new Estadisticas();
				Iterator<Integer> iteradorAntiguedad = datosEmpresa.keySet().iterator();
				while (iteradorAntiguedad.hasNext()) {
					antiguedadEnAnalisis = iteradorAntiguedad.next();
					// Se cogen sólo los datos con la antigüedad dento del rango a analizar
					if (antiguedadEnAnalisis < periodo) {
						HashMap<String, String> parametros = datosEmpresa.get(antiguedadEnAnalisis);
						estadisticasPrecio.addValue(new Double(parametros.get(NOMBRES_PARAMETROS.PRECIO.toString())));
						estadisticasVolumen.addValue(new Double(parametros.get(NOMBRES_PARAMETROS.VOLUMEN.toString())));
					} else {
						// Para los datos de antigüedad excesiva, se sale del bucle
						break;
					}
				}

				// Validación
				// PRECIOS
				estadisticasPrecio.debugValidacion(periodo);
				// VOLÚMENES
				estadisticasVolumen.debugValidacion(periodo);

			}
		}
	}

	public static Integer tiempoEnHoras(String T) throws Exception {
		// Traducción a horas (hábiles, con Bolsa abierta)
		Integer horas = 0;
		if (T == "H")
			horas = 1;
		else if (T == "D")
			// HAY DÍAS QUE LA BOLSA ABRE SÓLO MEDIA JORNADA, ASÍ QUE ESTO NO ES TOTALMENTE
			// CORRECTO. Normalmente son 7h al día
			horas = 7;
		else if (T == "H")
			throw new Exception("Tiempo erróneo");

		return horas;
	}

}
