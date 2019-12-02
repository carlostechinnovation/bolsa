package c30x.elaborados.construir;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

public class ContructorElaborados {

	public enum NOMBRES_PARAMETROS {
		PRECIO, VOLUMEN;
	}

	// META-PARAMETRIZACI�N
	// Periodo de la vela de entrada
	public final static String T_velaEntrada = "H";
	// x d�as
	public final static Integer HORAS_AL_DIA = 4;
	public final static Integer[] periodosHParaParametros = new Integer[] { 1 * HORAS_AL_DIA, 2 * HORAS_AL_DIA };

	// IMPORTANTE: se asume que los datos est�n ordenados de menor a mayor
	// antig�edad, y agrupados por empresa
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
		// El valor es un mapa, de antig�edad (en HORAS) + String de valores de
		// par�metros limpios
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
		datosParametrosEmpresaSNAP8.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "5000");
		datosParametrosEmpresaSNAP7.put(NOMBRES_PARAMETROS.PRECIO.toString(), "2");
		datosParametrosEmpresaSNAP7.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP6.put(NOMBRES_PARAMETROS.PRECIO.toString(), "3");
		datosParametrosEmpresaSNAP6.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP5.put(NOMBRES_PARAMETROS.PRECIO.toString(), "4");
		datosParametrosEmpresaSNAP5.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP4.put(NOMBRES_PARAMETROS.PRECIO.toString(), "5");
		datosParametrosEmpresaSNAP4.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP3.put(NOMBRES_PARAMETROS.PRECIO.toString(), "6");
		datosParametrosEmpresaSNAP3.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP2.put(NOMBRES_PARAMETROS.PRECIO.toString(), "7");
		datosParametrosEmpresaSNAP2.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP1.put(NOMBRES_PARAMETROS.PRECIO.toString(), "8");
		datosParametrosEmpresaSNAP1.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), "1000");
		datosParametrosEmpresaSNAP0.put(NOMBRES_PARAMETROS.PRECIO.toString(), "9");
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

		// C�LCULOS DE PAR�METROS ELABORADOS
		Integer antiguedadEnAnalisis;

		for (int x = 0; x < empresas.size(); x++) {
			// EXTRACCI�N DE DATOS DE LA EMPRESA
			datosEmpresa = datosEntrada.get(empresas.get(x));
			System.out.println("Empresa: " + empresas.get(x));
			for (Integer periodo : periodosHParaParametros) {
				// PARA CADA PERIODO DE C�LCULO DE PAR�METROS ELABORADOS...
				Estadisticas estadisticasPrecio = new Estadisticas();
				Estadisticas estadisticasVolumen = new Estadisticas();
				Iterator<Integer> iteradorAntiguedad = datosEmpresa.keySet().iterator();
				while (iteradorAntiguedad.hasNext()) {
					antiguedadEnAnalisis = iteradorAntiguedad.next();
					// Se cogen s�lo los datos con la antig�edad dento del rango a analizar
					if (antiguedadEnAnalisis < periodo) {
						HashMap<String, String> parametros = datosEmpresa.get(antiguedadEnAnalisis);
						estadisticasPrecio.addValue(new Double(parametros.get(NOMBRES_PARAMETROS.PRECIO.toString())));
						estadisticasVolumen.addValue(new Double(parametros.get(NOMBRES_PARAMETROS.VOLUMEN.toString())));
					} else {
						// Para los datos de antig�edad excesiva, se sale del bucle
						break;
					}
				}

				// VALIDACI�N DE ENTRADA
				// Si no se tienen todos los datos del periodo (por ejemplo, para una media de
				// 200 d�as, 200*7 valores hacia atr�s), lanzar� excepci�n
				System.out.println("********************************************************");
				if (estadisticasPrecio.getN() != periodo) {
					throw new Exception("PRECIO El n�mero de datos a analizar no es el adecuado. Se usan "
							+ estadisticasPrecio.getN() + " y se necesitan " + periodo);
				} else {
					System.out.println("PRECIO Se tienen " + estadisticasPrecio.getN() + " y se usan " + periodo);
					for (int i = 0; i < estadisticasPrecio.getN(); i++) {
						System.out.print(estadisticasPrecio.getElement(i) + ", ");
					}
					System.out.println("");
				}
				if (estadisticasVolumen.getN() != periodo) {
					throw new Exception("VOLUMEN El n�mero de datos a analizar no es el adecuado. Se usan "
							+ estadisticasVolumen.getN() + " y se necesitan " + periodo);
				} else {
					System.out.println("VOLUMEN Se usan " + estadisticasVolumen.getN() + " y se necesitan " + periodo);
					for (int i = 0; i < estadisticasVolumen.getN(); i++) {
						System.out.print(estadisticasVolumen.getElement(i) + ", ");
					}
					System.out.println("");
				}

				// C�LCULO
				// Validaci�n
				// PRECIOS
				estadisticasPrecio.debugValidacion();
				// VOL�MENES
				estadisticasVolumen.debugValidacion();

			}
		}
	}

	public static Integer tiempoEnHoras(String T) throws Exception {
		// Traducci�n a horas (h�biles, con Bolsa abierta)
		Integer horas = 0;
		if (T == "H")
			horas = 1;
		else if (T == "D")
			// HAY D�AS QUE LA BOLSA ABRE S�LO MEDIA JORNADA, AS� QUE ESTO NO ES TOTALMENTE
			// CORRECTO. Normalmente son 7h al d�a
			horas = 7;
		else if (T == "H")
			throw new Exception("Tiempo err�neo");

		return horas;
	}

}
