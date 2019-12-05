package c30x.elaborados.construir;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

public class ContructorElaborados {

	public final static Boolean DEPURAR = Boolean.TRUE;

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

		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosEntrada = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
		final File directorio = new File("C:\\\\Users\\\\t151521\\\\git\\\\bolsa\\\\BolsaJava\\\\ficherosEjemplo");
		ArrayList<File> ficherosEntradaEmpresas = GestorFicheros.listaFicherosDeDirectorio(directorio);

		String destino = "";
		Iterator<File> iterator = ficherosEntradaEmpresas.iterator();
		File ficheroGestionado;
		while (iterator.hasNext()) {
			ficheroGestionado = iterator.next();
			System.out.println("Fichero entrada: " + ficheroGestionado.getAbsolutePath());
			datosEntrada = GestorFicheros.leeFicheroDeSoloUnaEmpresa(ficheroGestionado.getPath());
			destino = ficheroGestionado.getParentFile().getAbsolutePath() + "\\salidaElaborada\\salida"
					+ ficheroGestionado.getName().substring(0, ficheroGestionado.getName().length() - 4) + ".csv";
			System.out.println("Fichero salida:  " + destino);
			GestorFicheros.creaFicheroDeSoloUnaEmpresa(datosEntrada, destino);
		}
		System.out.println("FIN");
	}

	/**
	 * 
	 * @param datosEntrada
	 * @return
	 * @throws Exception
	 */
	public HashMap<String, HashMap<Integer, HashMap<String, String>>> anadirParametrosElaboradosDeSoloUnaEmpresa(
			final HashMap<String, HashMap<Integer, HashMap<String, String>>> datosEntrada) throws Exception {

		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosSalida = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
		HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada = new HashMap<Integer, HashMap<String, String>>();
		HashMap<Integer, HashMap<String, String>> datosEmpresaFinales = new HashMap<Integer, HashMap<String, String>>();

		// C�LCULOS DE PAR�METROS ELABORADOS
		Integer antiguedad;
		Iterator<String> itParametros;
		Set<String> nombresParametros;
		String nombreParametro;
		String empresa = "";
		Set<String> empresas = datosEntrada.keySet();
		Iterator<String> itEmpresas = datosEntrada.keySet().iterator();
		if (empresas.size() != 1) {
			throw new Exception("Es est�n calculando par�metros elaborados de m�s de una empresa");
		} else {
			while (itEmpresas.hasNext())
				empresa = itEmpresas.next();
		}
		// EXTRACCI�N DE DATOS DE LA EMPRESA
		datosEmpresaEntrada = datosEntrada.get(empresa);
		System.out.println("Empresa: " + empresa);
		HashMap<String, String> parametros = new HashMap<String, String>();
		Iterator<Integer> iteradorAntiguedad;
		Set<Integer> periodos, antiguedades;
		HashMap<Integer, Estadisticas> estadisticasPorAntiguedad = new HashMap<Integer, Estadisticas>();
		Estadisticas estadisticas = new Estadisticas();
		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>();
		for (Integer periodo : periodosHParaParametros) {
			// PARA CADA PERIODO DE C�LCULO DE PAR�METROS ELABORADOS, que ser� un GRUPO de
			// COLUMNAS...
			iteradorAntiguedad = datosEmpresaEntrada.keySet().iterator();
			while (iteradorAntiguedad.hasNext()) {
				antiguedad = iteradorAntiguedad.next();
				// Se cogen s�lo los datos con la antig�edad dentro del rango a analizar
				if (antiguedad < periodo) {
					parametros = datosEmpresaEntrada.get(antiguedad);
					nombresParametros = parametros.keySet();
					itParametros = nombresParametros.iterator();
					while (itParametros.hasNext()) {
						nombreParametro = itParametros.next();
						estadisticas.addValue(new Double(nombreParametro));
					}
				} else {
					// Para los datos de antig�edad excesiva, se sale del bucle
					break;
				}
				estadisticasPorAntiguedad.put(antiguedad, estadisticas);
			}

			// VALIDACI�N DE ESTAD�STICAS
			if (DEPURAR)
				estadisticas.debugValidacion(periodo);

			estadisticasPorAntiguedadYPeriodo.put(periodo, estadisticasPorAntiguedad);
		}

		// ESTAD�STICAS: ir� calculando y rellenando
		periodos = estadisticasPorAntiguedadYPeriodo.keySet();
		Integer periodoActual;
		while (periodos.iterator().hasNext()) {
			periodoActual = periodos.iterator().next();
			antiguedades = estadisticasPorAntiguedadYPeriodo.get(periodoActual).keySet();

			while (antiguedades.iterator().hasNext()) {
				antiguedad = antiguedades.iterator().next();
				estadisticas = estadisticasPorAntiguedad.get(antiguedad);
				// Se cogen s�lo los datos con la antig�edad dentro del rango a analizar
				if (antiguedad < periodoActual) {
					parametros = datosEmpresaEntrada.get(antiguedad);
					// COSTE DE COMPUTACI�N
					// <<<<<<<<-------
					parametros.putAll(estadisticas.getParametros(periodoActual, Boolean.FALSE));
					// <<<<<<<------
				} else {
					// Para los datos de antig�edad excesiva, LOS RELLENO CON UN VALOR INV�LIDO
					parametros.putAll(estadisticas.getParametros(periodoActual, Boolean.TRUE));
				}
				// ADICI�N DE PAR�METROS ELABORADOS AL HASHMAP
				datosEmpresaFinales.put(antiguedad, parametros);
			}
		}
		datosSalida.put(empresa, datosEmpresaFinales);
		return datosSalida;
	}

	public static Integer tiempoEnHoras(String T) throws Exception {
		// Traducci�n a horas (h�biles, con Bolsa abierta)
		Integer horas = 0;
		if (T == "H")
			horas = 1;
		else if (T == "D")
			// HAY D�AS QUE LA BOLSA ABRE S�LO MEDIA JORNADA, AS� QUE ESTO NO ES TOTALMENTE
			// CORRECTO. Normalmente son 7h al d�a
			horas = HORAS_AL_DIA;
		else if (T == "H")
			throw new Exception("Tiempo err�neo");

		return horas;
	}
}
