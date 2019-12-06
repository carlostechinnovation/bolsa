package c30x.elaborados.construir;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

public class ConstructorElaborados {

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
	public static void main(String[] args) throws Exception {

		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosEntrada = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
		HashMap<Integer, String> ordenNombresParametros;
		final File directorio = new File("C:\\\\Users\\\\t151521\\\\git\\\\bolsa\\\\BolsaJava\\\\ficherosEjemplo");
		GestorFicheros gestorFicheros = new GestorFicheros(Boolean.TRUE);
		ArrayList<File> ficherosEntradaEmpresas = gestorFicheros.listaFicherosDeDirectorio(directorio);

		String destino = "";
		Iterator<File> iterator = ficherosEntradaEmpresas.iterator();
		File ficheroGestionado;
		while (iterator.hasNext()) {
			ficheroGestionado = iterator.next();
			System.out.println("Fichero entrada: " + ficheroGestionado.getAbsolutePath());
			datosEntrada = gestorFicheros.leeFicheroDeSoloUnaEmpresa(ficheroGestionado.getPath());
			destino = ficheroGestionado.getParentFile().getAbsolutePath() + "\\salidaElaborada\\salida"
					+ ficheroGestionado.getName().substring(0, ficheroGestionado.getName().length() - 4) + ".csv";
			System.out.println("Fichero salida:  " + destino);
			ordenNombresParametros = gestorFicheros.getOrdenNombresParametrosLeidos();
			anadirParametrosElaboradosDeSoloUnaEmpresa(datosEntrada, ordenNombresParametros);
			gestorFicheros.creaFicheroDeSoloUnaEmpresa(datosEntrada, ordenNombresParametros, destino);
		}
		System.out.println("FIN");
	}

	public static void anadirParametrosElaboradosDeSoloUnaEmpresa(
			HashMap<String, HashMap<Integer, HashMap<String, String>>> datos,
			HashMap<Integer, String> ordenNombresParametros) throws Exception {

		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosSalida = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
		HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada = new HashMap<Integer, HashMap<String, String>>();
		HashMap<Integer, HashMap<String, String>> datosEmpresaFinales = new HashMap<Integer, HashMap<String, String>>();

		// ORDEN DE PAR�METROS DE ENTRADA
		HashMap<Integer, String> ordenNombresParametrosSalida = new HashMap<Integer, String>();
		Integer numeroParametrosEntrada = ordenNombresParametros.size();
		for (int i = 0; i < numeroParametrosEntrada; i++) {
			ordenNombresParametrosSalida.put(i, ordenNombresParametros.get(i));
		}

		// C�LCULOS DE PAR�METROS ELABORADOS
		Integer antiguedad;
		String empresa = "";
		Set<String> empresas = datos.keySet();
		Iterator<String> itEmpresas = datos.keySet().iterator();
		if (empresas.size() != 1) {
			throw new Exception("Es est�n calculando par�metros elaborados de m�s de una empresa");
		} else {
			while (itEmpresas.hasNext())
				empresa = itEmpresas.next();
		}
		// EXTRACCI�N DE DATOS DE LA EMPRESA
		datosEmpresaEntrada = datos.get(empresa);
		System.out.println("Empresa: " + empresa);
		HashMap<String, String> parametros = new HashMap<String, String>();
		Iterator<Integer> itAntiguedad;
		Set<Integer> periodos, antiguedades;
		HashMap<Integer, Estadisticas> estadisticasPorAntiguedad = new HashMap<Integer, Estadisticas>();
		Estadisticas estadisticasPrecio = new Estadisticas();
		Estadisticas estadisticasVolumen = new Estadisticas();
		HashMap<Integer, HashMap<Integer, Estadisticas>> estadisticasPorAntiguedadYPeriodo = new HashMap<Integer, HashMap<Integer, Estadisticas>>();
		HashMap<Integer, String> ordenNombresParametrosElaborados = estadisticasPrecio
				.getOrdenNombresParametrosElaborados();
		Integer parametrosAcumulados = numeroParametrosEntrada;
		String auxPrecio, auxVolumen;
		Integer antiguedadHistoricaMaxima;
		for (Integer periodo : periodosHParaParametros) {

			// Se guarda el orden de los datos elaborados
			for (int i = 0; i < ordenNombresParametrosElaborados.size(); i++) {
				ordenNombresParametrosSalida.put(parametrosAcumulados + i,
						ordenNombresParametrosElaborados.get(i + 1) + periodo);
			}
			parametrosAcumulados += ordenNombresParametrosElaborados.size();
			itAntiguedad = datosEmpresaEntrada.keySet().iterator();
			while (itAntiguedad.hasNext()) {
				antiguedad = itAntiguedad.next();
				// PARA CADA PERIODO DE C�LCULO DE PAR�METROS ELABORADOS y cada antig�edad, que
				// ser� un GRUPO de
				// COLUMNAS...
				
				// Deben existir datos de una antigu�dadHist�rica= (antig�edad + periodo)
				antiguedadHistoricaMaxima = antiguedad + periodo;
				System.out.println("datosEmpresaEntrada.size(): " + datosEmpresaEntrada.size());
				if (DEPURAR) {
					System.out.println("Antig�edad: " + antiguedad);
				}
				if (antiguedadHistoricaMaxima < datosEmpresaEntrada.size()) {
					for (int i = 0; i < periodo; i++) {
						parametros = datosEmpresaEntrada.get(i + antiguedad);
						if (DEPURAR) {
							System.out.println("i + antig�edad: " + (i + antiguedad));
						}
						// Se toma el par�metro "close" para las estad�sticas de precio
						// Se toma el par�metro "volumen" para las estad�sticas de volumen
						auxPrecio = parametros.get("close");
						auxVolumen = parametros.get("volumen");
						estadisticasPrecio.addValue(new Double(auxPrecio));
						estadisticasVolumen.addValue(new Double(auxVolumen));
						if (DEPURAR) {
							System.out
									.println("(antig�edad: " + antiguedad + ", periodo: "+periodo+") Metido para estad�sticas: " + auxPrecio);
						}
					}
				} else {
					// Para los datos de antig�edad excesiva, se sale del bucle
					break;
				}
				// VALIDACI�N DE ESTAD�STICAS
				if (DEPURAR) {
					// La empresa y la antig�edad no las usamos
					estadisticasPrecio.debugValidacion(periodo);
				}
				estadisticasPorAntiguedad.put(antiguedad, estadisticasPrecio);
				// Se limpia este almac�n temporal
				estadisticasPrecio = new Estadisticas();
			}

			estadisticasPorAntiguedadYPeriodo.put(periodo, estadisticasPorAntiguedad);
		}

		// ESTAD�STICAS: ir� calculando y rellenando
		periodos = estadisticasPorAntiguedadYPeriodo.keySet();
		Integer periodoActual;
		Iterator<Integer> itPeriodo = periodos.iterator();
		while (itPeriodo.hasNext()) {
			periodoActual = itPeriodo.next();
			estadisticasPorAntiguedad = estadisticasPorAntiguedadYPeriodo.get(periodoActual);
			antiguedades = estadisticasPorAntiguedad.keySet();
			itAntiguedad = antiguedades.iterator();
			while (itAntiguedad.hasNext()) {
				antiguedad = itAntiguedad.next();
				estadisticasPrecio = estadisticasPorAntiguedad.get(antiguedad);
				antiguedadHistoricaMaxima = antiguedad + periodoActual;
				// Se cogen s�lo los datos con la antig�edad dentro del rango a analizar
				if (antiguedadHistoricaMaxima < datosEmpresaEntrada.size()) {
					parametros = datosEmpresaEntrada.get(antiguedad);
					// COSTE DE COMPUTACI�N
					// <<<<<<<<-------
					parametros.putAll(estadisticasPrecio.getParametros(periodoActual, Boolean.FALSE));
					// <<<<<<<------
				} else {
					// Para los datos de antig�edad excesiva, salgo del bucle
					break;
				}
				// ADICI�N DE PAR�METROS ELABORADOS AL HASHMAP
				datosEmpresaFinales.put(antiguedad, parametros);
			}
		}
		datosSalida.put(empresa, datosEmpresaFinales);
		datos = datosSalida;
		ordenNombresParametros.clear();
		ordenNombresParametros.putAll(ordenNombresParametrosSalida);
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
