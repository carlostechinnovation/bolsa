package c30x.elaborados.construir;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import org.apache.log4j.Logger;

public class GestorFicheros {

	static Logger MY_LOGGER = Logger.getLogger(ConstructorElaborados.class);

	// Para modificarlos, vete a la constructora
	private HashMap<Integer, String> ordenNombresParametrosLeidos;

	public void main(String[] args) throws Exception {

		HashMap<String, HashMap<Integer, HashMap<String, String>>> datos = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
		final File dirEntrada = new File("/bolsa/pasado/limpios/");
		String dirSalida = "/bolsa/pasado/elaborados/";
		ArrayList<File> ficherosEntradaEmpresas = listaFicherosDeDirectorio(dirEntrada);

		String destino = "";
		Iterator<File> iterator = ficherosEntradaEmpresas.iterator();
		File ficheroGestionado;
		while (iterator.hasNext()) {
			ficheroGestionado = iterator.next();

			datos = leeSoloParametrosNoElaboradosFicheroDeSoloUnaEmpresa(ficheroGestionado.getPath(), Boolean.TRUE);
			destino = dirSalida + ficheroGestionado.getName();
			MY_LOGGER.debug("Fichero entrada: " + ficheroGestionado.getAbsolutePath());
			MY_LOGGER.debug("Fichero salida:  " + destino);
			HashMap<Integer, String> ordenNombresParametros = getOrdenNombresParametrosLeidos();
			creaFicheroDeSoloUnaEmpresa(datos, ordenNombresParametros, destino);
		}
	}

	/**
	 * 
	 */
	public GestorFicheros() {
		ordenNombresParametrosLeidos = new HashMap<Integer, String>();
		ordenNombresParametrosLeidos.put(0, "empresa");
		ordenNombresParametrosLeidos.put(1, "antiguedad");
		ordenNombresParametrosLeidos.put(2, "mercado");
		ordenNombresParametrosLeidos.put(3, "anio");
		ordenNombresParametrosLeidos.put(4, "mes");
		ordenNombresParametrosLeidos.put(5, "dia");
		ordenNombresParametrosLeidos.put(6, "hora");
		ordenNombresParametrosLeidos.put(7, "minuto");
		ordenNombresParametrosLeidos.put(8, "volumen");
		ordenNombresParametrosLeidos.put(9, "high");
		ordenNombresParametrosLeidos.put(10, "low");
		ordenNombresParametrosLeidos.put(11, "close");
		ordenNombresParametrosLeidos.put(12, "open");
		ordenNombresParametrosLeidos.put(13, "Insider Own");
		ordenNombresParametrosLeidos.put(14, "Debt/Eq");
		ordenNombresParametrosLeidos.put(15, "P/E");
		ordenNombresParametrosLeidos.put(16, "Dividend %");
		ordenNombresParametrosLeidos.put(17, "Employees");
		ordenNombresParametrosLeidos.put(18, "Inst Own");
		ordenNombresParametrosLeidos.put(19, "Market Cap");

	}

	/**
	 * 
	 * @param folder
	 * @return
	 */
	public ArrayList<File> listaFicherosDeDirectorio(final File folder) {
		ArrayList<File> salida = new ArrayList<File>();
		for (final File fileEntry : folder.listFiles()) {
			if (!fileEntry.isDirectory()) {
				salida.add(fileEntry);
			}
		}
		return salida;
	}

	/**
	 * 
	 * @param pathFichero
	 * @param leeSoloCabeceraYUnaFila
	 * @return
	 * @throws Exception
	 * @throws IOException
	 */
	public HashMap<String, HashMap<Integer, HashMap<String, String>>> leeSoloParametrosNoElaboradosFicheroDeSoloUnaEmpresa(
			final String pathFichero, final Boolean leeSoloCabeceraYUnaFila) throws Exception, IOException {

		Boolean esPrimeraLinea = Boolean.TRUE;
		MY_LOGGER.info("Fichero leído: " + pathFichero);
		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosSalida = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
		String row, empresa = "";
		String[] data;
		HashMap<Integer, HashMap<String, String>> datosEmpresa = new HashMap<Integer, HashMap<String, String>>();
		HashMap<String, String> datosParametrosEmpresa;

		File csvFile = new File(pathFichero);
		if (!csvFile.isFile()) {
			throw new FileNotFoundException("Fichero no válido: " + pathFichero);
		}

		BufferedReader csvReader = new BufferedReader(new FileReader(pathFichero));

		try {
			while ((row = csvReader.readLine()) != null) {
				MY_LOGGER.debug("Fila leída: " + row);
				if (esPrimeraLinea) {
					// En la primera línea está la cabecera de parametros
					// Se valida que el nombre recibido es igual que el usado en la constructora, y
					// en dicho orden
					data = row.split("\\|");
					// Sólo se leen los parámetros NO ELABORADOS. Meto esto para que se pueda usar
					// en pasos posteriores del proyecto
					for (int i = 0; i < Math.min(data.length, ordenNombresParametrosLeidos.size()); i++) {
						if (ordenNombresParametrosLeidos.get(i).compareTo(data[i]) != 0)
							throw new Exception(
									"El orden de los par�metros le�dos (sólo se tratan los NO ELABORADOS) en el fichero no es el que espera el gestor de ficheros");
					}
					esPrimeraLinea = Boolean.FALSE;

				} else {

					data = row.split("\\|");
					empresa = data[0];
					datosParametrosEmpresa = new HashMap<String, String>();

					if (!data[1].equals("null")) {

						// Se excluyen los 2 primeros elementos
						for (int i = 2; i < ordenNombresParametrosLeidos.size(); i++) {
							datosParametrosEmpresa.put(ordenNombresParametrosLeidos.get(i), data[i]);
						}

						// Se guarda la antiguedad
						datosEmpresa.put(Integer.parseInt(data[1]), datosParametrosEmpresa);
					}

					if (leeSoloCabeceraYUnaFila) {
						break;
					}
				}
			}

		} finally {
			csvReader.close();
		}
		// Se guarda la empresa
		datosSalida.put(empresa, datosEmpresa);

		return datosSalida;
	}

	/**
	 * 
	 * @param pathFichero
	 * @return
	 * @throws Exception
	 * @throws IOException
	 */
	public HashMap<String, HashMap<Integer, HashMap<String, String>>> leeTodosLosParametrosFicheroDeSoloUnaEmpresaYFilaMasReciente(
			final String pathFichero) throws Exception, IOException {

		Boolean esPrimeraLinea = Boolean.TRUE;
		MY_LOGGER.info("Fichero leído: " + pathFichero);
		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosSalida = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
		String row, empresa = "";
		String[] data;
		HashMap<Integer, HashMap<String, String>> datosEmpresa = new HashMap<Integer, HashMap<String, String>>();
		HashMap<String, String> datosParametrosEmpresa;

		File csvFile = new File(pathFichero);
		if (!csvFile.isFile()) {
			throw new FileNotFoundException("Fichero no válido: " + pathFichero);
		}

		BufferedReader csvReader = new BufferedReader(new FileReader(pathFichero));
		HashMap<Integer, String> parametrosNombresConOrden = new HashMap<Integer, String>();
		HashMap<Integer, String> parametrosValoresConOrden = new HashMap<Integer, String>();
		try {
			while ((row = csvReader.readLine()) != null) {
				MY_LOGGER.debug("Fila leída: " + row);
				if (esPrimeraLinea) {
					// En la primera línea está la cabecera de parametros
					data = row.split("\\|");

					// Se guardan los parámetros con su orden
					for (int i = 0; i < data.length; i++) {
						parametrosNombresConOrden.put(i, data[i]);
					}

					esPrimeraLinea = Boolean.FALSE;

				} else {
					data = row.split("\\|");
					for (int i = 0; i < data.length; i++) {
						parametrosValoresConOrden.put(i, data[i]);
					}
					empresa = data[0];
					datosParametrosEmpresa = new HashMap<String, String>();

					// Se excluyen los dos primeros elementos
					for (int i = 2; i < parametrosNombresConOrden.size(); i++) {
						datosParametrosEmpresa.put(parametrosNombresConOrden.get(i), parametrosValoresConOrden.get(i));
					}
					// Se guarda la antiguedad
					datosEmpresa.put(Integer.parseInt(data[1]), datosParametrosEmpresa);

					// No se sigue leyendo el fichero
					break;
				}
			}

		} finally {
			csvReader.close();
		}
		// Se guarda la empresa
		datosSalida.put(empresa, datosEmpresa);

		return datosSalida;
	}

	/**
	 * Escribe los datos (incluidos los elaborados) de una empresa en un fichero.
	 * 
	 * @param datos
	 * @param ordenNombresParametros
	 * @param pathAbsolutoFichero
	 * @throws IOException
	 */
	public void creaFicheroDeSoloUnaEmpresa(final HashMap<String, HashMap<Integer, HashMap<String, String>>> datos,
			final HashMap<Integer, String> ordenNombresParametros, final String pathAbsolutoFichero)
			throws IOException {

		FileWriter csvWriter = new FileWriter(pathAbsolutoFichero);
		Set<String> empresas = datos.keySet();
		String empresa = "";
		Integer antiguedad = -99999;
		HashMap<Integer, HashMap<String, String>> datosEmpresa = new HashMap<Integer, HashMap<String, String>>();
		// Se asume que el fichero solo tiene una empresa
		Iterator<String> itEmpresas = empresas.iterator();
		while (itEmpresas.hasNext()) {
			empresa = itEmpresas.next();
			break;
		}

		datosEmpresa = datos.get(empresa);
		Set<Integer> antiguedades = datosEmpresa.keySet();
		Iterator<Integer> itAntiguedades = antiguedades.iterator();
		HashMap<String, String> parametrosEmpresa;
		String nombreParametro, elementoAAnadir;
		// En la primera fila del fichero, se aniade una cabecera
		for (int i = 0; i < ordenNombresParametros.size(); i++) {
			csvWriter.append(ordenNombresParametros.get(i));
			// Se aniade el pipe en todos los elementos menos en el �ltimo
			if (i < ordenNombresParametros.size() - 1) {
				csvWriter.append("|");
			}
		}
		csvWriter.append("\n");

		Integer numParametros = ordenNombresParametros.size();
		while (itAntiguedades.hasNext()) {
			antiguedad = itAntiguedades.next();
			parametrosEmpresa = datosEmpresa.get(antiguedad);
			// Se a�ade empresa y antig�edad
			csvWriter.append(empresa + "|" + antiguedad);
			for (int i = 0; i < numParametros; i++) {
				nombreParametro = ordenNombresParametros.get(i);
				// Empresa y antig�edad se tratan aparte. No se a�aden aqu�
				if (nombreParametro == "empresa" || nombreParametro == "antiguedad") {
					// no se hace nada
				} else {

					// Se a�aden el resto de parametros
					elementoAAnadir = "|" + parametrosEmpresa.get(nombreParametro);
					MY_LOGGER.debug("Elemento a aniadir i=" + i + ": \"" + nombreParametro + "\" con valor: "
							+ parametrosEmpresa.get(nombreParametro));
					csvWriter.append(elementoAAnadir);
				}
			}
			// Siguiente fila
			if (itAntiguedades.hasNext())
				csvWriter.append("\n");
		}
		csvWriter.flush();
		csvWriter.close();

	}

	/**
	 * @return the ordenNombresParametrosLeidos
	 */
	public HashMap<Integer, String> getOrdenNombresParametrosLeidos() {
		return ordenNombresParametrosLeidos;
	}

}
