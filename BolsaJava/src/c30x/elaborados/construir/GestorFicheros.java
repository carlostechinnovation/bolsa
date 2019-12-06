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

public class GestorFicheros {

	private Boolean activarDebug;
	// Para modificarlos, vete a la constructora
	private HashMap<Integer, String> ordenNombresParametrosLeidos;

	public void main(String[] args) throws Exception {
		HashMap<String, HashMap<Integer, HashMap<String, String>>> datos = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
		final File directorio = new File("C:\\\\Users\\\\t151521\\\\git\\\\bolsa\\\\BolsaJava\\\\ficherosEjemplo");
		ArrayList<File> ficherosEntradaEmpresas = listaFicherosDeDirectorio(directorio);

		String destino = "";
		Iterator<File> iterator = ficherosEntradaEmpresas.iterator();
		File ficheroGestionado;
		while (iterator.hasNext()) {
			ficheroGestionado = iterator.next();
			System.out.println("Fichero entrada: " + ficheroGestionado.getAbsolutePath());
			datos = leeFicheroDeSoloUnaEmpresa(ficheroGestionado.getPath());
			destino = ficheroGestionado.getParentFile().getAbsolutePath() + "\\validaGestionFichero\\salida"
					+ ficheroGestionado.getName().substring(0, ficheroGestionado.getName().length() - 4) + ".csv";
			System.out.println("Fichero salida:  " + destino);
			HashMap<Integer, String> ordenNombresParametros = getOrdenNombresParametrosLeidos();
			creaFicheroDeSoloUnaEmpresa(datos, ordenNombresParametros, destino);
		}
		System.out.println("FIN");
	}

	/**
	 * 
	 * @param quieresDepurar
	 */
	public GestorFicheros(final Boolean quieresDepurar) {
		activarDebug = quieresDepurar;
		ordenNombresParametrosLeidos = new HashMap<Integer, String>();
		ordenNombresParametrosLeidos.put(0, "empresa");
		ordenNombresParametrosLeidos.put(1, "antiguedad");
		ordenNombresParametrosLeidos.put(2, "anio");
		ordenNombresParametrosLeidos.put(3, "mes");
		ordenNombresParametrosLeidos.put(4, "dia");
		ordenNombresParametrosLeidos.put(5, "hora");
		ordenNombresParametrosLeidos.put(6, "minuto");
		ordenNombresParametrosLeidos.put(7, "volumen");
		ordenNombresParametrosLeidos.put(8, "high");
		ordenNombresParametrosLeidos.put(9, "low");
		ordenNombresParametrosLeidos.put(10, "close");
		ordenNombresParametrosLeidos.put(11, "open");
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
	 * @return
	 * @throws IOException
	 */
	public HashMap<String, HashMap<Integer, HashMap<String, String>>> leeFicheroDeSoloUnaEmpresa(
			final String pathFichero) throws Exception, IOException {
		Boolean esPrimeraLinea = Boolean.TRUE;
		if (activarDebug)
			System.out.println("Fichero le�do: " + pathFichero);
		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosSalida = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
		String row, empresa = "";
		String[] data;
		HashMap<Integer, HashMap<String, String>> datosEmpresa = new HashMap<Integer, HashMap<String, String>>();
		HashMap<String, String> datosParametrosEmpresa = new HashMap<String, String>();
		File csvFile = new File(pathFichero);
		if (!csvFile.isFile()) {
			throw new FileNotFoundException("Fichero no v�lido: " + pathFichero);
		}
		BufferedReader csvReader = new BufferedReader(new FileReader(pathFichero));

		while ((row = csvReader.readLine()) != null) {
			if (activarDebug)
				System.out.println("Fila le�da: " + row);
			if (esPrimeraLinea) {
				// En la primera l�nea est� la cabecera de par�metros
				// Se valida que el nombre recibido es igual que el usado en la constructora, y
				// en dicho orden
				data = row.split("\\|");
				for (int i = 0; i < data.length; i++) {
					if (ordenNombresParametrosLeidos.get(i).compareTo(data[i]) != 0)
						throw new Exception(
								"El orden de los par�metros le�dos en el fichero no es el que espera el gestor de ficheros");
				}
				esPrimeraLinea = Boolean.FALSE;
			} else {
				data = row.split("\\|");
				empresa = data[0];
				datosParametrosEmpresa = new HashMap<String, String>();
				// Se excluyen los 2 primeros elementos
				for (int i = 2; i < ordenNombresParametrosLeidos.size(); i++) {
					datosParametrosEmpresa.put(ordenNombresParametrosLeidos.get(i), data[i]);
				}

				// Se guarda la antig�edad
				datosEmpresa.put(Integer.parseInt(data[1]), datosParametrosEmpresa);
			}
		}
		csvReader.close();
		// Se guarda la empresa
		datosSalida.put(empresa, datosEmpresa);

		return datosSalida;
	}

	/**
	 * 
	 * @param datos
	 * @param directorio
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
		// Se asume que el fichero s�lo tiene una empresa
		Iterator<String> itEmpresas = empresas.iterator();
		while (itEmpresas.hasNext()) {
			empresa = itEmpresas.next();
			break;
		}

		datosEmpresa = datos.get(empresa);
		Set<Integer> antiguedades = datosEmpresa.keySet();
		Iterator<Integer> itAntiguedades = antiguedades.iterator();
		HashMap<String, String> parametrosEmpresa;
		Set<String> nombresParametros;
		String nombreParametro, elementoAAnadir;
		//En la primera fila del fichero, se a�ade una cabecera
		for (int i = 0; i < ordenNombresParametros.size(); i++) {
			csvWriter.append(ordenNombresParametros.get(i));
			//Se a�ade el pipe en todos los elementos menos en el �ltimo
			if(i<ordenNombresParametros.size()-1) {
				csvWriter.append("|");
			}
		}
		csvWriter.append("\n");
		
		while (itAntiguedades.hasNext()) {
			antiguedad = itAntiguedades.next();
			parametrosEmpresa = datosEmpresa.get(antiguedad);
			nombresParametros = parametrosEmpresa.keySet();
			// Se a�ade empresa y antig�edad
			csvWriter.append(empresa + "|" + antiguedad);
			// Las dos primeras no se leen (empresa y antig�edad)
			for (int i = 2; i < nombresParametros.size(); i++) {
				// Se a�aden el resto de par�metros
				nombreParametro = ordenNombresParametros.get(i);
				elementoAAnadir = "|" + parametrosEmpresa.get(nombreParametro);
				if (activarDebug)
					System.out.println("Elemento a a�adir: \"" + nombreParametro + "\" con valor: "
							+ parametrosEmpresa.get(nombreParametro));
				csvWriter.append(elementoAAnadir);
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
