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

import c30x.elaborados.construir.ContructorElaborados.NOMBRES_PARAMETROS;

public class GestorFicheros {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
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
			destino = ficheroGestionado.getParentFile().getAbsolutePath() + "\\salidaElaborada\\salida"
					+ ficheroGestionado.getName().substring(0, ficheroGestionado.getName().length() - 4) + ".csv";
			System.out.println("Fichero salida:  " + destino);
			creaFicheroDeSoloUnaEmpresa(datos, destino);
		}
		System.out.println("FIN");
	}

	/**
	 * 
	 * @param folder
	 * @return
	 */
	public static ArrayList<File> listaFicherosDeDirectorio(final File folder) {
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
	public static HashMap<String, HashMap<Integer, HashMap<String, String>>> leeFicheroDeSoloUnaEmpresa(
			final String pathFichero) throws IOException {
		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosSalida = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
		String row, empresa = "";
		String[] data;
		HashMap<Integer, HashMap<String, String>> datosEmpresa = new HashMap<Integer, HashMap<String, String>>();
		HashMap<String, String> datosParametrosEmpresa = new HashMap<String, String>();
		File csvFile = new File(pathFichero);
		if (!csvFile.isFile()) {
			throw new FileNotFoundException("Fichero no válido: " + pathFichero);
		}
		BufferedReader csvReader = new BufferedReader(new FileReader(pathFichero));
		while ((row = csvReader.readLine()) != null) {
			data = row.split(";");
			empresa = data[0];
			datosParametrosEmpresa = new HashMap<String, String>();
			datosParametrosEmpresa.put(NOMBRES_PARAMETROS.PRECIO.toString(), data[2]);
			datosParametrosEmpresa.put(NOMBRES_PARAMETROS.VOLUMEN.toString(), data[3]);

			datosEmpresa.put(Integer.parseInt(data[1]), datosParametrosEmpresa);
		}
		datosSalida.put(empresa, datosEmpresa);

		csvReader.close();
		return datosSalida;
	}

	/**
	 * 
	 * @param datos
	 * @param directorio
	 * @throws IOException
	 */
	public static void creaFicheroDeSoloUnaEmpresa(
			final HashMap<String, HashMap<Integer, HashMap<String, String>>> datos, final String pathAbsolutoFichero)
			throws IOException {
		FileWriter csvWriter = new FileWriter(pathAbsolutoFichero);
		Set<String> empresas = datos.keySet();
		String empresa = "";
		Integer antiguedad = -99999;
		HashMap<Integer, HashMap<String, String>> datosEmpresa = new HashMap<Integer, HashMap<String, String>>();
		// Se asume que el fichero sólo tiene una empresa
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
		Iterator<String> itNombresParametros;
		while (itAntiguedades.hasNext()) {
			antiguedad = itAntiguedades.next();
			parametrosEmpresa = datosEmpresa.get(antiguedad);
			nombresParametros = parametrosEmpresa.keySet();
			itNombresParametros = nombresParametros.iterator();
			// Se añade empresa y antigüedad
			csvWriter.append(empresa + ";" + itAntiguedades.next());
			while (itNombresParametros.hasNext()) {
				// Se añaden el resto de parámetros
				csvWriter.append(";" + parametrosEmpresa.get(itNombresParametros.next()));
			}
			// Siguiente fila
			if (itAntiguedades.hasNext())
				csvWriter.append("\n");
		}
		csvWriter.flush();
		csvWriter.close();

	}
}
