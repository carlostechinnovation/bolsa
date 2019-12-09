package c40X.subgrupos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import c30x.elaborados.construir.ElaboradosUtils;
import c30x.elaborados.construir.GestorFicheros;

/**
 * Crea los datasets (CSV) de cada subgrupo
 *
 */
public class CrearDatasetsSubgrupos {

	static Logger MY_LOGGER = Logger.getLogger(CrearDatasetsSubgrupos.class);

	private final static Integer marketCap_large_max = 199999;
	private final static Integer marketCap_mid_max = 9999;
	private final static Integer marketCap_small_max = 1999;
	private final static Integer marketCap_micro_max = 299;
	private final static Integer marketCap_nano_max = 49;

	private static HashMap<Integer, ArrayList<String>> empresasPorTipo;

	public CrearDatasetsSubgrupos() {
		super();
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws Exception {

		MY_LOGGER.info("INICIO");
		BasicConfigurator.configure();
		MY_LOGGER.setLevel(Level.INFO);

		String directorioIn = ElaboradosUtils.DIR_ELABORADOS; // DEFAULT
		String directorioOut = SubgruposUtils.DIR_SUBGRUPOS; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 2) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			directorioIn = args[0];
			directorioOut = args[1];
		}

		crearSubgruposYNormalizar(directorioIn, directorioOut);

		MY_LOGGER.info("FIN");
	}

	/**
	 * Crea un CSV para cada subgrupo
	 * 
	 * @param directorioIn
	 * @param directorioOut
	 * @throws Exception
	 */
	public static void crearSubgruposYNormalizar(String directorioIn, String directorioOut) throws Exception {
		// Debo leer el parámetro que me interese: de momento el market cap. En el
		// futuro sería conveniente separar por sector y liquidez (volumen medio de 6
		// meses en dólares).
		GestorFicheros gestorFicheros = new GestorFicheros();
		File directorioEntrada = new File(directorioIn);
		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosEntrada;
		ArrayList<File> ficherosEntradaEmpresas = gestorFicheros.listaFicherosDeDirectorio(directorioEntrada);
		Iterator<File> iterator = ficherosEntradaEmpresas.iterator();
		File ficheroGestionado;
		HashMap<String, String> parametros;
		HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada = new HashMap<Integer, HashMap<String, String>>();
		// Tipos de empresa
		// Tipo 1: MARKETCAP=MEGA
		// Tipo 2: MARKETCAP=LARGA
		// Tipo 3: MARKETCAP=MID
		// Tipo 4: MARKETCAP=SMALL
		// Tipo 5: MARKETCAP=MICRO
		// Tipo 6: MARKETCAP=NANO
		ArrayList<String> pathEmpresasTipo1 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo2 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo3 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo4 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo5 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo6 = new ArrayList<String>();
		while (iterator.hasNext()) {
			gestorFicheros = new GestorFicheros();
			datosEntrada = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
			ficheroGestionado = iterator.next();
			MY_LOGGER.debug("Fichero entrada: " + ficheroGestionado.getAbsolutePath());
			// Sólo leo la cabecera y la primera línea de datos, con antigüedad=0. Así
			// optimizo la lectura
			datosEntrada = gestorFicheros
					.leeSoloParametrosNoElaboradosFicheroDeSoloUnaEmpresa(ficheroGestionado.getPath(), Boolean.TRUE);
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
			parametros = datosEmpresaEntrada.get(0);
			Integer marketCapValor = Integer.valueOf(parametros.get("Market Cap"));

			// CLASIFICACIÓN DEL TIPO DE EMPRESA
			if (marketCapValor < marketCap_nano_max)
				pathEmpresasTipo6.add(ficheroGestionado.getAbsolutePath());
			else if (marketCapValor < marketCap_micro_max)
				pathEmpresasTipo5.add(ficheroGestionado.getAbsolutePath());
			else if (marketCapValor < marketCap_small_max)
				pathEmpresasTipo4.add(ficheroGestionado.getAbsolutePath());
			else if (marketCapValor < marketCap_mid_max)
				pathEmpresasTipo3.add(ficheroGestionado.getAbsolutePath());
			else if (marketCapValor < marketCap_large_max)
				pathEmpresasTipo2.add(ficheroGestionado.getAbsolutePath());
			else
				pathEmpresasTipo1.add(ficheroGestionado.getAbsolutePath());

		}
		// Almacenamiento del tipo de empresa en la lista
		empresasPorTipo = new HashMap<Integer, ArrayList<String>>();
		empresasPorTipo.put(1, pathEmpresasTipo1);
		empresasPorTipo.put(2, pathEmpresasTipo2);
		empresasPorTipo.put(3, pathEmpresasTipo3);
		empresasPorTipo.put(4, pathEmpresasTipo4);
		empresasPorTipo.put(5, pathEmpresasTipo5);
		empresasPorTipo.put(6, pathEmpresasTipo6);

		// Se crea un CSV para cada subgrupo
		Set<Integer> tipos = empresasPorTipo.keySet();
		Iterator<Integer> itTipos = tipos.iterator();
		Integer numEmpresasPorTipo;
		Integer tipo;
		String pathFichero;
		String row;
		Boolean esPrimeraLinea;
		while (itTipos.hasNext()) {
			tipo = itTipos.next();
			numEmpresasPorTipo = empresasPorTipo.size();
			if (numEmpresasPorTipo > 0) {
				// Hay alguna empresa de este tipo. Creo un CSV común para todas las del mismo
				// tipo
				ArrayList<String> pathFicheros = empresasPorTipo.get(tipo);
				FileWriter csvWriter = new FileWriter(directorioOut + tipo + ".csv");
				for (int i = 0; i < pathFicheros.size(); i++) {
					esPrimeraLinea = Boolean.TRUE;
					// Se lee el fichero de la empresa a meter en el CSV común
					pathFichero = pathFicheros.get(i);
					MY_LOGGER.debug("Fichero a leer para clasificar en subgrupo: " + pathFichero);
					BufferedReader csvReader = new BufferedReader(new FileReader(pathFichero));
					try {
						while ((row = csvReader.readLine()) != null) {
							MY_LOGGER.debug("Fila le�da: " + row);
							System.out.println("Fila le�da: " + row);
							// La cabecera se toma de la primera línea del primer fichero
							if (i == 0 && esPrimeraLinea) {
								// En la primera l�nea est� la cabecera de par�metros
								// Se valida que el nombre recibido es igual que el usado en la constructora, y
								// en dicho orden
								csvWriter.append(row + "\n");
							}
							if (!esPrimeraLinea) {
								// Para todos los ficheros que no sean el primero, se toman sólo los datos, sin
								// la cabecera
								csvWriter.append(row + "\n");
							}
							// Para las siguientes filas del fichero
							esPrimeraLinea = Boolean.FALSE;
						}
					} finally {
						csvReader.close();
					}

				}
				csvWriter.flush();
				csvWriter.close();
			}
		}

		// Se normaliza los datasets de cada subgrupo

	}

}
