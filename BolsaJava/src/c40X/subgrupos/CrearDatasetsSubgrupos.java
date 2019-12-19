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
		// Tipo 0: TODAS
		// Tipo 1: MARKETCAP=MEGA
		// Tipo 2: MARKETCAP=LARGA
		// Tipo 3: MARKETCAP=MID
		// Tipo 4: MARKETCAP=SMALL
		// Tipo 5: MARKETCAP=MICRO
		// Tipo 6: MARKETCAP=NANO
		ArrayList<String> pathEmpresasTipo0 = new ArrayList<String>();
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
			MY_LOGGER.info("Fichero entrada: " + ficheroGestionado.getAbsolutePath());
			// Sólo leo la cabecera y la primera línea de datos, con antigüedad=0. Así
			// optimizo la lectura
			datosEntrada = gestorFicheros
					.leeSoloParametrosNoElaboradosFicheroDeSoloUnaEmpresa(ficheroGestionado.getPath(), Boolean.TRUE);

			String empresa = "";
			Set<String> empresas = datosEntrada.keySet();
			Iterator<String> itEmpresas = datosEntrada.keySet().iterator();
			if (empresas.size() != 1) {
				throw new Exception("Se están calculando parámetros elaborados de más de una empresa");
			} else {
				while (itEmpresas.hasNext())
					empresa = itEmpresas.next();
			}

			// EXTRACCIÓN DE DATOS DE LA EMPRESA: sólo se usan los datos ESTATICOS, así que
			// basta coger la PRIMERA fila de datos
			datosEmpresaEntrada = datosEntrada.get(empresa);

			Set<Integer> a = datosEmpresaEntrada.keySet();
			Integer indicePrimeraFilaDeDatos = null;
			if (a.iterator().hasNext()) {
				indicePrimeraFilaDeDatos = a.iterator().next();
			}
			parametros = datosEmpresaEntrada.get(indicePrimeraFilaDeDatos); // PRIMERA FILA
			String mcStr = parametros.get("Market Cap");

			if (mcStr != null && !mcStr.isEmpty() && !"-".equals(mcStr)) {

				Float marketCapValor = Float.valueOf(mcStr);

				pathEmpresasTipo0.add(ficheroGestionado.getAbsolutePath()); // subgrupo default

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

			} else {
				MY_LOGGER.warn(ficheroGestionado.getAbsolutePath() + " -> Market Cap: " + mcStr);
			}
		}

		// Almacenamiento del tipo de empresa en la lista
		empresasPorTipo = new HashMap<Integer, ArrayList<String>>();
		empresasPorTipo.put(0, pathEmpresasTipo0);
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
		String row, rowTratada;
		Boolean esPrimeraLinea;

		// En el Gestor de Ficheros aparecen los nombres de los parámetros estáticos a
		// eliminar. Sólo se cuentan. Habrá tantos pipes como parámetros
		Integer numeroParametrosEstaticos = gestorFicheros.getOrdenNombresParametrosLeidos().size();
		String pipe = "|";
		Character characterPipe = pipe.charAt(0);
		String ficheroOut, ficheroListadoOut;
		ArrayList<String> pathFicheros;
		FileWriter csvWriter;
		FileWriter writerListadoEmpresas;
		while (itTipos.hasNext()) {

			tipo = itTipos.next();
			numEmpresasPorTipo = empresasPorTipo.size();

			if (numEmpresasPorTipo > 0) {
				// Hay alguna empresa de este tipo. Creo un CSV común para todas las del mismo
				// tipo
				pathFicheros = empresasPorTipo.get(tipo);
				ficheroOut = directorioOut + tipo + ".csv";
				ficheroListadoOut = directorioOut + "Listado-" + tipo + ".empresas";
				MY_LOGGER.info("Fichero a escribir: " + ficheroOut);
				csvWriter = new FileWriter(ficheroOut);
				writerListadoEmpresas = new FileWriter(ficheroListadoOut);

				for (int i = 0; i < pathFicheros.size(); i++) {

					esPrimeraLinea = Boolean.TRUE;
					// Se lee el fichero de la empresa a meter en el CSV común
					pathFichero = pathFicheros.get(i);
					MY_LOGGER.debug("Fichero a leer para clasificar en subgrupo: " + pathFichero);
					BufferedReader csvReader = new BufferedReader(new FileReader(pathFichero));

					// Añado la empresa al fichero de listado de empresas
					writerListadoEmpresas.append(pathFichero + "\n");

					try {

						while ((row = csvReader.readLine()) != null) {
							MY_LOGGER.debug("Fila leída: " + row);
							// Se eliminan los parámetros estáticos de la fila
							// Para cada fila de datos o de cabecera, de longitud variable, se eliminan los
							// datos estáticos
							rowTratada = SubgruposUtils.recortaPrimeraParteDeString(characterPipe,
									numeroParametrosEstaticos, row);
							MY_LOGGER.debug("Fila escrita: " + rowTratada);

							// La cabecera se toma de la primera línea del primer fichero
							if (i == 0 && esPrimeraLinea) {
								// En la primera línea está la cabecera de parámetros
								// Se valida que el nombre recibido es igual que el usado en la constructora, y
								// en dicho orden
								csvWriter.append(rowTratada);

							}
							if (!esPrimeraLinea) {
								// Para todos los ficheros, se escriben las filas 2 y siguientes
								csvWriter.append("\n" + rowTratada);
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
				writerListadoEmpresas.flush();
				writerListadoEmpresas.close();
			}
		}

		// Se normaliza los datasets de cada subgrupo

	}

}
