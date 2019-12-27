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

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c30x.elaborados.construir.ElaboradosUtils;
import c30x.elaborados.construir.Estadisticas;
import c30x.elaborados.construir.GestorFicheros;

/**
 * Crea los datasets (CSV) de cada subgrupo
 *
 */
public class CrearDatasetsSubgrupos {

	static Logger MY_LOGGER = Logger.getLogger(CrearDatasetsSubgrupos.class);

	private static CrearDatasetsSubgrupos instancia = null;

	private CrearDatasetsSubgrupos() {
		super();
	}

	public static CrearDatasetsSubgrupos getInstance() {
		if (instancia == null)
			instancia = new CrearDatasetsSubgrupos();

		return instancia;
	}

	private final static Integer marketCap_large_max = 199999;
	private final static Integer marketCap_mid_max = 9999;
	private final static Integer marketCap_small_max = 1999;
	private final static Integer marketCap_micro_max = 299;
	private final static Integer marketCap_nano_max = 49;

	private static HashMap<Integer, ArrayList<String>> empresasPorTipo;

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws Exception {

		Object appendersAcumulados = Logger.getRootLogger().getAllAppenders();
		if (appendersAcumulados instanceof NullEnumeration) {
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(PatternLayout.TTCC_CONVERSION_PATTERN)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

		String directorioIn = ElaboradosUtils.DIR_ELABORADOS; // DEFAULT
		String directorioOut = SubgruposUtils.DIR_SUBGRUPOS; // DEFAULT
		String coberturaMinima = SubgruposUtils.MIN_COBERTURA_CLUSTER; // DEFAULT
		String minEmpresasPorCluster = SubgruposUtils.MIN_EMPRESAS_POR_CLUSTER; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 4) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			directorioIn = args[0];
			directorioOut = args[1];
			coberturaMinima = args[2];
			minEmpresasPorCluster = args[3];
		}

		crearSubgruposYNormalizar(directorioIn, directorioOut, coberturaMinima, minEmpresasPorCluster);

		MY_LOGGER.info("FIN");
	}

	/**
	 * Crea un CSV para cada subgrupo
	 * 
	 * @param directorioIn
	 * @param directorioOut
	 * @param coberturaMinima
	 * @param minEmpresasPorCluster
	 * @throws Exception
	 */
	public static void crearSubgruposYNormalizar(String directorioIn, String directorioOut, String coberturaMinima,
			String minEmpresasPorCluster) throws Exception {

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
		String antiguedad;
		Double coberturaEmpresasPorCluster;
		Estadisticas estadisticas;
		String pathEmpresa;
		HashMap<String, Boolean> empresasConTarget;
		Iterator<String> itEmpresas;
		while (itTipos.hasNext()) {

			tipo = itTipos.next();
			numEmpresasPorTipo = empresasPorTipo.size();

			if (numEmpresasPorTipo > 0) {

				// Antes se comprobará, en cada cluster, qué porcentaje hay de empresas con al
				// menos una vela con target=1,
				// respecto del total de empresas del cluster (esto se llama Cobertura).
				// Sólo se guardarán los clusters con cobertura mayor que una cantidad mínima.

				ArrayList<String> pathFicherosEmpresas = empresasPorTipo.get(tipo);

				empresasConTarget = gestorFicheros.compruebaEmpresasConTarget(pathFicherosEmpresas);
				itEmpresas = empresasConTarget.keySet().iterator();
				estadisticas = new Estadisticas();

				while (itEmpresas.hasNext()) {
					pathEmpresa = itEmpresas.next();
					if (empresasConTarget.get(pathEmpresa)) {
						estadisticas.addValue(1);
					} else {
						estadisticas.addValue(0);
					}
					MY_LOGGER.debug("Empresa: " + pathEmpresa + " ¿tiene algún target=1? "
							+ empresasConTarget.get(pathEmpresa));
					System.out.println("Empresa: " + pathEmpresa + " ¿tiene algún target=1? "
							+ empresasConTarget.get(pathEmpresa));
				}

				// Se calcula la cobertura del target
				coberturaEmpresasPorCluster = estadisticas.getMean();
				MY_LOGGER.debug("COBERTURA DEL cluster " + tipo + ": " + coberturaEmpresasPorCluster * 100 + "%");
				System.out.println("COBERTURA DEL cluster " + tipo + ": " + coberturaEmpresasPorCluster * 100 + "%");

				// Para generar un fichero de dataset del cluster, la cobertura debe ser mayor
				// que un x%
				if (coberturaEmpresasPorCluster * 100 < Double.valueOf(coberturaMinima)) {
					MY_LOGGER.debug("El cluster " + tipo + ", con cobertura: " + coberturaEmpresasPorCluster * 100 + "%"
							+ " no llega al mínimo: " + coberturaMinima + "%. NO SE GENERA DATASET");
					System.out.println("El cluster " + tipo + ", con cobertura: " + coberturaEmpresasPorCluster * 100
							+ "%" + " no llega al mínimo: " + coberturaMinima + "%. NO SE GENERA DATASET");
				} else if (empresasConTarget.keySet().size() < Integer.valueOf(minEmpresasPorCluster)) {
					MY_LOGGER.debug("El cluster " + tipo + ", tiene: " + empresasConTarget.keySet().size()
							+ " empresas, pero el mínimo debe ser: " + minEmpresasPorCluster
							+ ". NO SE GENERA DATASET");
					System.out.println("El cluster " + tipo + ", tiene: " + empresasConTarget.keySet().size()
							+ " empresas, pero el mínimo debe ser: " + minEmpresasPorCluster
							+ ". NO SE GENERA DATASET");
				} else {

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
								// Se eliminan los parámetros estáticos de la fila, EXCEPTO LA ANTIGÜEDAD (que
								// será el segundo parámetro)
								// Para cada fila de datos o de cabecera, de longitud variable, se eliminan los
								// datos estáticos
								antiguedad = SubgruposUtils.recortaPrimeraParteDeString(characterPipe, 1, row);
								antiguedad = antiguedad.substring(0, antiguedad.indexOf(characterPipe));
								rowTratada = antiguedad + characterPipe + SubgruposUtils
										.recortaPrimeraParteDeString(characterPipe, numeroParametrosEstaticos, row);
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
		}

		// Se normaliza los datasets de cada subgrupo

	}

}
