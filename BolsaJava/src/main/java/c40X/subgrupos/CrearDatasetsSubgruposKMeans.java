package c40X.subgrupos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer.EmptyClusterStrategy;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c30x.elaborados.construir.ElaboradosUtils;
import c30x.elaborados.construir.Estadisticas;
import c30x.elaborados.construir.GestorFicheros;
import coordinador.Principal;

/**
 * Crear los datasets de SUBGRUPOS usando KMeans (Clustering, no supervisado).
 */
public class CrearDatasetsSubgruposKMeans implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	static Logger MY_LOGGER = Logger.getLogger(CrearDatasetsSubgruposKMeans.class);

	private static CrearDatasetsSubgruposKMeans instancia = null;

	private CrearDatasetsSubgruposKMeans() {
		super();
	}

	public static CrearDatasetsSubgruposKMeans getInstance() {
		if (instancia == null)
			instancia = new CrearDatasetsSubgruposKMeans();

		return instancia;
	}

	public static void main(String[] args) throws Exception {

		Object appendersAcumulados = Logger.getRootLogger().getAllAppenders();
		if (appendersAcumulados instanceof NullEnumeration) {
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(Principal.LOG_PATRON)));
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
	 * @param coberturaMinima       PORCENTAJE. En cada dataset, debe haber un
	 *                              minimo % de empresas con alguna vela de
	 *                              target=true. Si no, no genero el dataset.
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

		HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada = new HashMap<Integer, HashMap<String, String>>();

		List<EmpresaWrapper> clusterInput = new ArrayList<EmpresaWrapper>(ficherosEntradaEmpresas.size());
		EmpresaWrapper empresaWrapper;
		while (iterator.hasNext()) {

			gestorFicheros = new GestorFicheros();
			datosEntrada = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
			ficheroGestionado = iterator.next();
			MY_LOGGER.debug("Fichero entrada: " + ficheroGestionado.getAbsolutePath());
			// Sólo leo la cabecera y las dos primeras líneas de datos, con antigüedad=0.
			// Así optimizo la lectura
			datosEntrada = gestorFicheros.leeTodosLosParametrosFicheroDeSoloUnaEmpresaYNFilasDeDatosRecientes(
					ficheroGestionado.getPath(), 2);

			String empresa = "";
			Set<String> empresas = datosEntrada.keySet();
			Iterator<String> itEmpresas = datosEntrada.keySet().iterator();
			if (empresas.size() != 1) {
				throw new Exception("Se están calculando parámetros elaborados de más de una empresa");
			} else {
				while (itEmpresas.hasNext())
					empresa = itEmpresas.next();
			}

			datosEmpresaEntrada = datosEntrada.get(empresa);
			Empresa empresaDatos = new Empresa(ficheroGestionado, datosEmpresaEntrada);
			try {
				empresaWrapper = new EmpresaWrapper(empresaDatos);
				clusterInput.add(empresaWrapper);
			} catch (NumberFormatException e) {
				// Al crear el wrapper, se leen los valores para calcular la función distancia.
				// Es posible que los valores leídos no estén definidos en los datos.
				// En ese caso, no se incluirá esta empresa en el cluster
			}

		}

		// Clustering: algoritmo KMeans++
		KMeansPlusPlusClusterer<EmpresaWrapper> clusterer = new KMeansPlusPlusClusterer<EmpresaWrapper>(6, 10000,
				new EuclideanDistance(), new JDKRandomGenerator(), EmptyClusterStrategy.LARGEST_VARIANCE);
		List<CentroidCluster<EmpresaWrapper>> clusterResults = clusterer.cluster(clusterInput);

		// output the clusters
		for (int i = 0; i < clusterResults.size(); i++) {
			MY_LOGGER.debug("Cluster " + i);
			MY_LOGGER.debug("------------------------------Cluster " + i);
			for (EmpresaWrapper empresaLeida : clusterResults.get(i).getPoints()) {
				MY_LOGGER.debug("Empresa " + empresaLeida.getFichero().getAbsolutePath() + " con ValorClustering= "
						+ empresaLeida.getValorClustering());
				MY_LOGGER.debug("Empresa " + empresaLeida.getFichero().getAbsolutePath() + " con ValorClustering= "
						+ empresaLeida.getValorClustering());
			}
		}

		// Se crea un CSV para cada subgrupo
		EmpresaWrapper empresa;
		String row, rowTratada;
		Boolean esPrimeraLinea;

		Integer numParametrosEstaticos = gestorFicheros.getOrdenNombresParametrosLeidos().size();
		String pipe = "|";
		Character characterPipe = pipe.charAt(0);
		String ficheroOut, ficheroListadoOut;
		FileWriter csvWriter;
		FileWriter writerListadoEmpresas;
		Double coberturaEmpresasPorCluster;
		Estadisticas estadisticas;
		String pathEmpresa;
		HashMap<String, Boolean> empresasConTarget;
		Iterator<String> itEmpresas;
		for (int i = 0; i < clusterResults.size(); i++) {

			// Antes se comprobará, en cada cluster, qué porcentaje hay de empresas con al
			// menos una vela con target=1,
			// respecto del total de empresas del cluster (esto se llama Cobertura).
			// Sólo se guardarán los clusters con cobertura mayor que una cantidad mínima.
			List<EmpresaWrapper> empresas = clusterResults.get(i).getPoints();
			ArrayList<String> pathFicherosEmpresas = new ArrayList<String>();
			for (EmpresaWrapper wrapper : empresas) {
				pathFicherosEmpresas.add(wrapper.getFichero().getAbsolutePath());

			}
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
				MY_LOGGER.debug(
						"Empresa: " + pathEmpresa + " ¿tiene algún target=1? " + empresasConTarget.get(pathEmpresa));
			}
			// Se calcula la cobertura del target
			coberturaEmpresasPorCluster = estadisticas.getMean();
			MY_LOGGER.debug("COBERTURA DEL cluster " + i + ": " + coberturaEmpresasPorCluster * 100 + "%");
			System.out.println("COBERTURA DEL cluster " + i + ": " + coberturaEmpresasPorCluster * 100 + "%");

			// Para generar un fichero de dataset del cluster, la cobertura debe ser mayor
			// que un x%
			if (coberturaEmpresasPorCluster * 100 < Double.valueOf(coberturaMinima)) {
				MY_LOGGER.debug("El cluster " + i + ", con cobertura: " + coberturaEmpresasPorCluster * 100 + "%"
						+ " no llega al mínimo: " + coberturaMinima + "%. NO SE GENERA DATASET");

			} else if (empresasConTarget.keySet().size() < Integer.valueOf(minEmpresasPorCluster)) {
				MY_LOGGER.debug("El cluster " + i + ", tiene: " + empresasConTarget.keySet().size()
						+ " empresas, pero el mínimo debe ser: " + minEmpresasPorCluster + ". NO SE GENERA DATASET");

			} else {
				// Creo un CSV común para todas las del mismo tipo
				ficheroOut = directorioOut + i + ".csv";
				ficheroListadoOut = directorioOut + "Listado-" + i + ".empresas";
				MY_LOGGER.info("Fichero a escribir: " + ficheroOut);
				csvWriter = new FileWriter(ficheroOut);
				writerListadoEmpresas = new FileWriter(ficheroListadoOut);

				for (int z = 0; z < empresas.size(); z++) {

					esPrimeraLinea = Boolean.TRUE;
					// Se lee el fichero de la empresa a meter en el CSV común
					empresa = empresas.get(z);
					MY_LOGGER.debug(
							"Fichero a leer para clasificar en subgrupo: " + empresa.getFichero().getAbsolutePath());
					BufferedReader csvReader = new BufferedReader(
							new FileReader(empresa.getFichero().getAbsolutePath()));

					// Añado la empresa al fichero de listado de empresas
					writerListadoEmpresas.append(empresa.getFichero().getAbsolutePath() + "\n");

					try {

						while ((row = csvReader.readLine()) != null) {
							MY_LOGGER.debug("Fila leída: " + row);
							// Se eliminan los parámetros estáticos de la fila
							// Para cada fila de datos o de cabecera, de longitud variable, se eliminan los
							// datos estáticos
							rowTratada = SubgruposUtils.recortaPrimeraParteDeString(characterPipe,
									numParametrosEstaticos, row);
							MY_LOGGER.debug("Fila escrita: " + rowTratada);

							// La cabecera se toma de la primera línea del primer fichero
							if (z == 0 && esPrimeraLinea) {
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

}
