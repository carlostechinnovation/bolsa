package c40X.subgrupos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
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
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import c30x.elaborados.construir.ElaboradosUtils;
import c30x.elaborados.construir.GestorFicheros;

public class CrearDatasetsSubgruposKMeans {

	static Logger MY_LOGGER = Logger.getLogger(CrearDatasetsSubgrupos.class);

	
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

		HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada = new HashMap<Integer, HashMap<String, String>>();

		List<EmpresaWrapper> clusterInput = new ArrayList<EmpresaWrapper>(ficherosEntradaEmpresas.size());
		EmpresaWrapper empresaWrapper;
		while (iterator.hasNext()) {

			gestorFicheros = new GestorFicheros();
			datosEntrada = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
			ficheroGestionado = iterator.next();
			MY_LOGGER.info("Fichero entrada: " + ficheroGestionado.getAbsolutePath());
			// Sólo leo la cabecera y la primera línea de datos, con antigüedad=0. Así
			// optimizo la lectura
			datosEntrada = gestorFicheros
					.leeTodosLosParametrosFicheroDeSoloUnaEmpresaYFilaMasReciente(ficheroGestionado.getPath());

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

		// initialize a new clustering algorithm.
		// we use KMeans++ with 6 clusters and 10000 iterations maximum.
		// we did not specify a distance measure; the default (euclidean distance) is
		// used.
		KMeansPlusPlusClusterer<EmpresaWrapper> clusterer = new KMeansPlusPlusClusterer<EmpresaWrapper>(6, 10000,
				new EuclideanDistance(), new JDKRandomGenerator(), EmptyClusterStrategy.LARGEST_VARIANCE);
		List<CentroidCluster<EmpresaWrapper>> clusterResults = clusterer.cluster(clusterInput);

		// output the clusters
		for (int i = 0; i < clusterResults.size(); i++) {
			MY_LOGGER.debug("Cluster " + i);
			System.out.println("------------------------------Cluster " + i);
			for (EmpresaWrapper empresaLeida : clusterResults.get(i).getPoints()) {
				MY_LOGGER.debug("Empresa " + empresaLeida.getFichero().getAbsolutePath() + " con ValorClustering= "
						+ empresaLeida.getValorClustering());
				System.out.println("Empresa " + empresaLeida.getFichero().getAbsolutePath() + " con ValorClustering= "
						+ empresaLeida.getValorClustering());
			}
		}

		// Se crea un CSV para cada subgrupo
		CentroidCluster<EmpresaWrapper> centroid;
		EmpresaWrapper empresa;
		String row, rowTratada;
		Boolean esPrimeraLinea;

		Integer numParametrosEstaticos = gestorFicheros.getOrdenNombresParametrosLeidos().size();
		String pipe = "|";
		Character characterPipe = pipe.charAt(0);
		String ficheroOut, ficheroListadoOut;
		FileWriter csvWriter;
		FileWriter writerListadoEmpresas;
		for (int i = 0; i < clusterResults.size(); i++) {

			centroid = clusterResults.get(i);
			// Creo un CSV común para todas las del mismo tipo
			ficheroOut = directorioOut + i + ".csv";
			ficheroListadoOut = directorioOut + "Listado-" + i + ".empresas";
			MY_LOGGER.info("Fichero a escribir: " + ficheroOut);
			csvWriter = new FileWriter(ficheroOut);
			writerListadoEmpresas = new FileWriter(ficheroListadoOut);
			List<EmpresaWrapper> empresas = centroid.getPoints();

			for (int z = 0; z < empresas.size(); z++) {

				esPrimeraLinea = Boolean.TRUE;
				// Se lee el fichero de la empresa a meter en el CSV común
				empresa = empresas.get(z);
				MY_LOGGER
						.debug("Fichero a leer para clasificar en subgrupo: " + empresa.getFichero().getAbsolutePath());
				BufferedReader csvReader = new BufferedReader(new FileReader(empresa.getFichero().getAbsolutePath()));

				// Añado la empresa al fichero de listado de empresas
				writerListadoEmpresas.append(empresa.getFichero().getAbsolutePath() + "\n");

				try {

					while ((row = csvReader.readLine()) != null) {
						MY_LOGGER.debug("Fila leída: " + row);
						// Se eliminan los parámetros estáticos de la fila
						// Para cada fila de datos o de cabecera, de longitud variable, se eliminan los
						// datos estáticos
						rowTratada = SubgruposUtils.recortaPrimeraParteDeString(characterPipe, numParametrosEstaticos,
								row);
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

		// Se normaliza los datasets de cada subgrupo

	}

}