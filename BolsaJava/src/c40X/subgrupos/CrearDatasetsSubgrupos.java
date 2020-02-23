package c40X.subgrupos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c10X.brutos.EstaticosFinvizDescargarYParsear;
import c30x.elaborados.construir.ElaboradosUtils;
import c30x.elaborados.construir.Estadisticas;
import c30x.elaborados.construir.GestorFicheros;

/**
 * Crea los datasets (CSV) de cada subgrupo
 *
 */
public class CrearDatasetsSubgrupos implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

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

	private final static Float PER_umbral1 = 5.0F;
	private final static Float PER_umbral2 = 25.0F;
	private final static Float PER_umbral3 = 50.0F;

	private final static Float DE_umbral1 = 1.0F;
	private final static Float DE_umbral2 = 2.5F;
	private final static Float DE_umbral3 = 5.0F;

	private final static Integer SMA50RATIOPRECIO_umbral1 = 80;
	private final static Integer SMA50RATIOPRECIO_umbral2 = 100;
	private final static Integer SMA50RATIOPRECIO_umbral3 = 120;

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

		// Tipos de empresa segun MARKET CAP (0-6)
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

		// Tipos de empresa segun SECTOR ECONOMICO (7-15)
		ArrayList<String> pathEmpresasTipo7 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo8 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo9 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo10 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo11 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo12 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo13 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo14 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo15 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo16 = new ArrayList<String>();

		// Tipos de empresa segun PER
		ArrayList<String> pathEmpresasTipo17 = new ArrayList<String>(); // PER bajo
		ArrayList<String> pathEmpresasTipo18 = new ArrayList<String>(); // PER medio
		ArrayList<String> pathEmpresasTipo19 = new ArrayList<String>(); // PER alto
		ArrayList<String> pathEmpresasTipo20 = new ArrayList<String>(); // PER muy alto
		ArrayList<String> pathEmpresasTipo21 = new ArrayList<String>(); // PER desconocido

		// Tipos de empresa segun DEUDA/ACTIVOS
		ArrayList<String> pathEmpresasTipo22 = new ArrayList<String>(); // bajo
		ArrayList<String> pathEmpresasTipo23 = new ArrayList<String>(); // medio
		ArrayList<String> pathEmpresasTipo24 = new ArrayList<String>(); // alto
		ArrayList<String> pathEmpresasTipo25 = new ArrayList<String>(); // muy alto
		ArrayList<String> pathEmpresasTipo26 = new ArrayList<String>(); // desconocido

		// Tipos de empresa segun ratio SMA50 de precio
		ArrayList<String> pathEmpresasTipo27 = new ArrayList<String>(); // bajo
		ArrayList<String> pathEmpresasTipo28 = new ArrayList<String>(); // medio
		ArrayList<String> pathEmpresasTipo29 = new ArrayList<String>(); // alto
		ArrayList<String> pathEmpresasTipo30 = new ArrayList<String>(); // muy alto
		ArrayList<String> pathEmpresasTipo31 = new ArrayList<String>(); // desconocido

		// Para cada EMPRESA
		while (iterator.hasNext()) {

			gestorFicheros = new GestorFicheros();
			datosEntrada = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
			ficheroGestionado = iterator.next();
			MY_LOGGER.debug("Fichero entrada: " + ficheroGestionado.getAbsolutePath());
			// Sólo leo la cabecera y la primera línea de datos, con antigüedad=0. Así
			// optimizo la lectura
			datosEntrada = gestorFicheros.leeTodosLosParametrosFicheroDeSoloUnaEmpresaYNFilasDeDatosRecientes(
					ficheroGestionado.getPath(), 1);

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

			// ------ SUBGRUPOS según MARKET CAP ------------
			String mcStr = parametros.get("Market Cap");

			if (mcStr != null && !mcStr.isEmpty() && !"-".equals(mcStr)) {

				Float marketCapValor = Float.valueOf(mcStr);

				// default, incluye a todos. LO quitamos, porque es inutil
				// pathEmpresasTipo0.add(ficheroGestionado.getAbsolutePath());

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
				MY_LOGGER.debug(ficheroGestionado.getAbsolutePath() + " -> Market Cap: " + mcStr);
			}

			// ------ SUBGRUPOS según SECTOR ------------
			String sectorStr = parametros.get("sector");

			if (sectorStr != null && !sectorStr.isEmpty() && !"-".equals(sectorStr)) {

				if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_BM)) {
					pathEmpresasTipo7.add(ficheroGestionado.getAbsolutePath());
				} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_CONG)) {
					pathEmpresasTipo8.add(ficheroGestionado.getAbsolutePath());
				} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_CONSGO)) {
					pathEmpresasTipo9.add(ficheroGestionado.getAbsolutePath());
				} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_FIN)) {
					pathEmpresasTipo10.add(ficheroGestionado.getAbsolutePath());
				} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_HC)) {
					pathEmpresasTipo11.add(ficheroGestionado.getAbsolutePath());
				} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_IG)) {
					pathEmpresasTipo12.add(ficheroGestionado.getAbsolutePath());
				} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_SERV)) {
					pathEmpresasTipo13.add(ficheroGestionado.getAbsolutePath());
				} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_TECH)) {
					pathEmpresasTipo14.add(ficheroGestionado.getAbsolutePath());
				} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_UTIL)) {
					pathEmpresasTipo15.add(ficheroGestionado.getAbsolutePath());
				} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_RE)) {
					pathEmpresasTipo16.add(ficheroGestionado.getAbsolutePath());
				} else {
					MY_LOGGER.warn(ficheroGestionado.getAbsolutePath() + " -> Sector raro: " + sectorStr);
				}

			} else {
				MY_LOGGER.warn(ficheroGestionado.getAbsolutePath() + " -> PER desconocido: " + sectorStr);

			}

			// ------ SUBGRUPOS según PER ------------
			String perStr = parametros.get("P/E");

			if (perStr != null && !perStr.isEmpty() && !"-".equals(perStr)) {
				Float per = Float.valueOf(perStr);

				if (per > 0 && per < PER_umbral1) {
					pathEmpresasTipo17.add(ficheroGestionado.getAbsolutePath());
				} else if (per >= PER_umbral1 && per < PER_umbral2) {
					pathEmpresasTipo18.add(ficheroGestionado.getAbsolutePath());
				} else if (per >= PER_umbral2 && per < PER_umbral3) {
					pathEmpresasTipo19.add(ficheroGestionado.getAbsolutePath());
				} else {
					pathEmpresasTipo20.add(ficheroGestionado.getAbsolutePath());
				}

			} else {
				pathEmpresasTipo21.add(ficheroGestionado.getAbsolutePath());
			}

			// ------ SUBGRUPOS según Debt/Eq ------------
			String debtEqStr = parametros.get("Debt/Eq");

			if (debtEqStr != null && !debtEqStr.isEmpty() && !"-".equals(debtEqStr)) {
				Float debtEq = Float.valueOf(debtEqStr);

				if (debtEq > 0 && debtEq < DE_umbral1) {
					pathEmpresasTipo22.add(ficheroGestionado.getAbsolutePath());
				} else if (debtEq >= DE_umbral1 && debtEq < DE_umbral2) {
					pathEmpresasTipo23.add(ficheroGestionado.getAbsolutePath());
				} else if (debtEq >= DE_umbral2 && debtEq < DE_umbral3) {
					pathEmpresasTipo24.add(ficheroGestionado.getAbsolutePath());
				} else {
					pathEmpresasTipo25.add(ficheroGestionado.getAbsolutePath());
					// MY_LOGGER.warn("Empresa = " + empresa + " con Debt/Eq = " + debtEqStr);
				}

			} else {
				pathEmpresasTipo26.add(ficheroGestionado.getAbsolutePath());
				// MY_LOGGER.warn("Empresa = " + empresa + " con Debt/Eq = " + debtEqStr);
			}

			// ------ SUBGRUPOS según ratio de SMA50 de precio ------------
			String ratioSMA50PrecioStr = parametros.get("RATIO_SMA_50_PRECIO");
			// System.out.println("--ratioSMA50PrecioStr---: " + ratioSMA50PrecioStr);

			if (!ratioSMA50PrecioStr.contains("null") && !ratioSMA50PrecioStr.isEmpty()
					&& !"-".equals(ratioSMA50PrecioStr)) {
				Integer ratioSMA50Precio = Integer.valueOf(ratioSMA50PrecioStr);

				if (ratioSMA50Precio > 0 && ratioSMA50Precio < SMA50RATIOPRECIO_umbral1) {
					pathEmpresasTipo27.add(ficheroGestionado.getAbsolutePath());
				} else if (ratioSMA50Precio >= SMA50RATIOPRECIO_umbral1
						&& ratioSMA50Precio < SMA50RATIOPRECIO_umbral2) {
					pathEmpresasTipo28.add(ficheroGestionado.getAbsolutePath());
				} else if (ratioSMA50Precio >= SMA50RATIOPRECIO_umbral2
						&& ratioSMA50Precio < SMA50RATIOPRECIO_umbral3) {
					pathEmpresasTipo29.add(ficheroGestionado.getAbsolutePath());
				} else {
					pathEmpresasTipo30.add(ficheroGestionado.getAbsolutePath());
					// MY_LOGGER.warn("Empresa = " + empresa + " con RATIO_SMA_50_PRECIO = " +
					// ratioSMA50PrecioStr);
				}

			} else {
				pathEmpresasTipo31.add(ficheroGestionado.getAbsolutePath());
				// MY_LOGGER.warn("Empresa = " + empresa + " con RATIO_SMA_50_PRECIO = " +
				// ratioSMA50PrecioStr);
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

		empresasPorTipo.put(7, pathEmpresasTipo7);
		empresasPorTipo.put(8, pathEmpresasTipo8);
		empresasPorTipo.put(9, pathEmpresasTipo9);
		empresasPorTipo.put(10, pathEmpresasTipo10);
		empresasPorTipo.put(11, pathEmpresasTipo11);
		empresasPorTipo.put(12, pathEmpresasTipo12);
		empresasPorTipo.put(13, pathEmpresasTipo13);
		empresasPorTipo.put(14, pathEmpresasTipo14);
		empresasPorTipo.put(15, pathEmpresasTipo15);
		empresasPorTipo.put(16, pathEmpresasTipo16);

		empresasPorTipo.put(17, pathEmpresasTipo17);
		empresasPorTipo.put(18, pathEmpresasTipo18);
		empresasPorTipo.put(19, pathEmpresasTipo19);
		empresasPorTipo.put(20, pathEmpresasTipo20);
		empresasPorTipo.put(21, pathEmpresasTipo21);

		empresasPorTipo.put(22, pathEmpresasTipo22);
		empresasPorTipo.put(23, pathEmpresasTipo23);
		empresasPorTipo.put(24, pathEmpresasTipo24);
		empresasPorTipo.put(25, pathEmpresasTipo25);
		empresasPorTipo.put(26, pathEmpresasTipo26);

		empresasPorTipo.put(27, pathEmpresasTipo27);
		empresasPorTipo.put(28, pathEmpresasTipo28);
		empresasPorTipo.put(29, pathEmpresasTipo29);
		empresasPorTipo.put(30, pathEmpresasTipo30);
		empresasPorTipo.put(31, pathEmpresasTipo31);

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
		String infoParaValidacionFutura;
		Double coberturaEmpresasPorCluster;
		Estadisticas estadisticas;
		String pathEmpresa;
		HashMap<String, Boolean> empresasConTarget;
		Iterator<String> itEmpresas;

		while (itTipos.hasNext()) {

			tipo = itTipos.next();
			MY_LOGGER.info("Subgrupo con ID=" + tipo);

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
						// Si la empresa tiene al menos una vela con target=1
						estadisticas.addValue(1);
					} else {
						estadisticas.addValue(0);
					}
					MY_LOGGER.debug("Empresa: " + pathEmpresa + " ¿tiene algún target=1? "
							+ empresasConTarget.get(pathEmpresa));
				}

				// Se calcula la cobertura del target
				coberturaEmpresasPorCluster = estadisticas.getMean();
				MY_LOGGER.debug(
						"COBERTURA DEL cluster " + tipo + ": " + Math.round(coberturaEmpresasPorCluster * 100) + "%");

				// Para generar un fichero de dataset del cluster, la cobertura debe ser mayor
				// que un x%
				if (coberturaEmpresasPorCluster * 100 < Double.valueOf(coberturaMinima)) {
					MY_LOGGER.warn("Cluster " + tipo + " tiene un " + Math.round(coberturaEmpresasPorCluster * 100)
							+ "% de empresas (" + estadisticas.getSum() + " de " + estadisticas.getValues().length
							+ ") con al menos una vela positiva (target=1). "
							+ "Por tanto no se llega al mínimo deseado (" + coberturaMinima
							+ "%). NO SE GENERA DATASET");

				} else if (empresasConTarget.keySet().size() < Integer.valueOf(minEmpresasPorCluster)) {
					MY_LOGGER.warn("Cluster " + tipo + ", tiene " + empresasConTarget.keySet().size()
							+ " empresas. Es demasiado pequeño, porque debería tener al menos " + minEmpresasPorCluster
							+ " empresas." + " NO SE GENERA DATASET");
				} else {

					MY_LOGGER.info("Cluster " + tipo + " tiene un " + Math.round(coberturaEmpresasPorCluster * 100)
							+ "% de empresas (" + estadisticas.getSum() + " de " + estadisticas.getValues().length
							+ ") con al menos una vela positiva (target=1)." + " => Supera el mínimo deseado ("
							+ coberturaMinima + "%). SI SE GENERA DATASET");
					MY_LOGGER.info("También cluster " + tipo + " tiene " + empresasConTarget.keySet().size()
							+ " empresas. Es suficientemente grande porque supera el umbral de tener al menos "
							+ minEmpresasPorCluster + " empresas." + " SI SE GENERA DATASET");

					// Hay alguna empresa de este tipo. Creo un CSV común para todas las del mismo
					// tipo
					pathFicheros = empresasPorTipo.get(tipo);

					String dirSubgrupoOut = directorioOut + "SG_" + tipo + "/";
					MY_LOGGER.info("Creando la carpeta del subgrupo con ID=" + tipo + " en: " + dirSubgrupoOut);
					File dirSubgrupoOutFile = new File(dirSubgrupoOut);
					dirSubgrupoOutFile.mkdir();

					ficheroOut = dirSubgrupoOut + "COMPLETO.csv";
					ficheroListadoOut = dirSubgrupoOut + "EMPRESAS.txt";
					MY_LOGGER.info("CSV de subgrupo: " + ficheroOut);
					MY_LOGGER.info("Lista de empresas de subgrupo: " + ficheroListadoOut);
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
								// Se eliminan los parámetros estáticos de la fila, EXCEPTO los datos para la
								// validación futura
								// Para cada fila de datos o de cabecera, de longitud variable, se eliminan los
								// datos estáticos
								// También se añaden precios y volumen, para la validación económica en c7
								Integer indice = SubgruposUtils.indiceDeAparicion(characterPipe, 13, row);
								infoParaValidacionFutura = row.substring(0, indice);
								rowTratada = infoParaValidacionFutura + characterPipe + SubgruposUtils
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
