package c40X.subgrupos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c10X.brutos.BrutosUtils;
import c10X.brutos.EstaticosFinvizDescargarYParsear;
import c30x.elaborados.construir.ElaboradosUtils;
import c30x.elaborados.construir.Estadisticas;
import c30x.elaborados.construir.GestorFicheros;
import coordinador.Principal;

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

	private final static Float PER_umbral1 = 10.0F;
	private final static Float PER_umbral2 = 30.0F;
	private final static Float PER_umbral3 = 50.0F;
	// Ratio PER (price-earnings ratio): si lo conocemos y es muy alto, la empresa
	// está demasiado sobrevalorada. Si no lo conocemos, seguimos, pero al menos lo
	// hemos intentado filtrar
	private final static Float MAX_PER = 70.0F;

	private final static Float DE_umbral1 = 0.5F;
	private final static Float DE_umbral2 = 1.2F;
	private final static Float DE_umbral3 = 1.7F;
	// DEUDA MAXIMA PERMITIDA (tanto por uno)
	private final static Float MAX_DEUDA_PERMITIDA = 2.0F;

	private final static Float SMA50RATIOPRECIO_umbral1 = 0.40F * 1000000F;
	private final static Float SMA50RATIOPRECIO_umbral2 = 0.60F * 1000000F;
	private final static Float SMA50RATIOPRECIO_umbral3 = 0.80F * 1000000F;

	private final static Float IO_umbral1 = 20.0F;
	private final static Float IO_umbral2 = 60.0F;

	private final static Float factorPicoVolumenCP = 2.8F;// Pico en volumen
	private final static Float factorPicoVolumenMP = 4F;// Pico en volumen

	// Las empresas MUY PEQUEÑAS (dato posiblemente mal actualizado en Finviz) o con
	// empleados DESCONOCIDOS no son fiables para usar DINERO REAL --> Son OPACAS y
	// MUY INESTABLES (ruido para el modelo) ==> DESCARTADAS
	private final static Integer MINIMO_NUMERO_EMPLEADOS_INFORMADO = 100;

	// MINIMO QUICK RATIO
	private final static Float MIN_QUICK_RATIO = 0.75F;

	// RECOMENDACIONES DE ANALISTAS (1= Comprar ... 5=Vender)
	private final static Float MAX_RECOM_ANALISTAS = 4.1F;

	private static HashMap<Integer, ArrayList<String>> empresasPorTipo;

	/**
	 * @param args
	 * @throws IOException
	 */
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
		String modoTiempo = BrutosUtils.FUTURO; // DEFAULT
		Integer filtroDinamico1 = ElaboradosUtils.DINAMICA1; // DEFAULT
		Integer filtroDinamico2 = ElaboradosUtils.DINAMICA2; // DEFAULT
		String realimentacion = SubgruposUtils.REALIMENTACION; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");

		} else if (args.length != 8) {
			MY_LOGGER.error("Total Parametros de entrada: " + args.length);
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			for (String param : args) {
				MY_LOGGER.info("Param: " + param);
			}
			System.exit(-1);

		} else {
			directorioIn = args[0];
			directorioOut = args[1];
			coberturaMinima = args[2];
			minEmpresasPorCluster = args[3];
			modoTiempo = args[4];
			filtroDinamico1 = Integer.valueOf(args[5]);
			filtroDinamico2 = Integer.valueOf(args[6]);
			realimentacion = args[7];
		}

		MY_LOGGER.info("directorioIn: " + directorioIn);
		MY_LOGGER.info("directorioOut: " + directorioOut);
		MY_LOGGER.info("coberturaMinima: " + coberturaMinima);
		MY_LOGGER.info("minEmpresasPorCluster: " + minEmpresasPorCluster);
		MY_LOGGER.info("modoTiempo: " + modoTiempo);
		MY_LOGGER.info("filtroDinamico1: " + filtroDinamico1);
		MY_LOGGER.info("filtroDinamico2: " + filtroDinamico2);
		MY_LOGGER.info("realimentacion: " + realimentacion);

		crearSubgruposYNormalizar(directorioIn, directorioOut, coberturaMinima, minEmpresasPorCluster, modoTiempo,
				filtroDinamico1, filtroDinamico2, realimentacion);

		MY_LOGGER.info("FIN");
	}

	/**
	 * Crea un CSV para cada subgrupo
	 * 
	 * @param directorioIn
	 * @param directorioOut
	 * @param coberturaMinima
	 * @param minEmpresasPorCluster
	 * @param modoTiempo
	 * @throws Exception
	 */
	public static void crearSubgruposYNormalizar(String directorioIn, String directorioOut, String coberturaMinima,
			String minEmpresasPorCluster, String modoTiempo, Integer filtroDinamico1, Integer filtroDinamico2,
			String realimentacion) throws Exception {

		// Debo leer el parámetro que me interese: de momento el market cap. En el
		// futuro sería conveniente separar por sector y liquidez (volumen medio de 6
		// meses en dólares).
		GestorFicheros gestorFicheros = new GestorFicheros();
//		MY_LOGGER.info(">>>>directorioIn: "+directorioIn);
		File directorioEntrada = new File(directorioIn);
		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosEntrada;
		ArrayList<File> ficherosEntradaEmpresas = gestorFicheros.listaFicherosDeDirectorio(directorioEntrada);
		Iterator<File> iterator = ficherosEntradaEmpresas.iterator();
		File ficheroGestionado;
		HashMap<String, String> parametros;

		// Lista manual de empresas seleccionadas ("alcistas")
		List<String> listaSeleccionManual = leerListaManualEmpresasSeleccionadas();

		// Lista de empresas de subgrupos calculados con algortimo de CLUSTERING
		// (descartamos subgrupo 0 que es gigante y poco refinado)
		List<String> listaSeleccionClustering1 = leerListaClusteringAlternativo("1");
		List<String> listaSeleccionClustering2 = leerListaClusteringAlternativo("2");
		List<String> listaSeleccionClustering3 = leerListaClusteringAlternativo("3");
		List<String> listaSeleccionClustering4 = leerListaClusteringAlternativo("4");
		List<String> listaSeleccionClustering5 = leerListaClusteringAlternativo("5");
		List<String> listaSeleccionClustering6 = leerListaClusteringAlternativo("6");
		List<String> listaSeleccionClustering7 = leerListaClusteringAlternativo("7");
		List<String> listaSeleccionClustering8 = leerListaClusteringAlternativo("8");
		List<String> listaSeleccionClustering9 = leerListaClusteringAlternativo("9");
		List<String> listaSeleccionClustering10 = leerListaClusteringAlternativo("0");

		HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada = new HashMap<Integer, HashMap<String, String>>();

		// Para el subgrupo 0 siempre se añade
		// NO QUITAR, PARA QUE LAS GRÁFICAS FINALES SE PINTEN BASADAS EN ESTE GRUPO,
		// QUE CONTIENE TODO
		ArrayList<String> pathEmpresasTipo0 = new ArrayList<String>(); // TODAS

		// Tipos de empresa segun MARKET CAP (0-6)
		ArrayList<String> pathEmpresasTipo1 = new ArrayList<String>();// MARKETCAP=MEGA
		ArrayList<String> pathEmpresasTipo2 = new ArrayList<String>();// MARKETCAP=LARGA
		ArrayList<String> pathEmpresasTipo3 = new ArrayList<String>();// MARKETCAP=MID
		ArrayList<String> pathEmpresasTipo4 = new ArrayList<String>();// MARKETCAP=SMALL
		ArrayList<String> pathEmpresasTipo5 = new ArrayList<String>();// MARKETCAP=MICRO
		ArrayList<String> pathEmpresasTipo6 = new ArrayList<String>();// MARKETCAP=NANO

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

		// Empresas alcistas (lista manual)
		ArrayList<String> pathEmpresasTipo26 = new ArrayList<String>();

		// Tipos de empresa segun ratio SMA50 de precio
		ArrayList<String> pathEmpresasTipo27 = new ArrayList<String>(); // bajo
		ArrayList<String> pathEmpresasTipo28 = new ArrayList<String>(); // medio
		ArrayList<String> pathEmpresasTipo29 = new ArrayList<String>(); // alto
		ArrayList<String> pathEmpresasTipo30 = new ArrayList<String>(); // muy alto

		// Tipos de empresa segun geografia
		ArrayList<String> pathEmpresasTipo31 = new ArrayList<String>(); // BeNeLux
		ArrayList<String> pathEmpresasTipo32 = new ArrayList<String>(); // China
		ArrayList<String> pathEmpresasTipo33 = new ArrayList<String>(); // Israel
		ArrayList<String> pathEmpresasTipo34 = new ArrayList<String>(); // Europe
		ArrayList<String> pathEmpresasTipo35 = new ArrayList<String>(); // resto explicito
		ArrayList<String> pathEmpresasTipo36 = new ArrayList<String>(); // desconocido

		// Tipos de empresa según "% Institutional Own"
		ArrayList<String> pathEmpresasTipo37 = new ArrayList<String>(); // 0-20
		ArrayList<String> pathEmpresasTipo38 = new ArrayList<String>(); // 20-60
		ArrayList<String> pathEmpresasTipo39 = new ArrayList<String>(); // 60-100
		ArrayList<String> pathEmpresasTipo40 = new ArrayList<String>(); // desconocido

		// Tipos de empresa según "% Dividend"
		ArrayList<String> pathEmpresasTipo41 = new ArrayList<String>(); // CON dividendo
		ArrayList<String> pathEmpresasTipo42 = new ArrayList<String>(); // SIN dividendo

		// Combinaciones manuales
		ArrayList<String> pathEmpresasTipo43 = new ArrayList<String>(); // Healthcare sin dividendo (SG 11 + SG42)
		ArrayList<String> pathEmpresasTipo44 = new ArrayList<String>(); // Pico en Volumen en corto plazo
		ArrayList<String> pathEmpresasTipo45 = new ArrayList<String>(); // Pico en Volumen en largo plazo

		// Si tiene operaciones de INSIDERS
		ArrayList<String> pathEmpresasTipo46 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo47 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo48 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo49 = new ArrayList<String>();

		// Subgrupos calculados con algoritmo clustering "alternativo" (automático)
		// usando algunas de las variables estaticas
		ArrayList<String> pathEmpresasTipo50 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo51 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo52 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo53 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo54 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo55 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo56 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo57 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo58 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo59 = new ArrayList<String>();

		int contadorTotal = 0, contadorDescartadasPorEmpleados = 0, contadorDescartadasPorDeuda = 0,
				contadorDescartadasPorQR = 0, contadorDescartadasPorRecom = 0, contadorDescartadasPorPER = 0;

		// Para cada EMPRESA
		while (iterator.hasNext()) {

			contadorTotal++;

			gestorFicheros = new GestorFicheros();
			datosEntrada = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
			ficheroGestionado = iterator.next();
			MY_LOGGER.debug("Fichero entrada: " + ficheroGestionado.getAbsolutePath());
//			MY_LOGGER.info(">>>>>>>>Fichero entrada: "+ ficheroGestionado.getAbsolutePath());
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

			// MY_LOGGER.info("EMPRESA ANALIZADA: " + empresa);

			// EXTRACCIÓN DE DATOS DE LA EMPRESA: sólo se usan los datos ESTATICOS, así que
			// basta coger la PRIMERA fila de datos
//			MY_LOGGER.info(">>>>>empresa: "+empresa);
			datosEmpresaEntrada = datosEntrada.get(empresa);

			Set<Integer> a = datosEmpresaEntrada.keySet();
			Integer indicePrimeraFilaDeDatos = null;
			if (a.iterator().hasNext()) {
				indicePrimeraFilaDeDatos = a.iterator().next();
			}
//			MY_LOGGER.info(">>>>>indicePrimeraFilaDeDatos: "+indicePrimeraFilaDeDatos);
			parametros = datosEmpresaEntrada.get(indicePrimeraFilaDeDatos); // PRIMERA FILA

			boolean empleadosDesconocidos = false, suficientesEmpleados = false;
			boolean deudaDesconocida = false, deudaConocidaYBaja = false;
			boolean quickRatioDesconocido = false, suficienteLiquidezSegunQuickRatio = false;
			boolean recomAnalistasDesconocido = false, analistasRecomiendanComprar = false;
			boolean ratioPERDesconocido = false, ratioPERRazonable = false;

			if (parametros != null) {

//				for (String param : parametros.keySet()) {
//					MY_LOGGER.info(param + "->" + parametros.get(param));
//				}

				String empleados = parametros.get("Employees");
				empleadosDesconocidos = (empleados == null || empleados.equals("-") || empleados.isEmpty());
				suficientesEmpleados = !empleadosDesconocidos
						&& Integer.valueOf(empleados) >= MINIMO_NUMERO_EMPLEADOS_INFORMADO;

//				if (empleadosDesconocidos) {
//					MY_LOGGER.info("Numero de empleados desconocido (" + empleados + ") en empresa=" + empresa
//							+ " ==>DESCARTADA");
//					contadorDescartadasPorEmpleados++;
//				}
				if (suficientesEmpleados == false) {
					MY_LOGGER
							.info("Motivo suficiente para DESCARTE en empresa=" + empresa + " porque tiene " + empleados
									+ " empleados (umbral=" + MINIMO_NUMERO_EMPLEADOS_INFORMADO + "): " + empleados);
					contadorDescartadasPorEmpleados++;
				}

				String deudaTotal = parametros.get("Debt/Eq");
				deudaDesconocida = deudaTotal != null && (deudaTotal.equals("-") || deudaTotal.isEmpty());
				deudaConocidaYBaja = deudaTotal != null && !deudaTotal.equals("-") && !deudaTotal.isEmpty()
						&& Float.valueOf(deudaTotal) <= MAX_DEUDA_PERMITIDA;
				if (deudaDesconocida) {
					MY_LOGGER.info("Deuda desconocida en empresa=" + empresa + "==>DESCARTADA");
					contadorDescartadasPorDeuda++;

				} else if (deudaConocidaYBaja == false) {
					MY_LOGGER.info("Motivo suficiente para DESCARTE en empresa=" + empresa
							+ " porque tiene DEUDA muy alta (umbral=" + MAX_DEUDA_PERMITIDA + "): " + deudaTotal);
					contadorDescartadasPorDeuda++;
				}

				// quick ratio (Test acido) -> https://es.wikipedia.org/wiki/Test_%C3%A1cido
				String quickRatio = parametros.get("Quick Ratio");
				quickRatioDesconocido = quickRatio != null && (quickRatio.equals("-") || quickRatio.isEmpty());
				suficienteLiquidezSegunQuickRatio = quickRatio != null && !quickRatio.equals("-")
						&& !quickRatio.isEmpty() && Float.valueOf(quickRatio) >= MIN_QUICK_RATIO;
				if (quickRatioDesconocido) {
					// MY_LOGGER.info("Permitida, aunque no conocemos el quickRatio de la empresa="
					// + empresa);

				} else if (suficienteLiquidezSegunQuickRatio == false) {
					MY_LOGGER.info("Motivo suficiente para DESCARTE en empresa=" + empresa
							+ " porque tiene poca LIQUIDEZ INMEDIATA (QUICK RATIO) (umbral=" + MIN_QUICK_RATIO + "): "
							+ quickRatio);
					contadorDescartadasPorQR++;
				}

				// Recomendaciones de analistas (de finviz)
				String recomAnalistas = parametros.get("Recom");
				recomAnalistasDesconocido = recomAnalistas != null
						&& (recomAnalistas.equals("-") || recomAnalistas.isEmpty());
				analistasRecomiendanComprar = recomAnalistas != null && !recomAnalistas.equals("-")
						&& !recomAnalistas.isEmpty() && Float.valueOf(recomAnalistas) <= MAX_RECOM_ANALISTAS;
				if (recomAnalistasDesconocido) {
//					MY_LOGGER.info("Permitida, aunque no conocemos la recomendacion de analistas de la empresa=" + empresa);

				} else if (analistasRecomiendanComprar == false) {
					MY_LOGGER.info("Motivo suficiente para DESCARTE en empresa=" + empresa
							+ " porque analistas recomiendan vender (umbral=" + MAX_RECOM_ANALISTAS + "): "
							+ recomAnalistas);
					contadorDescartadasPorRecom++;

				}

				// Ratio PER
				String ratioPER = parametros.get("P/E");
				ratioPERDesconocido = ratioPER == null || ratioPER.isEmpty() || ratioPER.equals("-");
				ratioPERRazonable = ratioPER != null && !ratioPER.isEmpty() && !ratioPER.equals("-")
						&& Float.valueOf(ratioPER) <= MAX_PER;
				if (ratioPERDesconocido) {
					// MY_LOGGER.info("Permitida, aunque no conocemos el ratio PER de la empresa=" +
					// empresa);

				} else if (ratioPERRazonable == false) {
					MY_LOGGER.info("Motivo suficiente para DESCARTE en empresa=" + empresa
							+ " porque el PER es demasiado alto (umbral=" + MAX_PER + "): " + ratioPER);
					contadorDescartadasPorPER++;
				}

			}

			boolean empresaCumpleCriteriosComunes = (
			// empleadosDesconocidos == false &&
			suficientesEmpleados) && (deudaDesconocida == false && deudaConocidaYBaja)
					&& (quickRatioDesconocido || suficienteLiquidezSegunQuickRatio)
					&& (recomAnalistasDesconocido || analistasRecomiendanComprar)
					&& (ratioPERDesconocido || ratioPERRazonable);

			String empresaCumpleCriteriosComunesStr = empresaCumpleCriteriosComunes ? "ENTRA" : "DESCARTADA";

			MY_LOGGER.info("EMPRESA: " + empresa + " ==> " + empresaCumpleCriteriosComunesStr + " ==>"
					+ " suficientesEmpleados:" + suficientesEmpleados + " deudaConocidaYBaja:" + deudaConocidaYBaja
					+ " suficienteLiquidezSegunQuickRatio:" + suficienteLiquidezSegunQuickRatio
					+ " analistasRecomiendanComprar:" + analistasRecomiendanComprar + " ratioPERDesconocido:"
					+ ratioPERDesconocido + " ratioPERRazonable:" + ratioPERRazonable);

			if (parametros != null && empresaCumpleCriteriosComunes) {

				// Para el subgrupo 0 siempre se añade
				// NO QUITAR, PARA QUE LAS GRÁFICAS FINALES SE PINTEN BASADAS EN ESTE GRUPO,
				// QUE CONTIENE TODO
				pathEmpresasTipo0.add(ficheroGestionado.getAbsolutePath());

				String dinamica1Str = parametros.get("DINAMICA1");
				String dinamica2Str = parametros.get("DINAMICA2");
				Integer dinamica1 = 0;
				if (dinamica1Str != null && !dinamica1Str.isEmpty()) {
					dinamica1 = Integer.valueOf(dinamica1Str);
				}

				Integer dinamica2 = 0;
				if (isNumeric(dinamica2Str)) {
//					MY_LOGGER.info("dinamica2Str: " + dinamica2Str);
					dinamica2 = Integer.valueOf(dinamica2Str);
				}

				// FILTRO DINÁMICO 1 (para reducir el uso de SMOTEENN):
				// Se aplica el filtro dinámico a todos los subgrupos excepto al grupo 0 (porque
				// es la base donde estarán todas las empresas con las que comparar
				// Se añade la empresa sólo si el filtro dinámico está desactivado, o si el
				// filtro dinámico está activado + el parámetro calculado llamado dinamica1=1 en
				// la fila 0 de la empresa

				// FILTRO DINÁMICO 2 (para reducir el uso de SMOTEENN):
				// Se aplica el filtro dinámico a todos los subgrupos excepto al grupo 0 (porque
				// es la base donde estarán todas las empresas con las que comparar
				// Se añade la empresa sólo si el filtro dinámico está desactivado, o si el
				// filtro dinámico está activado + el parámetro calculado llamado dinamica2=1 en
				// la fila 0 de la empresa
				if (filtroDinamico1 == 0 || (filtroDinamico1 == 1 && dinamica1 == 1)) {
					if (filtroDinamico2 == 0 || (filtroDinamico2 == 1 && dinamica2 == 1)) {

						// ------ SUBGRUPOS según MARKET CAP ------------
						String mcStr = parametros.get("Market Cap");
						Float marketCapValor = null;

						if (mcStr != null && !mcStr.isEmpty() && !"-".equals(mcStr)) {

							marketCapValor = Float.valueOf(mcStr);

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

							if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_BM)
									|| sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_UTIL)
									|| sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_ENERGY)) {
								pathEmpresasTipo7.add(ficheroGestionado.getAbsolutePath());
							} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_CONG)) {
								pathEmpresasTipo8.add(ficheroGestionado.getAbsolutePath());
							} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_CONSGO)
									|| sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_CONSCY)
									|| sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_CONSDEF)) {
								pathEmpresasTipo9.add(ficheroGestionado.getAbsolutePath());
							} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_FIN)) {
								pathEmpresasTipo10.add(ficheroGestionado.getAbsolutePath());
							} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_HC)) {
								pathEmpresasTipo11.add(ficheroGestionado.getAbsolutePath());
							} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_IG)) {
								pathEmpresasTipo12.add(ficheroGestionado.getAbsolutePath());
							} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_SERV)) {
								pathEmpresasTipo13.add(ficheroGestionado.getAbsolutePath());
							} else if (sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_TECH)
									|| sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_COMM)) {
								pathEmpresasTipo14.add(ficheroGestionado.getAbsolutePath());
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
						Float per = null;

						if (perStr != null && !perStr.isEmpty() && !"-".equals(perStr)) {
							per = Float.valueOf(perStr);

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
						Float debtEq = null;

						if (debtEqStr != null && !debtEqStr.isEmpty() && !"-".equals(debtEqStr)) {
							debtEq = Float.valueOf(debtEqStr);

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
							// pathEmpresasTipoXX.add(ficheroGestionado.getAbsolutePath());
							// MY_LOGGER.warn("Empresa = " + empresa + " con Debt/Eq = " + debtEqStr);
						}

						// ----------- SUBGRUPO CON LISTA MANUAL DE EMPRESAS -----
						if (listaSeleccionManual.contains(empresa)) {
							pathEmpresasTipo26.add(ficheroGestionado.getAbsolutePath());
						}

						// ------ SUBGRUPOS según ratio de SMA50 de precio ------------
						String ratioSMA50PrecioStr = parametros.get("RATIO_MAXRELATIVO_50_CLOSE");
						Integer ratioSMA50Precio = null;

						if (ratioSMA50PrecioStr != null && !ratioSMA50PrecioStr.contains("null")
								&& !ratioSMA50PrecioStr.isEmpty() && !"-".equals(ratioSMA50PrecioStr)) {
							ratioSMA50Precio = Integer.valueOf(ratioSMA50PrecioStr);

							if (ratioSMA50Precio > 0 && ratioSMA50Precio < SMA50RATIOPRECIO_umbral1) {
								pathEmpresasTipo27.add(ficheroGestionado.getAbsolutePath());
								System.out.println("empresa=" + empresa + " RATIO_MAXRELATIVO_50_CLOSE="
										+ ratioSMA50Precio + " -->MUY BAJO");
							} else if (ratioSMA50Precio >= SMA50RATIOPRECIO_umbral1
									&& ratioSMA50Precio < SMA50RATIOPRECIO_umbral2) {
								pathEmpresasTipo28.add(ficheroGestionado.getAbsolutePath());
								System.out.println("empresa=" + empresa + " RATIO_MAXRELATIVO_50_CLOSE="
										+ ratioSMA50Precio + " -->BAJO");
							} else if (ratioSMA50Precio >= SMA50RATIOPRECIO_umbral2
									&& ratioSMA50Precio < SMA50RATIOPRECIO_umbral3) {
								pathEmpresasTipo29.add(ficheroGestionado.getAbsolutePath());
								System.out.println("empresa=" + empresa + " RATIO_MAXRELATIVO_50_CLOSE="
										+ ratioSMA50Precio + " -->ALTO");
							} else {
								pathEmpresasTipo30.add(ficheroGestionado.getAbsolutePath());
								System.out.println("empresa=" + empresa + " RATIO_MAXRELATIVO_50_CLOSE="
										+ ratioSMA50Precio + " -->MUY ALTO");
							}

						} else {

							// Empresa con ratio desconocido (alto o bajo) --> No la usamos

							// MY_LOGGER.warn("Empresa = " + empresa + " con RATIO_SMA_50_PRECIO = " +
							// ratioSMA50PrecioStr);
						}

						// ------ SUBGRUPOS según GEOGRAFIA ------------
						String geoStr = parametros.get("geo");

						if (geoStr != null && !geoStr.isEmpty() && !"-".equals(geoStr)) {

							if (geoStr.contains("Netherlands") || geoStr.contains("BeNeLux")
									|| geoStr.contains("Belgium") || geoStr.contains("Luxembourg")) {
								pathEmpresasTipo31.add(ficheroGestionado.getAbsolutePath());
							} else if (geoStr.equalsIgnoreCase("China")) {
								pathEmpresasTipo32.add(ficheroGestionado.getAbsolutePath());
							} else if (geoStr.equalsIgnoreCase("Israel")) {
								pathEmpresasTipo33.add(ficheroGestionado.getAbsolutePath());
							} else if (esUnionEuropeaSeria(geoStr)) {
								pathEmpresasTipo34.add(ficheroGestionado.getAbsolutePath());
							} else {
								pathEmpresasTipo35.add(ficheroGestionado.getAbsolutePath());
							}

						} else {
							pathEmpresasTipo36.add(ficheroGestionado.getAbsolutePath());
						}

						// ------ SUBGRUPOS según INSTITUTIONAL OWN ------------
						String instOwnStr = parametros.get("Inst Own");
						Float instOwn = null;

						if (instOwnStr != null && !instOwnStr.isEmpty() && !"-".equals(instOwnStr)) {
							instOwn = Float.valueOf(instOwnStr);

							if (instOwn > 0 && instOwn < IO_umbral1) {
								pathEmpresasTipo37.add(ficheroGestionado.getAbsolutePath());
							} else if (instOwn >= IO_umbral1 && instOwn < IO_umbral2) {
								pathEmpresasTipo38.add(ficheroGestionado.getAbsolutePath());
							} else if (instOwn >= IO_umbral2 && instOwn <= 100) {
								pathEmpresasTipo39.add(ficheroGestionado.getAbsolutePath());
							} else {
								pathEmpresasTipo40.add(ficheroGestionado.getAbsolutePath());
							}

						} else {
							pathEmpresasTipo40.add(ficheroGestionado.getAbsolutePath());
						}

						// ------ SUBGRUPOS según %Dividendo ------------
						String pctDividendoStr = parametros.get("Dividend %");
						Float pctDividendo = null;

						if (pctDividendoStr != null && !pctDividendoStr.isEmpty()
								&& !"-".equals(pctDividendoStr.replace("%", "").trim())) {
							pctDividendo = Float.valueOf(pctDividendoStr.replace("%", "").trim());

							if (pctDividendo > 0) {
								pathEmpresasTipo41.add(ficheroGestionado.getAbsolutePath());
							} else {
								pathEmpresasTipo42.add(ficheroGestionado.getAbsolutePath());
							}

						} else {
							pathEmpresasTipo42.add(ficheroGestionado.getAbsolutePath());
						}

						// HEALTHCARE y SIN DIVIDENDO
						if (sectorStr != null && !sectorStr.isEmpty() && !"-".equals(sectorStr)
								&& sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_HC)
								&& pctDividendo == null) {
							pathEmpresasTipo43.add(ficheroGestionado.getAbsolutePath());
						}

						// --------------------------------------------
						// Casos en los que recientemente ha habido un pico en volumen,
						// respecto de la media de muchos dias antes

						// PICO en VOLUMEN
						String ratioMaxVolumenCP = parametros.get("VARREL_7_VOLUMEN");
						String ratioMaxVolumenMP = parametros.get("VARREL_20_VOLUMEN");

						if (parametros.containsKey("VARREL_7_VOLUMEN") == false
								&& parametros.containsKey("VARREL_20_VOLUMEN") == false) {

							MY_LOGGER.error("Al construir SG_44 y SG_45 para empresa " + empresa
									+ " faltan columnas esperables.");

						} else {

							if (ratioMaxVolumenCP != null && !ratioMaxVolumenCP.isEmpty()
									&& !"-".equals(ratioMaxVolumenCP) && !"null".equals(ratioMaxVolumenCP)) {

								Float maxVolumenCPf = Float.valueOf(ratioMaxVolumenCP);

								// A corto plazo ha habido un pico en volumen
								if (maxVolumenCPf > factorPicoVolumenCP) {
									pathEmpresasTipo44.add(ficheroGestionado.getAbsolutePath());
								}
							}

							if (ratioMaxVolumenMP != null && !ratioMaxVolumenMP.isEmpty()
									&& !"-".equals(ratioMaxVolumenMP) && !"null".equals(ratioMaxVolumenMP)) {

								Float maxVolumenMPf = Float.valueOf(ratioMaxVolumenMP);

								// A medio plazo ha habido un pico en volumen
								if (maxVolumenMPf > factorPicoVolumenMP) {
									pathEmpresasTipo45.add(ficheroGestionado.getAbsolutePath());
								}
							}
						}

						// ------ SUBGRUPOS según OPERACIONES DE INSIDERS ------------
						String insiders90dias = parametros.get("flagOperacionesInsiderUltimos90dias");
						String insiders30dias = parametros.get("flagOperacionesInsiderUltimos30dias");
						String insiders15dias = parametros.get("flagOperacionesInsiderUltimos15dias");
						String insiders5dias = parametros.get("flagOperacionesInsiderUltimos5dias");

						if (insiders90dias != null && !insiders90dias.isEmpty()) {
							pathEmpresasTipo46.add(ficheroGestionado.getAbsolutePath());
						}
						if (insiders30dias != null && !insiders30dias.isEmpty()) {
							pathEmpresasTipo47.add(ficheroGestionado.getAbsolutePath());
						}
						if (insiders15dias != null && !insiders15dias.isEmpty()) {
							pathEmpresasTipo48.add(ficheroGestionado.getAbsolutePath());
						}
						if (insiders5dias != null && !insiders5dias.isEmpty()) {
							pathEmpresasTipo49.add(ficheroGestionado.getAbsolutePath());
						}

						// ---------- SUBGRUPOS CALCULADOS CON CLUSTERING ------
						if (listaSeleccionClustering10.contains(empresa)) {
							pathEmpresasTipo50.add(ficheroGestionado.getAbsolutePath());
						}
						if (listaSeleccionClustering1.contains(empresa)) {
							pathEmpresasTipo51.add(ficheroGestionado.getAbsolutePath());
						}
						if (listaSeleccionClustering2.contains(empresa)) {
							pathEmpresasTipo52.add(ficheroGestionado.getAbsolutePath());
						}
						if (listaSeleccionClustering3.contains(empresa)) {
							pathEmpresasTipo53.add(ficheroGestionado.getAbsolutePath());
						}
						if (listaSeleccionClustering4.contains(empresa)) {
							pathEmpresasTipo54.add(ficheroGestionado.getAbsolutePath());
						}
						if (listaSeleccionClustering5.contains(empresa)) {
							pathEmpresasTipo55.add(ficheroGestionado.getAbsolutePath());
						}
						if (listaSeleccionClustering6.contains(empresa)) {
							pathEmpresasTipo56.add(ficheroGestionado.getAbsolutePath());
						}
						if (listaSeleccionClustering7.contains(empresa)) {
							pathEmpresasTipo57.add(ficheroGestionado.getAbsolutePath());
						}
						if (listaSeleccionClustering8.contains(empresa)) {
							pathEmpresasTipo58.add(ficheroGestionado.getAbsolutePath());
						}
						if (listaSeleccionClustering9.contains(empresa)) {
							pathEmpresasTipo59.add(ficheroGestionado.getAbsolutePath());
						}

					}
				}
				// ---------------------------------------------------------------------------------------
			}
		}

		MY_LOGGER.info("=============== CONTADORES DE EMPRESAS PROCESADAS ===============");
		MY_LOGGER.info("Numero empresas total a la ENTRADA: " + contadorTotal + " empresas");
		MY_LOGGER.info("De las descartadas, algunas tienen pocos EMPLEADOS (< " + MINIMO_NUMERO_EMPLEADOS_INFORMADO
				+ "): " + contadorDescartadasPorEmpleados + " empresas o lo desconocemos");
		MY_LOGGER.info("De las descartadas, algunas tienen demasiada DEUDA (> " + (100 * MAX_DEUDA_PERMITIDA) + " %): "
				+ contadorDescartadasPorDeuda + " empresas o lo desconocemos");
		MY_LOGGER.info("De las descartadas, algunas tienen un QUICK RATIO muy bajo (< " + MIN_QUICK_RATIO + "): "
				+ contadorDescartadasPorQR + " empresas (conocemos el dato concreto)");
		MY_LOGGER.info("De las descartadas, algunas tienen RECOMENDACION de VENTA FUERTE (>" + MAX_RECOM_ANALISTAS
				+ "): " + contadorDescartadasPorRecom + " empresas (conocemos el dato concreto)");
		MY_LOGGER.info("De las descartadas, algunas tienen el ratio PER demasiado alto (> " + MAX_PER + " años): "
				+ contadorDescartadasPorPER + " empresas (conocemos el dato concreto)");
		MY_LOGGER.info("Numero empresas total a la SALIDA (con las que hacemos los subgrupos): "
				+ pathEmpresasTipo0.size() + " empresas");
		MY_LOGGER.info("=================================================================");

		// Almacenamiento del tipo de empresa en la lista
		empresasPorTipo = new HashMap<Integer, ArrayList<String>>();
		// Para el subgrupo 0 siempre se añade
		// NO QUITAR, PARA QUE LAS GRÁFICAS FINALES SE PINTEN BASADAS EN ESTE GRUPO,
		// QUE CONTIENE TODOS
		MY_LOGGER.info("El SUBGRUPO 0 tiene " + pathEmpresasTipo0.size() + " empresas");
		empresasPorTipo.put(0, pathEmpresasTipo0);

		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 1, pathEmpresasTipo1, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 2, pathEmpresasTipo2, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 3, pathEmpresasTipo3, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 4, pathEmpresasTipo4, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 5, pathEmpresasTipo5, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 6, pathEmpresasTipo6, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 7, pathEmpresasTipo7, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 8, pathEmpresasTipo8, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 9, pathEmpresasTipo9, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 10, pathEmpresasTipo10, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 11, pathEmpresasTipo11, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 12, pathEmpresasTipo12, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 13, pathEmpresasTipo13, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 14, pathEmpresasTipo14, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 15, pathEmpresasTipo15, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 16, pathEmpresasTipo16, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 17, pathEmpresasTipo17, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 18, pathEmpresasTipo18, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 19, pathEmpresasTipo19, realimentacion);
		////// decidirSiMeterSubgrupoEnLista(empresasPorTipo, 20, pathEmpresasTipo20,
		////// realimentacion);//No meter dinero real NUNCA aqui
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 21, pathEmpresasTipo21, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 22, pathEmpresasTipo22, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 23, pathEmpresasTipo23, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 24, pathEmpresasTipo24, realimentacion);
		////// decidirSiMeterSubgrupoEnLista(empresasPorTipo, 25, pathEmpresasTipo25,
		////// realimentacion);//No meter dinero real NUNCA aqui
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 26, pathEmpresasTipo26, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 27, pathEmpresasTipo27, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 28, pathEmpresasTipo28, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 29, pathEmpresasTipo29, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 30, pathEmpresasTipo30, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 31, pathEmpresasTipo31, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 32, pathEmpresasTipo32, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 33, pathEmpresasTipo33, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 34, pathEmpresasTipo34, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 35, pathEmpresasTipo35, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 36, pathEmpresasTipo36, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 37, pathEmpresasTipo37, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 38, pathEmpresasTipo38, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 39, pathEmpresasTipo39, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 40, pathEmpresasTipo40, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 41, pathEmpresasTipo41, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 42, pathEmpresasTipo42, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 43, pathEmpresasTipo43, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 44, pathEmpresasTipo44, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 45, pathEmpresasTipo45, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 46, pathEmpresasTipo46, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 47, pathEmpresasTipo47, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 48, pathEmpresasTipo48, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 49, pathEmpresasTipo49, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 50, pathEmpresasTipo50, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 51, pathEmpresasTipo51, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 52, pathEmpresasTipo52, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 53, pathEmpresasTipo53, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 54, pathEmpresasTipo54, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 55, pathEmpresasTipo55, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 56, pathEmpresasTipo56, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 57, pathEmpresasTipo57, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 58, pathEmpresasTipo58, realimentacion);
		decidirSiMeterSubgrupoEnLista(empresasPorTipo, 59, pathEmpresasTipo59, realimentacion);

		MY_LOGGER.info("Se crea un CSV para cada subgrupo...");
		Set<Integer> tipos = empresasPorTipo.keySet();
		List<Integer> tiposLista = new ArrayList<Integer>();
		tiposLista.addAll(tipos);
		Collections.sort(tiposLista); // ordenado, para leerlo mejor
		Iterator<Integer> itTipos = tiposLista.iterator();
		Integer numSubgruposCreados = empresasPorTipo.size();
		Integer tipo;
		String pathFichero;
		String row, rowTratada;
		Boolean esPrimeraLinea;

		// En el Gestor de Ficheros aparecen los nombres de los parámetros estáticos a
		// eliminar. Sólo se cuentan. Habrá tantos pipes como parámetros
		Integer numeroParametrosEstaticos = gestorFicheros.getOrdenNombresParametrosLeidos().size();

		String ficheroOut, ficheroListadoOut;
		ArrayList<String> pathFicheros;
		FileWriter csvWriter;
		FileWriter writerListadoEmpresas;

		Double coberturaEmpresasPorCluster;
		Estadisticas estadisticas;
		String pathEmpresa;
		HashMap<String, Boolean> empresasConTarget;
		Iterator<String> itEmpresas;

		MY_LOGGER.info("Iteracion para cada subgrupo (el subgrupo 0 es gigante y suele tardar cierto tiempo)...");
		while (numSubgruposCreados > 0 && itTipos.hasNext()) {

			tipo = itTipos.next();
			// MY_LOGGER.info("***** Subgrupo " + tipo + " *****");

			ArrayList<String> pathFicherosEmpresas = empresasPorTipo.get(tipo);

			// *******************************************************************
			// Mete FEATURES DE SUBGRUPO en los CSVs de entrada (elaborados)
			GeneradorFeaturesDeSubgrupo.anhadirColumnasDependientesDelSubgrupo(pathFicherosEmpresas, MY_LOGGER);
			// *******************************************************************

			// Antes se comprobará, en cada cluster, qué porcentaje hay de empresas con al
			// menos una vela con target=1,
			// respecto del total de empresas del cluster (esto se llama Cobertura).
			// Sólo se guardarán los clusters con cobertura mayor que una cantidad mínima.

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
				MY_LOGGER.debug(
						"Empresa: " + pathEmpresa + " ¿tiene algún target=1? " + empresasConTarget.get(pathEmpresa));
			}

			// Se calcula la cobertura del target
			coberturaEmpresasPorCluster = estadisticas.getMean();
			MY_LOGGER.debug(
					"COBERTURA DEL cluster " + tipo + ": " + Math.round(coberturaEmpresasPorCluster * 100) + "%");

			// Para generar un fichero de dataset del cluster, la cobertura debe ser mayor
			// que un x%
			if (modoTiempo.equalsIgnoreCase("pasado")
					&& coberturaEmpresasPorCluster * 100 < Double.valueOf(coberturaMinima)) {
				MY_LOGGER.warn(
						"Cluster " + tipo + " con " + Math.round(coberturaEmpresasPorCluster * 100) + "% empresas ("
								+ estadisticas.getSum() + " de " + estadisticas.getValues().length + " ; mínimo = "
								+ coberturaMinima + " %) con al menos una vela positiva." + " NO SE GENERA DATASET");

			} else if (modoTiempo.equalsIgnoreCase("pasado")
					&& empresasConTarget.keySet().size() < Integer.valueOf(minEmpresasPorCluster)) {
				MY_LOGGER.warn("Cluster " + tipo + ", tiene " + empresasConTarget.keySet().size()
						+ " empresas. Demasiado pequeño (mínimo= " + minEmpresasPorCluster
						+ " empresas). NO SE GENERA DATASET");
			} else {

				if (modoTiempo.equalsIgnoreCase("pasado")) {
					MY_LOGGER.info("Cluster " + tipo + " con " + Math.round(coberturaEmpresasPorCluster * 100)
							+ "% empresas (" + estadisticas.getSum() + " de " + estadisticas.getValues().length
							+ " ; mínimo = " + coberturaMinima + " %) con al menos una vela positiva." + " Y tiene "
							+ empresasConTarget.keySet().size() + " empresas (mínimo deseado = " + minEmpresasPorCluster
							+ ")." + " ==> SI SE GENERA DATASET");
				} else {
					MY_LOGGER.info("Cluster " + tipo + " tiene " + empresasConTarget.keySet().size() + " empresas."
							+ " En modo FUTURO, SIEMPRE SE GENERA DATASET.");
				}

				// Hay alguna empresa de este tipo. Creo un CSV común para todas las del mismo
				// tipo
				pathFicheros = empresasPorTipo.get(tipo);

				String dirSubgrupoOut = directorioOut + "SG_" + tipo + "/";
				MY_LOGGER.debug("Creando la carpeta del subgrupo con ID=" + tipo + " en: " + dirSubgrupoOut);
				File dirSubgrupoOutFile = new File(dirSubgrupoOut);
				dirSubgrupoOutFile.mkdir();

				ficheroOut = dirSubgrupoOut + "COMPLETO.csv";
				ficheroListadoOut = dirSubgrupoOut + "EMPRESAS.txt";
				MY_LOGGER.info("CSV de subgrupo: " + ficheroOut);
				MY_LOGGER.info("Lista de empresas de subgrupo: " + ficheroListadoOut);
				csvWriter = new FileWriter(ficheroOut);
				writerListadoEmpresas = new FileWriter(ficheroListadoOut);

				int numColumnasCabecera = -1;
				String cabecera = "";

				for (int i = 0; i < pathFicheros.size(); i++) { // cada ITERACION es una EMPRESA

					esPrimeraLinea = Boolean.TRUE;
					// Se lee el fichero de la empresa a meter en el CSV común
					pathFichero = pathFicheros.get(i);
					MY_LOGGER.debug("Fichero a leer para clasificar en subgrupo: " + pathFichero);
					BufferedReader csvReader = new BufferedReader(new FileReader(pathFichero));

					// Añado la empresa al fichero de listado de empresas
					writerListadoEmpresas.append(pathFichero + "\n");

					try {

						while ((row = csvReader.readLine()) != null) {

							rowTratada = procesarFilaQuitandoColumnasEstaticasYAlgunasColumnasEspeciales(row, tipo); // IMPORTANTE

							// La cabecera se toma de la primera línea del primer fichero
							if (i == 0 && esPrimeraLinea) {
								// En la primera línea está la cabecera de parámetros
								cabecera = rowTratada;
								// se cuenta el numero de campos
								numColumnasCabecera = countElements(rowTratada);

								// Se valida que el nombre recibido es igual que el usado en la constructora, y
								// en dicho orden
								csvWriter.append(rowTratada);

							}
							if (!esPrimeraLinea) {
								// Para todos los ficheros, se tratan las filas 2 y siguientes

								// Se controla que el numero de campos sea el mismo que la cabecera
								if (numColumnasCabecera != countElements(rowTratada)) {
									MY_LOGGER.error("Al generar el fichero " + ficheroOut
											+ " se ha detectado que algunas filas  no tienen los mismos campos ("
											+ countElements(rowTratada) + ") que la cabecera (" + numColumnasCabecera
											+ "):");
									MY_LOGGER.error("Cabecera: " + cabecera);
									MY_LOGGER.error("Fila: " + rowTratada);
									MY_LOGGER.error("Se aborta la ejecución...");

									System.exit(-1);
								}

								// Se escribe en fichero
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

	/**
	 * Trata cada fila, incluida la cabecera. Quita las columnas ESTATICAS. Y quita
	 * algunas columnas especiales solo en ciertos casos en que estarían muy vacias
	 * (y en la capa 5 provocaría eliminar muchas filas).
	 * 
	 * @param row        Fila de entrada
	 * @param idSubgrupo
	 * @return Fila de salida
	 */
	public static String procesarFilaQuitandoColumnasEstaticasYAlgunasColumnasEspeciales(String row,
			Integer idSubgrupo) {

		MY_LOGGER.debug("Fila leída: " + row);

		String identificadoresYdinamicos;
		Character characterPipe = "|".charAt(0);

		// Cada fila leida tiene estas partes:
		// - identificadores de fila
		// - campos estáticos (usados sólo para crear subgrupos)
		// - campos dinámicos (datos diarios)
		//
		// Aquí se ELIMINAN los ESTÁTICOS, porque ya no hacen falta
		// También se añaden precios y volumen, para la validación económica en c7

		Integer indiceInicioInsiders = SubgruposUtils.indiceDeAparicion(characterPipe,
				GestorFicheros.INDICE_PRIMER_CAMPO_INSIDERS, row);
		Integer indiceInicioEstaticos = SubgruposUtils.indiceDeAparicion(characterPipe,
				GestorFicheros.INDICE_PRIMER_CAMPO_ESTATICO, row);

		boolean subgrupoDeOperacionesDeInsiders = idSubgrupo.intValue() == 46 || idSubgrupo.intValue() == 47
				|| idSubgrupo.intValue() == 48 || idSubgrupo.intValue() == 49;

		if (subgrupoDeOperacionesDeInsiders) {
			identificadoresYdinamicos = row.substring(0, indiceInicioEstaticos);
		} else {
			// Quitamos las columnas semi-dinamicas (operaciones de insiders) porque
			// mayoritariamente estan vacías (solo conocemos las operaciones de insiders de
			// muy pocas empresas)
			identificadoresYdinamicos = row.substring(0, indiceInicioInsiders);
		}

		String rowTratada = identificadoresYdinamicos + characterPipe + SubgruposUtils
				.recortaPrimeraParteDeString(characterPipe, GestorFicheros.INDICE_ULTIMO_CAMPO_ESTATICO + 1, row);

		MY_LOGGER.debug("Fila escrita: " + rowTratada);

		return rowTratada;
	}

	public static boolean isNumeric(String strNum) {
		if (strNum == null) {
			return false;
		}
		try {
			double d = Double.parseDouble(strNum);
		} catch (NumberFormatException nfe) {
			return false;
		}
		return true;
	}

	/**
	 * Indica si es un pais serio de la UE (se excluyen los poco serios, como los
	 * soviéticos o Grecia o los minúsculos como Islandia). Se incluye a España.
	 * 
	 * @param geoStr Nombre del pais que aparece en cada empresa en FINVIZ
	 * @return True si pertenece a la UE
	 */
	public static boolean esUnionEuropeaSeria(String geoStr) {

		return geoStr != null && !geoStr.isEmpty() && (

		geoStr.equalsIgnoreCase("belgium") || geoStr.equalsIgnoreCase("benelux") || geoStr.equalsIgnoreCase("denmark")
				|| geoStr.equalsIgnoreCase("finland") || geoStr.equalsIgnoreCase("france")
				|| geoStr.equalsIgnoreCase("germany") || geoStr.equalsIgnoreCase("ireland")
				|| geoStr.equalsIgnoreCase("italy") || geoStr.equalsIgnoreCase("luxembourg")
				|| geoStr.equalsIgnoreCase("netherlands") || geoStr.equalsIgnoreCase("norway")
				|| geoStr.equalsIgnoreCase("portugal") || geoStr.equalsIgnoreCase("spain")
				|| geoStr.equalsIgnoreCase("sweden") || geoStr.equalsIgnoreCase("unitedkingdom"));
	}

	/**
	 * Decide si ese subgrupo es válido para poder invertir. Descarta los que ya
	 * sabemos que tienen demasiados falsos positivos (encima de un umbral).
	 * 
	 * @param empresasPorTipo
	 * @param subgrupoId
	 * @param pathEmpresasTipo
	 * @param realimentacion
	 * @throws IOException
	 */
	public static void decidirSiMeterSubgrupoEnLista(HashMap<Integer, ArrayList<String>> empresasPorTipo,
			Integer subgrupoId, ArrayList<String> pathEmpresasTipo, String realimentacion) throws IOException {

		List<FalsosPositivosSubgrupo> listaSubgruposConDemasiadosFP = InterpreteFalsosPositivos
				.extraerSubgruposConDemasiadosFP();

		FalsosPositivosSubgrupo fps = null;
		for (FalsosPositivosSubgrupo item : listaSubgruposConDemasiadosFP) {
			if (item.subgrupoId.equals(subgrupoId)) {
				fps = item;
			}
		}

		if (esSubgrupoListaManualoClusteringAuto(subgrupoId)) {
			MY_LOGGER.info(
					"Subgrupo especial (lista manual o calculado con algoritmo de clustering). Metemos siempre el SUBGRUPO: "
							+ subgrupoId);
			empresasPorTipo.put(subgrupoId, pathEmpresasTipo);

		} else {

			if (realimentacion.equals("N")) {
				MY_LOGGER.info("Realimentacion no activa. Metemos siempre el SUBGRUPO: " + subgrupoId);
				empresasPorTipo.put(subgrupoId, pathEmpresasTipo);

			} else if (realimentacion.equals("S") && fps == null) {
				MY_LOGGER.info("El SUBGRUPO " + subgrupoId
						+ " no está en la lista de subgrupos analizados previamente. Si no tenemos info de falsos positivos, sí procesamos el subgrupo.");
				empresasPorTipo.put(subgrupoId, pathEmpresasTipo);

			} else if (realimentacion.equals("S")
					&& fps.ratioFalsosPositivos <= InterpreteFalsosPositivos.UMBRAL_MAX_RATIOSUBGRUPO_FP) {
				MY_LOGGER.info("SUBGRUPO " + subgrupoId + " conocido y debajo del umbral ==> Lo queremos. Subgrupo: "
						+ subgrupoId);
				empresasPorTipo.put(subgrupoId, pathEmpresasTipo);

			} else if (realimentacion.equals("S")
					&& fps.ratioFalsosPositivos > InterpreteFalsosPositivos.UMBRAL_MAX_RATIOSUBGRUPO_FP) {
				MY_LOGGER.info("SUBGRUPO " + subgrupoId + " con DEMASIADOS falsos positivos (ratio="
						+ fps.ratioFalsosPositivos + " %). No añadimos el subgrupo: " + subgrupoId);
			}
		}

	}

	/**
	 * Analiza si es un grupo especial: lista manual o calculado con algoritmo de
	 * clsutering
	 * 
	 * @param subgrupoId
	 * @return True si es un subgrupo de lista manual o calculado con algoritmo de
	 *         clustering. Devuelve false en otro caso.
	 */
	public static boolean esSubgrupoListaManualoClusteringAuto(Integer subgrupoId) {
		return subgrupoId.intValue() == 26 || (subgrupoId.intValue() >= 50 && subgrupoId.intValue() <= 59);
	}

	/**
	 * @return
	 * @throws IOException
	 */
	public static List<String> leerListaManualEmpresasSeleccionadas() throws IOException {

		String PATH_LISTA_MANUAL = "empresas_seleccion_manual/lista.csv";
		List<String> lista = new ArrayList<String>();

		FileReader fr = new FileReader(PATH_LISTA_MANUAL);
		BufferedReader br = new BufferedReader(fr);
		String actual;

		while ((actual = br.readLine()) != null) {
			if (actual != null && !actual.isEmpty()) {
				lista.add(actual);
			}
		}

		// Quitar duplicados
		List<String> listWithoutDuplicates = lista.stream().distinct().collect(Collectors.toList());

		// cerrar fichero
		br.close();

		return listWithoutDuplicates;
	}

	/**
	 * @return
	 * @throws IOException
	 */
	public static List<String> leerListaClusteringAlternativo(String idSubgrupoAlternativoElegido) throws IOException {

		String PATH_CSV_CLUSTERING_ALTERNATIVO = "/bolsa/pasado/empresas_clustering.csv";
		List<String> lista = new ArrayList<String>();
		if (!Files.exists(Paths.get(PATH_CSV_CLUSTERING_ALTERNATIVO))) {
			MY_LOGGER.error("No hay fichero de clustering alternativo. Revisar si se ha ejecutado (ver master.sh).");

		} else {
			FileReader fr = new FileReader(PATH_CSV_CLUSTERING_ALTERNATIVO);
			BufferedReader br = new BufferedReader(fr);
			String actual;

			while ((actual = br.readLine()) != null) {

				String[] partes = actual.split("\\|");

				if (actual != null && !actual.isEmpty() && partes[1].equals(idSubgrupoAlternativoElegido)) {
					lista.add(partes[0]);
				}
			}

			// cerrar fichero
			br.close();
		}

		// Quitar duplicados
		List<String> listWithoutDuplicates = lista.stream().distinct().collect(Collectors.toList());
		MY_LOGGER.info("leerListaClusteringAlternativo --> Subgrupo alternativo usado: " + idSubgrupoAlternativoElegido
				+ " --> " + listWithoutDuplicates.size() + " empresas");

		return listWithoutDuplicates;
	}

	/**
	 * Count the number of elements in a given CSV string that uses a pipe (|) as a
	 * separator.
	 *
	 * @param csv The input CSV string with | as separator.
	 * @return The number of elements in the CSV string.
	 */
	public static int countElements(String csv) {
		if (csv == null || csv.isEmpty()) {
			return 0;
		}

		String[] elements = csv.split("\\|");
		return elements.length;
	}

}
