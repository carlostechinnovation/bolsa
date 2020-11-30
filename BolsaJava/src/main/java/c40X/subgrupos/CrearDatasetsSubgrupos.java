package c40X.subgrupos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

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

	private final static Float DE_umbral1 = 0.7F;
	private final static Float DE_umbral2 = 1.5F;
	private final static Float DE_umbral3 = 2.8F;

	private final static Integer SMA50RATIOPRECIO_umbral1 = 80;
	private final static Integer SMA50RATIOPRECIO_umbral2 = 100;
	private final static Integer SMA50RATIOPRECIO_umbral3 = 120;

	private final static Float IO_umbral1 = 20.0F;
	private final static Float IO_umbral2 = 60.0F;

	private final static Float factorPicoVolumen = 1.09F;// Pico en volumen
	private final static Float factorPicoPrecio = 1.09F; // Pico en precio

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
		String modoTiempo = BrutosUtils.PASADO; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 5) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			directorioIn = args[0];
			directorioOut = args[1];
			coberturaMinima = args[2];
			minEmpresasPorCluster = args[3];
			modoTiempo = args[4];
		}

		crearSubgruposYNormalizar(directorioIn, directorioOut, coberturaMinima, minEmpresasPorCluster, modoTiempo);

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
			String minEmpresasPorCluster, String modoTiempo) throws Exception {

		// Debo leer el parámetro que me interese: de momento el market cap. En el
		// futuro sería conveniente separar por sector y liquidez (volumen medio de 6
		// meses en dólares).
		GestorFicheros gestorFicheros = new GestorFicheros();
//		System.out.println(">>>>directorioIn: "+directorioIn);
		File directorioEntrada = new File(directorioIn);
		HashMap<String, HashMap<Integer, HashMap<String, String>>> datosEntrada;
		ArrayList<File> ficherosEntradaEmpresas = gestorFicheros.listaFicherosDeDirectorio(directorioEntrada);
		Iterator<File> iterator = ficherosEntradaEmpresas.iterator();
		File ficheroGestionado;
		HashMap<String, String> parametros;

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
		ArrayList<String> pathEmpresasTipo26 = new ArrayList<String>(); // desconocido

		// Tipos de empresa segun ratio SMA50 de precio
		ArrayList<String> pathEmpresasTipo27 = new ArrayList<String>(); // bajo
		ArrayList<String> pathEmpresasTipo28 = new ArrayList<String>(); // medio
		ArrayList<String> pathEmpresasTipo29 = new ArrayList<String>(); // alto
		ArrayList<String> pathEmpresasTipo30 = new ArrayList<String>(); // muy alto
		ArrayList<String> pathEmpresasTipo31 = new ArrayList<String>(); // desconocido

		// Tipos de empresa segun geografia
		ArrayList<String> pathEmpresasTipo32 = new ArrayList<String>(); // china
		ArrayList<String> pathEmpresasTipo33 = new ArrayList<String>(); // USA
		ArrayList<String> pathEmpresasTipo34 = new ArrayList<String>(); // India
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
		ArrayList<String> pathEmpresasTipo44 = new ArrayList<String>(); // 3 días: Pico en Volumen y precio
		ArrayList<String> pathEmpresasTipo45 = new ArrayList<String>(); // 7 días: Pico en Volumen y precio

		// Si tiene operaciones de INSIDERS
		ArrayList<String> pathEmpresasTipo46 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo47 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo48 = new ArrayList<String>();
		ArrayList<String> pathEmpresasTipo49 = new ArrayList<String>();

		// Para cada EMPRESA
		while (iterator.hasNext()) {

			gestorFicheros = new GestorFicheros();
			datosEntrada = new HashMap<String, HashMap<Integer, HashMap<String, String>>>();
			ficheroGestionado = iterator.next();
			MY_LOGGER.debug("Fichero entrada: " + ficheroGestionado.getAbsolutePath());
//			System.out.println(">>>>>>>>Fichero entrada: "+ ficheroGestionado.getAbsolutePath());
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
//			System.out.println(">>>>>empresa: "+empresa);
			datosEmpresaEntrada = datosEntrada.get(empresa);

			Set<Integer> a = datosEmpresaEntrada.keySet();
			Integer indicePrimeraFilaDeDatos = null;
			if (a.iterator().hasNext()) {
				indicePrimeraFilaDeDatos = a.iterator().next();
			}
//			System.out.println(">>>>>indicePrimeraFilaDeDatos: "+indicePrimeraFilaDeDatos);
			parametros = datosEmpresaEntrada.get(indicePrimeraFilaDeDatos); // PRIMERA FILA

			if (parametros != null) {

				// Para el subgrupo 0 siempre se añade
				// NO QUITAR, PARA QUE LAS GRÁFICAS FINALES SE PINTEN BASADAS EN ESTE GRUPO,
				// QUE CONTIENE TODO
				pathEmpresasTipo0.add(ficheroGestionado.getAbsolutePath());

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
					pathEmpresasTipo26.add(ficheroGestionado.getAbsolutePath());
					// MY_LOGGER.warn("Empresa = " + empresa + " con Debt/Eq = " + debtEqStr);
				}

				// ------ SUBGRUPOS según ratio de SMA50 de precio ------------
				String ratioSMA50PrecioStr = parametros.get("RATIO_SMA_50_PRECIO");
				Integer ratioSMA50Precio = null;

				if (ratioSMA50PrecioStr != null && !ratioSMA50PrecioStr.contains("null")
						&& !ratioSMA50PrecioStr.isEmpty() && !"-".equals(ratioSMA50PrecioStr)) {
					ratioSMA50Precio = Integer.valueOf(ratioSMA50PrecioStr);

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

				// ------ SUBGRUPOS según GEOGRAFIA ------------
				String geoStr = parametros.get("geo");

				if (geoStr != null && !geoStr.isEmpty() && !"-".equals(geoStr)) {

					if (geoStr.equalsIgnoreCase("china")) {
						pathEmpresasTipo32.add(ficheroGestionado.getAbsolutePath());
					} else if (geoStr.equalsIgnoreCase("USA")) {
						pathEmpresasTipo33.add(ficheroGestionado.getAbsolutePath());
					} else if (geoStr.equalsIgnoreCase("India")) {
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
						&& sectorStr.equals(EstaticosFinvizDescargarYParsear.SECTOR_HC) && pctDividendo == null) {
					pathEmpresasTipo43.add(ficheroGestionado.getAbsolutePath());
				}

				// --------------------------------------------
				// Casos en los que recientemente ha habido un pico en volumen y precio,
				// respecto de la media de muchos dias antes

				// PICO en VOLUMEN
				String max3Volumen = parametros.get("MAXIMO_3_VOLUMEN"); // el pico se dio en las ultimas 3 velas
				String max7Volumen = parametros.get("MAXIMO_7_VOLUMEN"); // el pico se dio en las ultimas 7 velas
				String mediaSma20Volumen = parametros.get("MEDIA_SMA_20_VOLUMEN");

				// Pico en PRECIO
				String max3Precio = parametros.get("MAXIMO_3_PRECIO");// el pico se dio en las ultimas 3 velas
				String max7Precio = parametros.get("MAXIMO_7_PRECIO");// el pico se dio en las ultimas 7 velas
				String mediaSma20Precio = parametros.get("MEDIA_SMA_20_PRECIO");

				if (max3Volumen != null && !max3Volumen.isEmpty() && !"-".equals(max3Volumen)
						&& !"null".equals(max3Volumen)

						&& max7Volumen != null && !max7Volumen.isEmpty() && !"-".equals(max7Volumen)
						&& !"null".equals(max7Volumen)

						&& mediaSma20Volumen != null && !mediaSma20Volumen.isEmpty() && !"-".equals(mediaSma20Volumen)
						&& !"null".equals(mediaSma20Volumen)

						&& max3Precio != null && !max3Precio.isEmpty() && !"-".equals(max3Precio)
						&& !"null".equals(max3Precio)

						&& max7Precio != null && !max7Precio.isEmpty() && !"-".equals(max7Precio)
						&& !"null".equals(max7Precio)

						&& mediaSma20Precio != null && !mediaSma20Precio.isEmpty() && !"-".equals(mediaSma20Precio)
						&& !"null".equals(mediaSma20Precio)) {

					Float max3Volumenf = Float.valueOf(max3Volumen);
					Float max7Volumenf = Float.valueOf(max7Volumen);
					Float mediaSma20Volumenf = Float.valueOf(mediaSma20Volumen);
					Float max3Preciof = Float.valueOf(max3Precio);
					Float max7Preciof = Float.valueOf(max7Precio);
					Float mediaSma20Preciof = Float.valueOf(mediaSma20Precio);

					// En las ultimas 3 ó 7 velas ha habido un pico en volumen y precio, respecto de
					// la media de 20 dias. Ha caido el precio, pero puede que se repita el pico...
					boolean hayPicoEnVolumen3 = max3Volumenf > factorPicoVolumen * mediaSma20Volumenf;
					boolean hayPicoEnVolumen7 = max7Volumenf > factorPicoVolumen * mediaSma20Volumenf;
					boolean hayPicoEnPrecio3 = max3Preciof > factorPicoPrecio * mediaSma20Preciof;
					boolean hayPicoEnPrecio7 = max7Preciof > factorPicoPrecio * mediaSma20Preciof;

					if (hayPicoEnVolumen3 && hayPicoEnPrecio3) {
						pathEmpresasTipo44.add(ficheroGestionado.getAbsolutePath());
					}
					if (hayPicoEnVolumen7 && hayPicoEnPrecio7) {
						pathEmpresasTipo45.add(ficheroGestionado.getAbsolutePath());
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

				// ---------------------------------------------------------------------------------------
			}
		}

		// Almacenamiento del tipo de empresa en la lista
		empresasPorTipo = new HashMap<Integer, ArrayList<String>>();
		// Para el subgrupo 0 siempre se añade
		// NO QUITAR, PARA QUE LAS GRÁFICAS FINALES SE PINTEN BASADAS EN ESTE GRUPO,
		// QUE CONTIENE TODOS
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

		empresasPorTipo.put(32, pathEmpresasTipo32);
		empresasPorTipo.put(33, pathEmpresasTipo33);
		empresasPorTipo.put(34, pathEmpresasTipo34);
		empresasPorTipo.put(35, pathEmpresasTipo35);
		empresasPorTipo.put(36, pathEmpresasTipo36);

		empresasPorTipo.put(37, pathEmpresasTipo37);
		empresasPorTipo.put(38, pathEmpresasTipo38);
		empresasPorTipo.put(39, pathEmpresasTipo39);
		empresasPorTipo.put(40, pathEmpresasTipo40);

		empresasPorTipo.put(41, pathEmpresasTipo41);
		empresasPorTipo.put(42, pathEmpresasTipo42);

		empresasPorTipo.put(43, pathEmpresasTipo43);
		empresasPorTipo.put(44, pathEmpresasTipo44);
		empresasPorTipo.put(45, pathEmpresasTipo45);

		empresasPorTipo.put(46, pathEmpresasTipo46);
		empresasPorTipo.put(47, pathEmpresasTipo47);
		empresasPorTipo.put(48, pathEmpresasTipo48);
		empresasPorTipo.put(49, pathEmpresasTipo49);

		// Se crea un CSV para cada subgrupo
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

		while (numSubgruposCreados > 0 && itTipos.hasNext()) {

			tipo = itTipos.next();
			MY_LOGGER.info("***** Subgrupo " + tipo + " *****");

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
					MY_LOGGER.info("Cluster " + tipo + " tiene un " + Math.round(coberturaEmpresasPorCluster * 100)
							+ "% de empresas (" + estadisticas.getSum() + " de " + estadisticas.getValues().length
							+ " ; mínimo = " + coberturaMinima + " %) con al menos una vela positiva." + " Y tiene "
							+ empresasConTarget.keySet().size() + " empresas."
							+ " Pero es FUTURO, así que SIEMPRE SE GENERA DATASET.");
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
				MY_LOGGER.debug("CSV de subgrupo: " + ficheroOut);
				MY_LOGGER.debug("Lista de empresas de subgrupo: " + ficheroListadoOut);
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

							rowTratada = procesarFilaQuitandoColumnasEstaticasYAlgunasColumnasEspeciales(row, tipo); // IMPORTANTE

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

}
