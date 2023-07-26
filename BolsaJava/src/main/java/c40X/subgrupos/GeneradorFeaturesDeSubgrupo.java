/**
 * 
 */
package c40X.subgrupos;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import coordinador.Principal;

/**
 * A partir de una lista de ficheros CSV de un subgrupo, añade a TODAS esas
 * empresas unas FEATURES DEL SUBGRUPO.
 *
 */
public class GeneradorFeaturesDeSubgrupo implements Serializable {

	private static final long serialVersionUID = 1L;

	static Logger MY_LOGGER = Logger.getLogger(CrearDatasetsSubgrupos.class);

	public static void main(String[] args) throws IOException {

		Object appendersAcumulados = Logger.getRootLogger().getAllAppenders();
		if (appendersAcumulados instanceof NullEnumeration) {
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(Principal.LOG_PATRON)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

		// DEFAULT Sólo sirve para debuguear
		List<String> pathsFicherosEmpresasDeUnSubgrupo = new ArrayList<String>();
		pathsFicherosEmpresasDeUnSubgrupo.add("/bolsa/pasado/elaborados/NASDAQ_AESE.csv");
		pathsFicherosEmpresasDeUnSubgrupo.add("/bolsa/pasado/elaborados/NASDAQ_AEY.csv");

		anhadirColumnasDependientesDelSubgrupo(pathsFicherosEmpresasDeUnSubgrupo, MY_LOGGER);
	}

	/**
	 * Calcula las estadisticas de todas las empresas de un subgrupo (PENDIENTE_3D,
	 * PENDIENTE_7D de los precios HIGH y LOW). Estas columnas se AÑADEN a los CSV
	 * de todas las empresas de ese subgrupo (lista de entrada), porque se supone
	 * que arrastrarán a la empresa antes o despues hacia la tendencia del subgrupo.
	 * 
	 * @param pathFicherosEmpresas
	 * @throws IOException
	 */
	public static void anhadirColumnasDependientesDelSubgrupo(List<String> pathsFicherosEmpresasDeUnSubgrupo,
			Logger MY_LOGGER) throws IOException {

		String row = "";
		Integer posicionAnio = null, posicionMes = null, posicionDia = null;
		Integer posicionHigh = null, posicionLow = null;

		Map<String, Map<String, AuxiliarEmpresaAMDpendientes>> mapaEmpresaYsusFeaturesCalculadas = new HashMap<String, Map<String, AuxiliarEmpresaAMDpendientes>>(
				1);

		// Todos los ANIOMESDIA que aparecen en todas las empresas
		// Set<String> todosLosAnioMesDia = new HashSet<String>(); // sin duplicados

		for (String pathFichero : pathsFicherosEmpresasDeUnSubgrupo) {

			// System.out.println("FEATURES de subgrupo - Empresa: " + pathFichero); //
			// MY_LOGGER.info("FEATURES de subgrupo - Empresa: " + pathFichero);

			BufferedReader csvReader = new BufferedReader(new FileReader(pathFichero));
			int numFila = 0;
			Map<Integer, AuxiliarEmpresaAMDprecios> datosUtilesEmpresa = new HashMap<Integer, AuxiliarEmpresaAMDprecios>(
					1);
			int ordenInsercion = 0;

			while ((row = csvReader.readLine()) != null) {

				numFila++;
				if (numFila == 1) {// cabecera

					// Se extrae la posicion de las columnas HIGH y LOW en el CSV
					List<String> partes = Arrays.asList(row.split("\\|"));
					posicionAnio = partes.indexOf("anio");
					posicionMes = partes.indexOf("mes");
					posicionDia = partes.indexOf("dia");
					posicionHigh = partes.indexOf("high");
					posicionLow = partes.indexOf("low");

				} else if (numFila > 1 && posicionHigh != null && posicionLow != null) {// saltamos la cabecera

					// Se extraen los precios HIGH y LOW. Los metemos en un nuevo array, pequeño.
					List<String> partes = Arrays.asList(row.split("\\|"));
					if (partes.get(posicionHigh) != null && !partes.get(posicionHigh).isEmpty()
							&& partes.get(posicionLow) != null && !partes.get(posicionLow).isEmpty()) {

						ordenInsercion++; // SE SUPONE QUE LAS VELAS DE UNA EMPRESA VIENEN ORDENADAS EN ORDEN
											// CRONOLOGICO

						Integer anio = Integer.valueOf(partes.get(posicionAnio));
						Integer mes = Integer.valueOf(partes.get(posicionMes));
						Integer dia = Integer.valueOf(partes.get(posicionDia));

						datosUtilesEmpresa.put(

								ordenInsercion,

								new AuxiliarEmpresaAMDprecios(ordenInsercion, anio, mes, dia,
										Float.valueOf(partes.get(posicionHigh)),
										Float.valueOf(partes.get(posicionLow))));

						// todosLosAnioMesDia.add(anio + "-" + mes + "-" + dia);

					}
				}
			}

			// ESTADISTICA DE CADA EMPRESA: PENDIENTE_3D,PENDIENTE_7D de HIGH y LOW
			// Para cada anio-mes-dia, obtiene la pendiente (primera derivada) de HIGH y LOW
			// mirando X dias hacia atras (si los hay).
			Map<String, AuxiliarEmpresaAMDpendientes> featuresCalculadasEmpresa = calcularFeaturesEmpresa(
					datosUtilesEmpresa);
			mapaEmpresaYsusFeaturesCalculadas.put(pathFichero, featuresCalculadasEmpresa);

			// liberar recursos
			csvReader.close();
		}

		// Para cada AMD, acumular todas las features
		Map<String, List<AuxiliarEmpresaAMDpendientes>> mapaAMDFeaturesTodasLasEmpresas = new HashMap<String, List<AuxiliarEmpresaAMDpendientes>>();

		for (String empresa : mapaEmpresaYsusFeaturesCalculadas.keySet()) {
			Map<String, AuxiliarEmpresaAMDpendientes> amdFeaturesUnaEmpresa = mapaEmpresaYsusFeaturesCalculadas
					.get(empresa);

			if (amdFeaturesUnaEmpresa != null) {

				for (String amd : amdFeaturesUnaEmpresa.keySet()) {
					if (mapaAMDFeaturesTodasLasEmpresas.containsKey(amd) == false) {
						mapaAMDFeaturesTodasLasEmpresas.put(amd, new ArrayList<AuxiliarEmpresaAMDpendientes>(1));
					}

					mapaAMDFeaturesTodasLasEmpresas.get(amd).add(amdFeaturesUnaEmpresa.get(amd));
				}
			}
		}

		// ESTADISTICAS DE SUBGRUPO: para cada AMD, calcular las FEATURES MEDIAS
		Map<String, List<Float>> highPendiente3Dtodas = new HashMap<String, List<Float>>(1);
		Map<String, List<Float>> highPendiente7Dtodas = new HashMap<String, List<Float>>(1);
		Map<String, List<Float>> lowPendiente20Dtodas = new HashMap<String, List<Float>>(1);
		Map<String, List<Float>> lowPendiente7Dtodas = new HashMap<String, List<Float>>(1);
		Float numAniadidos1 = 0F, numAniadidos2 = 0F, numAniadidos3 = 0F, numAniadidos4 = 0F;
		for (String amd : mapaAMDFeaturesTodasLasEmpresas.keySet()) {

			List<AuxiliarEmpresaAMDpendientes> featuresDeEmpresas = mapaAMDFeaturesTodasLasEmpresas.get(amd);

			highPendiente3Dtodas.put(amd, new ArrayList<Float>(1));
			highPendiente7Dtodas.put(amd, new ArrayList<Float>(1));
			lowPendiente20Dtodas.put(amd, new ArrayList<Float>(1));
			lowPendiente7Dtodas.put(amd, new ArrayList<Float>(1));

			for (AuxiliarEmpresaAMDpendientes featuresUnaEmpresa : featuresDeEmpresas) {
				if (featuresUnaEmpresa.highPendiente3D != null) {
					highPendiente3Dtodas.get(amd).add(featuresUnaEmpresa.highPendiente3D);
					numAniadidos1++;
				}
				if (featuresUnaEmpresa.highPendiente7D != null) {
					highPendiente7Dtodas.get(amd).add(featuresUnaEmpresa.highPendiente7D);
					numAniadidos2++;
				}
				if (featuresUnaEmpresa.lowPendiente20D != null) {
					lowPendiente20Dtodas.get(amd).add(featuresUnaEmpresa.lowPendiente20D);
					numAniadidos3++;
				}
				if (featuresUnaEmpresa.lowPendiente7D != null) {
					lowPendiente7Dtodas.get(amd).add(featuresUnaEmpresa.lowPendiente7D);
					numAniadidos4++;
				}

			}
		}

		Map<String, AuxiliarEmpresaAMDpendientes> mapaAMDFeaturesMediasSubgrupo = new HashMap<String, AuxiliarEmpresaAMDpendientes>(
				1);
		for (String amd : mapaAMDFeaturesTodasLasEmpresas.keySet()) {
			String[] amdStr = amd.split("-");

			mapaAMDFeaturesMediasSubgrupo.put(amd,
					new AuxiliarEmpresaAMDpendientes(Integer.valueOf(amdStr[0]), Integer.valueOf(amdStr[1]),
							Integer.valueOf(amdStr[2]),
							sumarItemsEnLista(highPendiente3Dtodas.get(amd)) / numAniadidos1,
							sumarItemsEnLista(highPendiente7Dtodas.get(amd)) / numAniadidos2,
							sumarItemsEnLista(lowPendiente20Dtodas.get(amd)) / numAniadidos3,
							sumarItemsEnLista(lowPendiente7Dtodas.get(amd)) / numAniadidos4));
		}

		// Meter las estadisticas del subgrupo a cada empresa de este subgrupo y
		// GUARDAR, AMPLIANDO cada fichero existente.
		guardarAmpliandoFicheros(pathsFicherosEmpresasDeUnSubgrupo, mapaAMDFeaturesMediasSubgrupo);
	}

	/**
	 * Calcula las FEATURES de una EMPRESA que se usarán para calcular las FEATURES
	 * DE SUBGRUPO (viendo todas las empresas de un mismo subgrupo).
	 * 
	 * @param datosUtilesEmpresa Mapa con estos elementos [ordenInsercion,
	 *                           datosempresaDeEseDia]
	 * @return Mapa [añomesdia, featuresCalculadasDeEseDia]
	 */
	public static Map<String, AuxiliarEmpresaAMDpendientes> calcularFeaturesEmpresa(
			Map<Integer, AuxiliarEmpresaAMDprecios> datosUtilesEmpresa) {

		Collection<AuxiliarEmpresaAMDprecios> precios = datosUtilesEmpresa.values();

		Map<String, AuxiliarEmpresaAMDpendientes> salida = new HashMap<String, AuxiliarEmpresaAMDpendientes>(0);

		for (AuxiliarEmpresaAMDprecios precio : precios) {

			if (precio.ordenCreacion != null) {
				String claveAMD = precio.anio + "-" + String.format("%02d", precio.mes) + "-"
						+ String.format("%02d", precio.dia);

				Integer ordenHace3D = precio.ordenCreacion + 3;
				Integer ordenHace7D = precio.ordenCreacion + 7;
				Integer ordenHace20D = precio.ordenCreacion + 20;

				AuxiliarEmpresaAMDprecios precioHace3D = ordenHace3D != null ? datosUtilesEmpresa.get(ordenHace3D)
						: null;
				AuxiliarEmpresaAMDprecios precioHace7D = ordenHace7D != null ? datosUtilesEmpresa.get(ordenHace7D)
						: null;
				AuxiliarEmpresaAMDprecios precioHace20D = ordenHace20D != null ? datosUtilesEmpresa.get(ordenHace20D)
						: null;

				Float highPendiente3D = precioHace3D != null
						? 100 * (precio.high - precioHace3D.high) / precioHace3D.high
						: null;
				Float highPendiente7D = precioHace7D != null
						? 100 * (precio.high - precioHace7D.high) / precioHace7D.high
						: null;
				Float lowPendiente20D = precioHace20D != null
						? 100 * (precio.low - precioHace20D.low) / precioHace20D.low
						: null;
				Float lowPendiente7D = precioHace7D != null ? 100 * (precio.low - precioHace7D.low) / precioHace7D.low
						: null;

				salida.put(claveAMD, new AuxiliarEmpresaAMDpendientes(precio.anio, precio.mes, precio.dia,
						highPendiente3D, highPendiente7D, lowPendiente20D, lowPendiente7D));
			}
		}

		return salida;
	}

	/**
	 * @param lista
	 * @return
	 */
	public static Float sumarItemsEnLista(List<Float> lista) {

		Float out = 0.0F;
		if (lista != null && !lista.isEmpty()) {
			for (Float item : lista) {
				out += item;
			}
		}

		return out;
	}

	/**
	 * SOBREESCRIBIR los ficheros existentes, ampliandolos metiendo las features de
	 * subgrupo: TODAS las empresas del subgrupo tendrán las mismas columnas
	 * añadidas.
	 * 
	 * Si esos ficheros ya tuvieran FEATURES de un SUBGRUPO anterior, se
	 * sobreescriben.
	 * 
	 * @param pathsFicherosEmpresasDeUnSubgrupo
	 * @param mapaAMDFeaturesMediasSubgrupo
	 * @throws IOException
	 */
	public static void guardarAmpliandoFicheros(List<String> pathsFicherosEmpresasDeUnSubgrupo,
			Map<String, AuxiliarEmpresaAMDpendientes> mapaAMDFeaturesMediasSubgrupo) throws IOException {

		// Cabecera FEATURES DE SUBGRUPO

		List<List<String>> filasCsvLeidas = null;

		for (String pathCsv : pathsFicherosEmpresasDeUnSubgrupo) {

			filasCsvLeidas = new ArrayList(1);

			// LECTURA
			BufferedReader csvReader = new BufferedReader(new FileReader(pathCsv));
			String linea = null;
			boolean cabecera = true;
			int posicionAnio = -1;
			int posicionMes = -1;
			int posicionDia = -1;
			int posAux1 = -1, posAux2 = -1, posAux3 = -1, posAux4 = -1;
			int posicionTarget = -1; // SOLO APARECE EN PASADO!!

			while ((linea = csvReader.readLine()) != null) {
				String[] data = linea.split("\\|");
				List<String> filaLista = Arrays.asList(data);
				if (cabecera) {
					posicionAnio = filaLista.indexOf("anio");
					posicionMes = filaLista.indexOf("mes");
					posicionDia = filaLista.indexOf("dia");

					posAux1 = filaLista.indexOf(AuxiliarEmpresaAMDpendientes.SG_HIGH_PENDIENTE3D);
					posAux2 = filaLista.indexOf(AuxiliarEmpresaAMDpendientes.SG_HIGH_PENDIENTE7D);
					posAux3 = filaLista.indexOf(AuxiliarEmpresaAMDpendientes.SG_LOW_PENDIENTE20D);
					posAux4 = filaLista.indexOf(AuxiliarEmpresaAMDpendientes.SG_LOW_PENDIENTE7D);

					posicionTarget = filaLista.indexOf("TARGET");
					filasCsvLeidas.add(filaLista);

				} else {
					filasCsvLeidas.add(filaLista);
				}
				cabecera = false;
			}
			csvReader.close();

			// Añadir FEATURES DE SUBGRUPO, mirando el AMD
			List<String> filasCsvSalida = new ArrayList<String>(1);
			cabecera = true;
			List<String> listaAux = new ArrayList<String>(1);
			for (List<String> filaLista : filasCsvLeidas) {

				String target = null;
				target = posicionTarget != -1 ? filaLista.get(filaLista.size() - 1) : null; // Ultimo elemento es el
																							// target (SOLO PASADO)

				listaAux = new ArrayList<String>(1);
				listaAux.addAll(filaLista);

				if (posicionTarget != -1) {
					listaAux.remove(posicionTarget); // Borramos este primero, porque es el mas lejano
				}

				if (posAux1 != -1 && posAux2 != -1 && posAux3 != -1 && posAux4 != -1 && listaAux.contains(posAux1)
						&& listaAux.contains(posAux2) && listaAux.contains(posAux3) && listaAux.contains(posAux4)) {
					// FEATURES DE SUBGRUPO preexistentes (si la empresa apareció en otro subgrupo
					// antes)
					listaAux.remove(posAux4);
					listaAux.remove(posAux3);
					listaAux.remove(posAux2);
					listaAux.remove(posAux1);
				}

				if (cabecera) {
					String nuevaCabecera = listaAcadena(listaAux, "|")
							+ AuxiliarEmpresaAMDpendientes.getCabeceraSinAMD();
					nuevaCabecera += target != null ? ("|" + target) : "";
					filasCsvSalida.add(nuevaCabecera);

				} else {
					String claveAMD = filaLista.get(posicionAnio) + "-" + filaLista.get(posicionMes) + "-"
							+ filaLista.get(posicionDia);

					AuxiliarEmpresaAMDpendientes featuresSubgrupoDeAMD = mapaAMDFeaturesMediasSubgrupo.get(claveAMD);

					String featuresSubgrupo = featuresSubgrupoDeAMD != null
							? featuresSubgrupoDeAMD.getDatosParaCSVSinAMD()
							: AuxiliarEmpresaAMDpendientes.getDatosParaCSVSinAMDnulo();

					String nuevaFilaDatos = listaAcadena(listaAux, "|") + featuresSubgrupo;
					nuevaFilaDatos += target != null ? ("|" + target) : "";
					filasCsvSalida.add(nuevaFilaDatos);
				}
				cabecera = false;
			}

			/// ESCRITURA
			// System.out.println("Sobreescribiendo CSV de elaborados, con las FEATURES DE
			/// SUBGRUPO: " + pathCsv); // DEBUG
			File antiguoCsv = new File(pathCsv);
			antiguoCsv.delete();
			File nuevoCsv = new File(pathCsv);
			FileWriter escritor = new FileWriter(nuevoCsv, false);
			for (String cadena : filasCsvSalida) {
				escritor.write(cadena + "\n");
			}
			escritor.flush();
			escritor.close();

		}

	}

	/**
	 * Convierte una lista en cadena con separadores.
	 * 
	 * @param lista Lista de Strings
	 * @param sep   Separador
	 * @return
	 */
	public static String listaAcadena(List<String> lista, String sep) {
		String out = "";
		if (lista != null && !lista.isEmpty() && sep != null && !sep.isEmpty()) {
			for (String item : lista) {
				out += (!out.isEmpty()) ? sep : "";
				out += item;
			}
		}
		return out;
	}

}
