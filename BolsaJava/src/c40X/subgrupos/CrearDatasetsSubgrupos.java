package c40X.subgrupos;

import java.io.IOException;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import c30x.elaborados.construir.ElaboradosUtils;

/**
 * Crea los datasets (CSV) de cada subgrupo
 *
 */
public class CrearDatasetsSubgrupos {

	static Logger MY_LOGGER = Logger.getLogger(CrearDatasetsSubgrupos.class);

	public CrearDatasetsSubgrupos() {
		super();
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {

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

		definirSubgruposPorFiltrosEstaticos();

		// TODO Para cada subgrupo:
		crearDatasetDeCadaSubgrupo(directorioIn, directorioOut);
		normalizarDatasetDeSubgrupo(directorioOut);

		MY_LOGGER.info("FIN");
	}

	/**
	 * Cada subgrupo está definido por filtros sobre las variables ESTATICAS
	 */
	public static void definirSubgruposPorFiltrosEstaticos() {

		// TODO pendiente

	}

	/**
	 * Crea un CSV para cada subgrupo
	 * 
	 * @param directorioIn
	 * @param directorioOut
	 */
	public static void crearDatasetDeCadaSubgrupo(String directorioIn, String directorioOut) {

		// TODO pendiente

	}

	/**
	 * Normaliza los datasets de cada subgrupo
	 * 
	 * @param directorioOut
	 */
	public static void normalizarDatasetDeSubgrupo(String directorioOut) {

		// TODO pendiente

	}

}
