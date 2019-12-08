package c20X.limpios;

import java.io.IOException;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import c10X.brutos.BrutosUtils;

public class LimpiarOperaciones {

	static Logger MY_LOGGER = Logger.getLogger(LimpiarOperaciones.class);

	public LimpiarOperaciones() {
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

		String directorioIn = BrutosUtils.DIR_BRUTOS_CSV; // DEFAULT
		String directorioOut = LimpiosUtils.DIR_LIMPIOS; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 2) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			directorioIn = args[0];
			directorioOut = args[1];
		}

		operacionesLimpieza(directorioIn, directorioOut);
		MY_LOGGER.info("FIN");
	}

	/**
	 * @param directorioIn
	 * @param directorioOut
	 */
	public static void operacionesLimpieza(String directorioIn, String directorioOut) {

		// TODO Quitar anomalias (outliers)

		// TODO Valores vacios (missing values): borramos esa fila o imputamos valores

		// TODO Temporalmente copio los limpios en elaborados

	}

}
