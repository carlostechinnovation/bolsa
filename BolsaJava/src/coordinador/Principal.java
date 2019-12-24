package coordinador;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import c10X.brutos.EstaticosFinvizDescargarYParsear;
import c10X.brutos.EstaticosNasdaqDescargarYParsear;
import c10X.brutos.JuntarEstaticosYDinamicosCSVunico;
import c10X.brutos.LimpiarCSVBrutosTemporales;
import c10X.brutos.YahooFinance01Descargar;
import c10X.brutos.YahooFinance02Parsear;
import c20X.limpios.LimpiarOperaciones;
import c30x.elaborados.construir.ConstructorElaborados;
import c40X.subgrupos.CrearDatasetsSubgruposKMeans;

/**
 * Clase PRINCIPAL
 */
public class Principal {

	static Logger MY_LOGGER = Logger.getLogger(Principal.class);

	/**
	 * Punto de entrada a este JAR
	 * 
	 * @param args Parametros de entrada
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		MY_LOGGER.info("INICIO");

		BasicConfigurator.configure();
		MY_LOGGER.setLevel(Level.INFO);

		if (args.length < 1) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {

			int numParams = args.length;
			MY_LOGGER.info("Numero de parametros: " + numParams);
			for (String param : args) {
				MY_LOGGER.info("Param: " + param);
			}

			List<String> args2 = new ArrayList<String>();
			String programa = null;
			int i = 0;
			for (String item : args) {
				i++;
				if (i == 1 || i == 2) {
					programa = item;
				} else if (i == 3) {
					programa = item;
				} else {
					args2.add(item);
				}
			}
			String[] args2array = args2.toArray(new String[0]);
			ejecutarProgramaConParams(programa, args2array);
		}

		MY_LOGGER.info("FIN");
	}

	/**
	 * @param programa
	 * @param params
	 * @throws Exception
	 */
	public static void ejecutarProgramaConParams(String programa, String[] params) throws Exception {

		if (programa != null && !programa.isEmpty()) {

			if (programa.equals("c10X.brutos.EstaticosNasdaqDescargarYParsear")) {
				EstaticosNasdaqDescargarYParsear.main(params);
			} else if (programa.equals("c10X.brutos.YahooFinance01Descargar")) {
				YahooFinance01Descargar.main(params);
			} else if (programa.equals("c10X.brutos.YahooFinance02Parsear")) {
				YahooFinance02Parsear.main(params);
			} else if (programa.equals("c10X.brutos.EstaticosFinvizDescargarYParsear")) {
				EstaticosFinvizDescargarYParsear.main(params);
			} else if (programa.equals("c10X.brutos.JuntarEstaticosYDinamicosCSVunico")) {
				JuntarEstaticosYDinamicosCSVunico.main(params);
			} else if (programa.equals("c10X.brutos.LimpiarCSVBrutosTemporales")) {
				LimpiarCSVBrutosTemporales.main(params);
			} else if (programa.equals("c30X.elaborados.LimpiarOperaciones")) {
				LimpiarOperaciones.main(params);
			} else if (programa.equals("c30X.elaborados.ConstructorElaborados")) {
				ConstructorElaborados.main(params);
			} else if (programa.equals("c40X.subgrupos.CrearDatasetsSubgruposKMeans")) {
				CrearDatasetsSubgruposKMeans.main(params);
			} else {
				MY_LOGGER.error("PROGRAMA NO ESPERADO: " + programa);
				System.exit(-1);
			}

		}

	}

}
