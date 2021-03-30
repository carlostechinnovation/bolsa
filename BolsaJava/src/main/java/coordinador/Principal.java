package coordinador;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c10X.brutos.EstaticosFinvizDescargarYParsear;
import c10X.brutos.EstaticosNasdaqDescargarYParsear;
import c10X.brutos.JuntarEstaticosYDinamicosCSVunico;
import c10X.brutos.LimpiarCSVBrutosTemporales;
import c10X.brutos.YahooFinance01Descargar;
import c10X.brutos.YahooFinance02Parsear;
import c20X.limpios.LimpiarOperaciones;
import c30x.elaborados.construir.ConstructorElaborados;
import c40X.subgrupos.CrearDatasetsSubgrupos;
import c40X.subgrupos.CrearDatasetsSubgruposKMeans;
import c70x.validacion.GeneradorInformeHtml;
import c70x.validacion.Validador;
import testIntegracion.ExtractorFeatures;
import testIntegracion.ParserCsvEnTablaHtml;

/**
 * Clase PRINCIPAL
 */
public class Principal implements Serializable {

	public final static String LOG_PATRON = "%d{ISO8601} %c [%t]%x %p - %m%n";

	static Logger MY_LOGGER = Logger.getLogger(Principal.class);

	/**
	 * Punto de entrada a este JAR
	 * 
	 * @param args Parametros de entrada
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		Object appendersAcumulados = Logger.getRootLogger().getAllAppenders();
		if (appendersAcumulados instanceof NullEnumeration) {
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(Principal.LOG_PATRON)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

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
				EstaticosNasdaqDescargarYParsear.getInstance();
				EstaticosNasdaqDescargarYParsear.main(params);
			} else if (programa.equals("c10X.brutos.YahooFinance01Descargar")) {
				YahooFinance01Descargar.getInstance();
				YahooFinance01Descargar.main(params);
			} else if (programa.equals("c10X.brutos.YahooFinance02Parsear")) {
				YahooFinance02Parsear.getInstance();
				YahooFinance02Parsear.main(params);
			} else if (programa.equals("c10X.brutos.EstaticosFinvizDescargarYParsear")) {
				EstaticosFinvizDescargarYParsear.getInstance();
				EstaticosFinvizDescargarYParsear.main(params);
			} else if (programa.equals("c10X.brutos.JuntarEstaticosYDinamicosCSVunico")) {
				JuntarEstaticosYDinamicosCSVunico.getInstance();
				JuntarEstaticosYDinamicosCSVunico.main(params);
			} else if (programa.equals("c10X.brutos.LimpiarCSVBrutosTemporales")) {
				LimpiarCSVBrutosTemporales.getInstance();
				LimpiarCSVBrutosTemporales.main(params);
			} else if (programa.equals("c30X.elaborados.LimpiarOperaciones")) {
				LimpiarOperaciones.getInstance();
				LimpiarOperaciones.main(params);
			} else if (programa.equals("c30X.elaborados.ConstructorElaborados")) {
				ConstructorElaborados.getInstance();
				ConstructorElaborados.main(params);
			} else if (programa.equals("c40X.subgrupos.CrearDatasetsSubgrupos")) {
				CrearDatasetsSubgrupos.getInstance();
				CrearDatasetsSubgrupos.main(params);
			} else if (programa.equals("c40X.subgrupos.CrearDatasetsSubgruposKMeans")) {
				CrearDatasetsSubgruposKMeans.getInstance();
				CrearDatasetsSubgruposKMeans.main(params);
			} else if (programa.equals("c70X.validacion.Validador")) {
				Validador.getInstance();
				Validador.main(params);
			} else if (programa.equals("c70X.validacion.GeneradorInformeHtml")) {
				GeneradorInformeHtml.getInstance();
				GeneradorInformeHtml.main(params);
			} else if (programa.equals("testIntegracion.ExtractorFeatures")) {
				ExtractorFeatures.getInstance();
				ExtractorFeatures.main(params);
			} else if (programa.equals("testIntegracion.ParserCsvEnTablaHtml")) {
				ParserCsvEnTablaHtml.getInstance();
				testIntegracion.ParserCsvEnTablaHtml.main(params);
			} else {
				MY_LOGGER.error("PROGRAMA NO ESPERADO: " + programa);
				System.exit(-1);
			}

		}

	}

}
