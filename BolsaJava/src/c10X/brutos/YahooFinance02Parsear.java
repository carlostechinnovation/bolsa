package c10X.brutos;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

/**
 * Descarga de pruebas de una página de Yahoo Finance
 *
 */
/**
 * @author casa
 *
 */
public class YahooFinance02Parsear {

	static Logger MY_LOGGER = Logger.getLogger(YahooFinance02Parsear.class);

	public YahooFinance02Parsear() {
		super();
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		MY_LOGGER.info("INICIO");

		BasicConfigurator.configure();
		MY_LOGGER.setLevel(Level.INFO);

		// DEFAULT
		String pathBruto = "C:\\bolsa\\pasado\\brutos\\bruto_NASDAQ_CGIX.txt";
		String pathBrutoCsv = "C:\\bolsa\\pasado\\brutos_csv\\bruto_NASDAQ_CGIX.csv";

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 2) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			pathBruto = args[0];
			pathBrutoCsv = args[1];
		}

		MY_LOGGER.info("pathBruto=" + pathBruto);
		MY_LOGGER.info("pathBrutoCsv=" + pathBrutoCsv);

		YahooFinance02Parsear instancia = new YahooFinance02Parsear();
		Boolean out = instancia.parsearJson(pathBruto, pathBrutoCsv);

		MY_LOGGER.info("Resultado: " + out);
		MY_LOGGER.info("FIN");
	}

	/**
	 * Lee un fichero bruto de datos, los extrae y los escribe en un CSV
	 * (estructurados)
	 * 
	 * @param pathBrutoEntrada
	 * @param pathBrutoCsvSalida
	 */
	public Boolean parsearJson(String pathBrutoEntrada, String pathBrutoCsvSalida) {

		MY_LOGGER.info("Parseando JSON...");
		Boolean out = false;

		JSONParser parser = new JSONParser();

		try (Reader reader = new FileReader(pathBrutoEntrada)) {

			JSONObject primerJson = (JSONObject) parser.parse(reader);

			Map<String, JSONObject> mapaChart = (HashMap<String, JSONObject>) primerJson.get("chart");
			Object resultValor = mapaChart.get("result");
			JSONArray a1 = (JSONArray) resultValor;
			JSONObject a2 = (JSONObject) a1.get(0);
			Set<String> claves = a2.keySet();
			for (String clave : claves) {
				System.out.println(clave);
			}

			Object meta = a2.get("meta");
			Object indicators = a2.get("indicators");
			Object tiempos = a2.get("timestamp");

			System.out.println("meta ---> " + meta);
			System.out.println("indicators ---> " + indicators);
			System.out.println("tiempos ---> " + tiempos);

//			// loop array
//			JSONArray msg = (JSONArray) jsonObject.get("messages");
//			Iterator<String> iterator = msg.iterator();
//			while (iterator.hasNext()) {
//				System.out.println(iterator.next());
//			}

			out = true;

		} catch (IOException e) {
			System.err.println(e.getMessage());
		} catch (ParseException e) {
			System.err.println(e.getMessage());
		}

		return out;
	}

}
