package c10X.brutos;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
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
		String pathBruto = "C:\\bolsa\\pasado\\brutos\\bruto_NASDAQ_AXAS.txt";
		String pathBrutoCsv = "C:\\bolsa\\pasado\\brutos_csv\\bruto_NASDAQ_AXAS.csv";

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

			// ---------------------------- LECTURA ------------------
			MY_LOGGER.info("Lectura...");
			JSONObject primerJson = (JSONObject) parser.parse(reader);

			Map<String, JSONObject> mapaChart = (HashMap<String, JSONObject>) primerJson.get("chart");
			Object resultValor = mapaChart.get("result");
			JSONArray a1 = (JSONArray) resultValor;
			JSONObject a2 = (JSONObject) a1.get(0);
			Set<String> claves = a2.keySet();
			for (String clave : claves) {
				MY_LOGGER.info("clave=" + clave);
			}

			// Object meta = a2.get("meta");
			JSONObject indicators = (JSONObject) a2.get("indicators");
			// Object tiempos = a2.get("timestamp");

			JSONArray quote1 = (JSONArray) indicators.get("quote");
			JSONObject quote2 = (JSONObject) quote1.get(0);

			JSONArray listaVolumenes = (JSONArray) quote2.get("volume");
			JSONArray listaPreciosHigh = (JSONArray) quote2.get("high");
			JSONArray listaPreciosLow = (JSONArray) quote2.get("low");
			JSONArray listaPreciosClose = (JSONArray) quote2.get("close");
			JSONArray listaPreciosOpen = (JSONArray) quote2.get("open");

			MY_LOGGER.info("Tamanios --> " + listaVolumenes.size() + "|" + listaPreciosHigh.size() + "|"
					+ listaPreciosLow.size() + "|" + listaPreciosClose.size() + "|" + listaPreciosOpen.size());

			// ---------------------------- ESCRITURA ---------------
			MY_LOGGER.info("Escritura...");
			File fout = new File(pathBrutoCsvSalida);
			FileOutputStream fos = new FileOutputStream(fout, false);
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));

			// Cabecera
			bw.write("antiguedad|volumen|high|low|close|open");
			bw.newLine();

			int i = 0;
			int numTodos = listaVolumenes.size();
			int antiguedad = 0;
			for (i = 0; i < listaVolumenes.size(); i++) {

				antiguedad = numTodos - i - 1;
				// MY_LOGGER.info("i=" + i + " antiguedad=" + String.valueOf(antiguedad));
				bw.write(antiguedad + "|" + listaVolumenes.get(i) + "|" + listaPreciosHigh.get(i) + "|"
						+ listaPreciosLow.get(i) + "|" + listaPreciosClose.get(i) + "|" + listaPreciosOpen.get(i));
				bw.newLine();
			}

			bw.close();

			out = true;

		} catch (IOException e) {
			MY_LOGGER.error(e.getMessage());
		} catch (ParseException e) {
			MY_LOGGER.error(e.getMessage());
		}

		return out;
	}

}
