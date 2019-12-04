package c10X.brutos;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
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
 * Parsear datos de Yahoo Finance
 */
public class YahooFinance02Parsear {

	static Logger MY_LOGGER = Logger.getLogger(YahooFinance02Parsear.class);

	public YahooFinance02Parsear() {
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

		String directorioIn = "/bolsa/pasado/brutos/"; // DEFAULT
		String directorioOut = "/bolsa/pasado/brutos_csv/"; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 2) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			directorioIn = args[0];
			directorioOut = args[1];
		}

		List<EstaticoNasdaqModelo> nasdaqEstaticos1 = EstaticosNasdaqDescargarYParsear.descargarNasdaqEstaticos1();
		parsearNasdaqDinamicos01(nasdaqEstaticos1, directorioIn, directorioOut);

		MY_LOGGER.info("FIN");
	}

	/**
	 * @param nasdaqEstaticos1
	 * @param directorioIn
	 * @param directorioOut
	 * @return
	 * @throws IOException
	 */
	public static Boolean parsearNasdaqDinamicos01(List<EstaticoNasdaqModelo> nasdaqEstaticos1, String directorioIn,
			String directorioOut) throws IOException {

		MY_LOGGER.info("parsearNasdaqDinamicos01 --> " + directorioIn + "|" + directorioOut);

		String mercado = "NASDAQ"; // DEFAULT
		Boolean out = false;
		String ticker;

		MY_LOGGER.info("mercado=" + mercado);

		String pathBruto;
		String pathBrutoCsv;

		for (int i = 0; i < nasdaqEstaticos1.size(); i++) {

			ticker = nasdaqEstaticos1.get(i).symbol;
			pathBruto = directorioIn + "bruto_" + mercado + "_" + ticker + ".txt";
			pathBrutoCsv = directorioOut + "bruto_" + mercado + "_" + ticker + ".csv";

			MY_LOGGER.info("pathBruto|pathBrutoCsv =" + pathBruto + "|" + pathBrutoCsv);

			if (Files.exists(Paths.get(pathBruto))) {
				Files.deleteIfExists(Paths.get(pathBrutoCsv)); // Borramos el fichero de salida si existe

				out = parsearJson(pathBruto, pathBrutoCsv);

				if (out.booleanValue() == false) {
					MY_LOGGER.error("La descarga de datos estaticos 1 de " + mercado + " - " + ticker
							+ " ha fallado. Saliendo...");
				}

			} else {
				MY_LOGGER.warn("Fichero de entrada no existe: " + pathBruto + " Continúa...");
			}

		}
		return out;
	}

	/**
	 * Lee un fichero bruto de datos, los extrae y los escribe en un CSV
	 * (estructurados)
	 * 
	 * @param pathBrutoEntrada
	 * @param pathBrutoCsvSalida
	 */
	public static Boolean parsearJson(String pathBrutoEntrada, String pathBrutoCsvSalida) {

		MY_LOGGER.info("parsearJson... --> " + pathBrutoEntrada + "|" + pathBrutoCsvSalida);
		Boolean out = false;
		JSONParser parser = new JSONParser();

		try (Reader reader = new FileReader(pathBrutoEntrada)) {

			// ---------------------------- LECTURA ------------------
			MY_LOGGER.debug("Lectura...");
			JSONObject primerJson = (JSONObject) parser.parse(reader);

			Map<String, JSONObject> mapaChart = (HashMap<String, JSONObject>) primerJson.get("chart");
			Object resultValor = mapaChart.get("result");
			JSONArray a1 = (JSONArray) resultValor;
			JSONObject a2 = (JSONObject) a1.get(0);
			Set<String> claves = a2.keySet();

			Object meta = a2.get("meta");
			JSONObject indicators = (JSONObject) a2.get("indicators");
			JSONArray tiemposEnSegundosDesde1970 = (JSONArray) a2.get("timestamp");

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
			MY_LOGGER.debug("Escritura...");
			File fout = new File(pathBrutoCsvSalida);
			FileOutputStream fos = new FileOutputStream(fout, false);
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));

			// Cabecera
			bw.write("anio|mes|dia|hora|minuto|volumen|high|low|close|open");
			bw.newLine();

			int i = 0;
			DateFormat df = new SimpleDateFormat("yyyy|MM|dd|HH|mm");

			for (i = 0; i < listaVolumenes.size(); i++) {

				long msegDesde1970 = (long) tiemposEnSegundosDesde1970.get(i) * 1000L;

				bw.write(df.format(new Date(msegDesde1970)) + "|" + listaVolumenes.get(i) + "|"
						+ listaPreciosHigh.get(i) + "|" + listaPreciosLow.get(i) + "|" + listaPreciosClose.get(i) + "|"
						+ listaPreciosOpen.get(i));
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
