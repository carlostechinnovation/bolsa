/**
 * 
 */
package c70x.validacion;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.helpers.NullEnumeration;

import c10X.brutos.BrutosUtils;
import coordinador.Principal;

/**
 * Genera informe HTML del estado actual de DATOS en la carpeta de tiempo
 * indicada (pasado/futuro).
 *
 */
public class GeneradorInformeHtml implements Serializable {

	private static final long serialVersionUID = 1L;

	public static String DIR_PASADO = "/bolsa/pasado/";
	public static String DIR_FUTURO = "/bolsa/futuro/";
	public static String HTML_TEMPLATE = "src/main/resources/webvacia.html";

	private static GeneradorInformeHtml instancia = null;

	private GeneradorInformeHtml() {
		super();
	}

	public static GeneradorInformeHtml getInstance() {
		if (instancia == null)
			instancia = new GeneradorInformeHtml();

		return instancia;
	}

	static Logger MY_LOGGER = Logger.getLogger(GeneradorInformeHtml.class);

	public static void main(String[] args) throws IOException {

		Object appendersAcumulados = Logger.getRootLogger().getAllAppenders();
		if (appendersAcumulados instanceof NullEnumeration) {
			MY_LOGGER.addAppender(new ConsoleAppender(new PatternLayout(Principal.LOG_PATRON)));
		}
		MY_LOGGER.setLevel(Level.INFO);
		MY_LOGGER.info("INICIO");

		String dirTiempo = DIR_PASADO; // DEFAULT

		if (args.length == 0) {
			MY_LOGGER.info("Sin parametros de entrada. Rellenamos los DEFAULT...");
		} else if (args.length != 1) {
			MY_LOGGER.error("Parametros de entrada incorrectos!!");
			System.exit(-1);
		} else {
			dirTiempo = args[0];
			MY_LOGGER.info("dirTiempo=" + dirTiempo);
		}

		generarInformeHtml(dirTiempo);

		MY_LOGGER.info("FIN");
	}

	/**
	 * Analiza la carpeta pasada como parámetro, genera un informe HTML y lo guarda.
	 * 
	 * @param dirTiempo
	 * @throws IOException
	 */
	public static void generarInformeHtml(String dirTiempo) throws IOException {

		File htmlTemplateFile = new File(HTML_TEMPLATE);
		String htmlString = FileUtils.readFileToString(htmlTemplateFile, "UTF-8");

		String title = "Informe de ficheros la carpeta temporal";
		String body = generarCuerpoHtml(dirTiempo);

		htmlString = htmlString.replace("$title", title);
		htmlString = htmlString.replace("$body", body);
		String pathSalida = dirTiempo + "informe_datos" + ".html";
		MY_LOGGER.info("Escribiendo: " + pathSalida);
		File newHtmlFile = new File(pathSalida);
		FileUtils.writeStringToFile(newHtmlFile, htmlString, "UTF-8", false);
	}

	/**
	 * @param dirTiempo
	 * @return
	 * @throws IOException
	 */
	public static String generarCuerpoHtml(String dirTiempo) throws IOException {

		String out = "<h2>CARPETA ANALIZADA: " + dirTiempo + "</h2>";

		String fechaAhora = LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"));
		out += "FECHA: " + fechaAhora;

		out += "<h3>1. BRUTOS</h3>";
		out += generarCuerpoBrutosHtml(dirTiempo);

		out += "<h3>2. BRUTOS_CSV</h3>";
		out += generarCuerpoBrutosCsvHtml(dirTiempo);

		out += "<h3>3. LIMPIOS</h3>";
		out += generarCuerpoLimpiosHtml(dirTiempo);

		out += "<h3>4. ELABORADOS</h3>";
		out += generarCuerpoElaboradosHtml(dirTiempo);

		out += "<h3>5. SUBGRUPOS</h3>";
		out += generarCuerpoSubgruposHtml(dirTiempo);

		return out;

	}

	/**
	 * @param dirTiempo
	 * @return
	 * @throws IOException
	 */
	public static String generarCuerpoBrutosHtml(String dirTiempo) throws IOException {

		StringBuilder outSB = new StringBuilder();

		List<String> brutosFinviz = Files.walk(Paths.get(dirTiempo + "brutos")).map(x -> x.toString())
				.filter(f -> f.contains("/" + BrutosUtils.FINVIZ_ESTATICOS) && f.endsWith(".html"))
				.collect(Collectors.toList());
		outSB.append("Ficheros de FINVIZ (datos estaticos): " + brutosFinviz.size() + "<br>");
//		brutosFinviz.forEach(outSB::append);

		List<String> brutosFinvizInsiders = Files.walk(Paths.get(dirTiempo + "brutos_csv_OLD")).map(x -> x.toString())
				.filter(f -> f.contains("/" + BrutosUtils.FINVIZ_INSIDERS) && f.endsWith(".csv"))
				.collect(Collectors.toList());
		outSB.append("Ficheros de FINVIZ (datos dinamicos de operaciones de insiders): " + brutosFinvizInsiders.size()
				+ "<br>");
//		brutosFinvizInsiders.forEach(outSB::append);

		List<String> brutosYF = Files.walk(Paths.get(dirTiempo + "brutos")).map(x -> x.toString())
				.filter(f -> f.contains("/" + BrutosUtils.YAHOOFINANCE)).collect(Collectors.toList());
		outSB.append("Ficheros de YAHOO FINANCE (datos dinamicos): " + brutosYF.size() + "<br>");
//		brutosYF.forEach(outSB::append);

		return outSB.toString();
	}

	/**
	 * @param dirTiempo
	 * @return
	 * @throws IOException
	 */
	public static String generarCuerpoBrutosCsvHtml(String dirTiempo) throws IOException {

		StringBuilder outSB = new StringBuilder();

		List<String> csvFZ = Files.walk(Paths.get(dirTiempo + "brutos_csv")).map(x -> x.toString())
				.filter(f -> f.endsWith(".csv")).collect(Collectors.toList());
		outSB.append("Ficheros CSV: " + csvFZ.size() + "<br>");

		return outSB.toString();
	}

	/**
	 * @param dirTiempo
	 * @return
	 * @throws IOException
	 */
	public static String generarCuerpoLimpiosHtml(String dirTiempo) throws IOException {
		StringBuilder outSB = new StringBuilder();
		List<String> lista = Files.walk(Paths.get(dirTiempo + "limpios")).map(x -> x.toString())
				.filter(f -> f.endsWith(".csv")).collect(Collectors.toList());
		outSB.append("Ficheros CSV: " + lista.size() + "<br>");
		return outSB.toString();
	}

	/**
	 * @param dirTiempo
	 * @return
	 * @throws IOException
	 */
	public static String generarCuerpoElaboradosHtml(String dirTiempo) throws IOException {
		StringBuilder outSB = new StringBuilder();
		List<String> lista = Files.walk(Paths.get(dirTiempo + "elaborados")).map(x -> x.toString())
				.filter(f -> f.endsWith(".csv")).collect(Collectors.toList());
		outSB.append("Ficheros CSV: " + lista.size() + "<br>");
		return outSB.toString();
	}

	/**
	 * @param dirTiempo
	 * @return
	 * @throws IOException
	 */
	public static String generarCuerpoSubgruposHtml(String dirTiempo) throws IOException {

		StringBuilder outSB = new StringBuilder();
		boolean esFuturo = dirTiempo.contains("uturo");

		outSB.append("<table><tr><th>Subgrupo</th><th>COMPLETO.csv</th><th>REDUCIDO.csv</th>");
		outSB.append("<th>Ratio de reducción (reducido/completo)</th>");
		outSB.append(esFuturo ? "<th>TARGETS_PREDICHOS.csv_humano</th>" : "");
		outSB.append("</tr>");

		List<String> carpetasSubgrupos = Files.walk(Paths.get(dirTiempo + "subgrupos")).map(x -> x.toString())
				.filter(f -> f.contains("/SG_") && StringUtils.countMatches(f, "/") == 4).collect(Collectors.toList());

		List<String> carpetasSubgruposOrdenadas = ordenarCarpetasSubgrupos(carpetasSubgrupos);

		for (String carpetaSubgrupo : carpetasSubgruposOrdenadas) {

			String tamCompletoStr = analizarCsv(carpetaSubgrupo + "/COMPLETO.csv");
			Long numFilasCompleto = tamCompletoStr.contains("x") ? Long.valueOf(tamCompletoStr.split("x")[0].trim())
					: 0L;
			String tamReducidoStr = analizarCsv(carpetaSubgrupo + "/REDUCIDO.csv");
			Long numFilasReducido = tamReducidoStr.contains("x") ? Long.valueOf(tamReducidoStr.split("x")[0].trim())
					: 0L;
			Float ratioReduccion = 100.0F - (100.0F * numFilasReducido / numFilasCompleto);
			String ratioReduccionStr = numFilasReducido > 0 ? String.format("%.0f", ratioReduccion) : "-";

			String ratioReduccionColor = "white";
			if (numFilasReducido > 0 && ratioReduccion >= 40 && ratioReduccion < 70) {
				ratioReduccionColor = "yellow";
			} else if (numFilasReducido > 0 && ratioReduccion >= 70) {
				ratioReduccionColor = "orange";
			} else {
				ratioReduccionColor = "white";
			}

			outSB.append("<tr>");

			outSB.append("<td>" + carpetaSubgrupo.split("/")[4] + "</td>");
			outSB.append("<td>" + tamCompletoStr + "</td>");
			outSB.append("<td>" + tamReducidoStr + "</td>");
			outSB.append("<td bgcolor=\"" + ratioReduccionColor + "\">" + ratioReduccionStr + " %" + "</td>");
			outSB.append(esFuturo ? ("<td>" + analizarCsv(carpetaSubgrupo + "/TARGETS_PREDICHOS.csv_humano") + "</td>")
					: "");

			outSB.append("</tr>");
		}

		outSB.append("</table>");

		return outSB.toString();
	}

	/**
	 * @param in
	 * @return
	 */
	public static List<String> ordenarCarpetasSubgrupos(List<String> in) {

		// Extraer los numeros de subgrupo
		List<Integer> sg = new ArrayList<Integer>();
		for (String inItem : in) {
			sg.add(Integer.valueOf(inItem.split("/")[4].replace("SG_", "")));

		}

		// Primera parte del path
		String primeraPartePath = in.get(0).substring(0, in.get(0).lastIndexOf("/")) + "/SG_";
		// Ordenar los numeros:
		Collections.sort(sg);

		List<String> out = new ArrayList<String>();
		for (Integer numero : sg) {
			out.add(primeraPartePath + numero.toString());
		}

		return out;

	}

	/**
	 * @param pathCsv
	 * @return
	 * @throws IOException
	 */
	public static String analizarCsv(String pathCsv) throws IOException {

		String out = "";

		MY_LOGGER.debug("Analizando: " + pathCsv);

		File f = new File(pathCsv);
		if (pathCsv != null && !pathCsv.isEmpty() && f.exists() && f.length() > 0) {
			long numLineas = Files.lines(Paths.get(pathCsv)).count();

			Object o1 = Files.lines(Paths.get(pathCsv)).map(s -> s.split(","));
			if (o1 != null) {
				Optional<String[]> o2 = Files.lines(Paths.get(pathCsv)).map(s -> s.split(",")).findFirst();
				if (o2 != null && o2.get() != null) {

					String[] primeraLinea = o2.get();
					String cabecera = primeraLinea[0];
					String[] campos = cabecera.split("\\|");

					out = numLineas + " x " + campos.length;
				}
			}

		} else {
			out += " --";
		}

		return out;
	}

}