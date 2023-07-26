package c10X.brutos;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

/**
 * Modelo de las noticias de una empresa que aparecen en FINVIZ. Son datos
 * DINAMICOS (noticias en un dia concreto).
 */
public class FinvizNoticiasEmpresa {

	public String mercado, empresa;
	public Map<Integer, List<String>> mapa;

	public FinvizNoticiasEmpresa(String mercado, String empresa) {
		super();
		this.mercado = mercado;
		this.empresa = empresa;
		this.mapa = new HashMap<Integer, List<String>>();
	}

	/**
	 * @param mercado
	 * @param empresa
	 * @param noticias     Noticias de una empresa
	 * @param rutaCsvBruto
	 * @throws IOException
	 */
	public void volcarDatosNoticiasEnCSV(String mercado, String empresa, String rutaCsvBruto, Logger MY_LOGGER)
			throws IOException {

		MY_LOGGER.debug(
				"volcarDatosInsidersEnCSV --> " + mercado + "|" + empresa + "|" + mapa.size() + "|" + rutaCsvBruto);

		// ---------------------------- ESCRITURA ---------------
		if (mapa != null && !mapa.isEmpty()) {
			File fout = new File(rutaCsvBruto);
			FileOutputStream fos = new FileOutputStream(fout, false);
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));

			// Se escribe la cabecera
			String CABECERA = "fecha|noticias";
			bw.write(CABECERA);
			bw.newLine();

			// Una fila por cada día en el que haya noticias
			for (Integer amd : mapa.keySet()) {
				List<String> noticiasDeUnDia = mapa.get(amd);

				if (noticiasDeUnDia != null && !noticiasDeUnDia.isEmpty()) {
					String acumulado = "";
					for (String noticia : noticiasDeUnDia) {
						acumulado += noticia + " ";
					}

					String fila = amd + "|" + acumulado;

					bw.write(fila);
					bw.newLine();
				}
			}

			bw.close();

		} else {
			MY_LOGGER.debug("No escribimos fichero FINVIZ_NOTICIAS de empresa=" + empresa
					+ " porque no se han extraido datos de NOTICIAS. Es normal, seguimos.");
		}

	}

}
