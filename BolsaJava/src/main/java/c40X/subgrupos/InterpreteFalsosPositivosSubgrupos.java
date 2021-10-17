package c40X.subgrupos;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import c20X.limpios.LimpiosUtils;

public class InterpreteFalsosPositivosSubgrupos implements Serializable {

	private static final long serialVersionUID = 1L;

	// En tanto por ciento: por encima de este umbral descartamos subgrupo
	public static final Float UMBRAL_MAX_RATIO_FP = 20.0F;

	public static final String PATH_FP_SUBGRUPOS = "/bolsa/logs/falsospositivos_subgrupos.csv";

	private static InterpreteFalsosPositivosSubgrupos instancia = null;

	private InterpreteFalsosPositivosSubgrupos() {
		super();
	}

	public static InterpreteFalsosPositivosSubgrupos getInstance() {
		if (instancia == null)
			instancia = new InterpreteFalsosPositivosSubgrupos();

		return instancia;
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {

		InterpreteFalsosPositivosSubgrupos instancia = getInstance();
		instancia.extraerSubgruposConDemasiadosFP();

	}

	/**
	 * @return Lista de los subgrupos con demasiados FALSOS POSITIVOS
	 * @throws IOException
	 */
	public static List<FalsosPositivosSubgrupo> extraerSubgruposConDemasiadosFP() throws IOException {

		List<FalsosPositivosSubgrupo> listaOutModelos = new ArrayList<FalsosPositivosSubgrupo>();
		File tempFile = new File(PATH_FP_SUBGRUPOS);
		boolean exists = tempFile.exists();
		if (exists) {

			List<List<String>> falsosPositivosSubgrupos = LimpiosUtils
					.leerFicheroHaciaListasDeColumnas(PATH_FP_SUBGRUPOS, null);

			for (int i = 1; i < falsosPositivosSubgrupos.get(0).size(); i++) { // La primera fila (i=0) es la cabecera

				listaOutModelos.add(new FalsosPositivosSubgrupo(falsosPositivosSubgrupos.get(0).get(i), // subgrupo
						falsosPositivosSubgrupos.get(1).get(i), // numvelasfp
						falsosPositivosSubgrupos.get(2).get(i), // numeroPredicciones
						falsosPositivosSubgrupos.get(3).get(i) // ratioFalsosPositivos
				));
			}

		} else {
			System.out.println("No existe el fichero de subgrupos con muchos FALSOS POSITIVOS: " + PATH_FP_SUBGRUPOS);
		}

		return listaOutModelos;
	}

}
