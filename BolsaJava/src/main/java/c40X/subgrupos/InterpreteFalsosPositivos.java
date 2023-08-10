package c40X.subgrupos;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import c20X.limpios.LimpiosUtils;

public class InterpreteFalsosPositivos implements Serializable {

	private static final long serialVersionUID = 1L;

	// FALSOS POSITIVOS: por encima de este umbral (%) descartamos subgrupo
	public static final Float UMBRAL_MAX_RATIOSUBGRUPO_FP = 85.0F;
	// FALSOS POSITIVOS: por encima de este umbral (%) descartamos empresa
	public static final Float UMBRAL_MAX_RATIOEMPRESA_FP = 90.0F;

	public static final String PATH_FP_SUBGRUPOS = "realimentacion/falsospositivos_subgrupos.csv";
	public static final String PATH_FP_EMPRESAS = "realimentacion/falsospositivos_empresas.csv";

	private static InterpreteFalsosPositivos instancia = null;

	private InterpreteFalsosPositivos() {
		super();
	}

	public static InterpreteFalsosPositivos getInstance() {
		if (instancia == null)
			instancia = new InterpreteFalsosPositivos();

		return instancia;
	}

	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		InterpreteFalsosPositivos.extraerSubgruposConDemasiadosFP();
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
			System.out.println("Fichero de subgrupos con muchos FALSOS POSITIVOS: " + tempFile.getAbsolutePath());
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
