package c40X.subgrupos;

import java.io.Serializable;

public class FalsosPositivosSubgrupo implements Serializable {

	private static final long serialVersionUID = 1L;
	public static final String POSIBLE_PREFIJO = "SG_";

	public Integer subgrupoId = null;
	public String subgrupo;
	public Long numvelasfp;
	public Long numeroPredicciones;
	public Float ratioFalsosPositivos;

	public FalsosPositivosSubgrupo(String subgrupo, String numvelasfp, String numeroPredicciones,
			String ratioFalsosPositivos) {
		super();
		this.subgrupo = subgrupo;
		this.numvelasfp = Long.valueOf(numvelasfp);
		this.numeroPredicciones = Long.valueOf(numeroPredicciones);
		this.ratioFalsosPositivos = Float.valueOf(ratioFalsosPositivos);

		if (subgrupo != null && !subgrupo.isEmpty()) {
			if (subgrupo.contains(POSIBLE_PREFIJO)) {
				subgrupoId = Integer.valueOf(subgrupo.replace(POSIBLE_PREFIJO, ""));
			} else {
				subgrupoId = Integer.valueOf(subgrupo.replace(POSIBLE_PREFIJO, ""));
			}
		}
	}

	@Override
	public String toString() {
		return "FalsosPositivosSubgrupo [subgrupoId=" + subgrupoId + ", subgrupo=" + subgrupo + ", numvelasfp="
				+ numvelasfp + ", numeroPredicciones=" + numeroPredicciones + ", ratioFalsosPositivos="
				+ ratioFalsosPositivos + "]";
	}

}
