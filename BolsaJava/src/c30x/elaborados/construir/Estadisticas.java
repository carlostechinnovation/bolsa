package c30x.elaborados.construir;

import java.util.HashMap;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class Estadisticas extends DescriptiveStatistics {

	private static final long serialVersionUID = 1L;

	public enum COMIENZO_NOMBRES_PARAMETROS_ELABORADOS {
		media_zzz_sma, std_zzz_sma, pendiente_zzz_sma, ratio_zzz_sma, ratio_zzz_maxrelativo, ratio_zzz_minrelativo,
		curtosis_zzz, skewness_zzz;
	}

	public final static String VALOR_INVALIDO = "NULL";

	/** Como si fuera la derivada, pero sólo con el primer y último valores */
	public double getPendiente() {
		return (this.getElement((int) (this.getN() - 1)) - this.getElement(0)) / this.getN();
	}

	/**
	 * Ratio, en porcentaje, entre el PRIMER dato y la MEDIA del conjunto de datos.
	 * Puede tener valores negativos.
	 */
	public int getRatioSMA() {
		return (int) Math.ceil(100 * (this.getElement(0) / this.getMean()));
	}

	/**
	 * Ratio, en porcentaje, entre el PRIMER dato y el MÁXIMO del conjunto de datos.
	 * Puede tener valores negativos.
	 */
	public int getRatioMax() {
		return (int) Math.ceil(100 * (this.getElement(0) / this.getMax()));
	}

	/**
	 * Ratio, en porcentaje, entre el PRIMER dato y el MÍNIMO del conjunto de datos.
	 * Puede tener valores negativos.
	 */
	public int getRatioMin() {
		return (int) Math.ceil(100 * (this.getElement(0) / this.getMin()));
	}

	/**
	 * Se imprime al log una validación de los cálculos de esta función.
	 */
	public void debugValidacion(final Integer periodo) throws Exception {
		// VALIDACIÓN DE ENTRADA
		// Si no se tienen todos los datos del periodo (por ejemplo, para una media de
		// 200 días, 200*7 valores hacia atrás), lanzará excepción
		System.out.println("********************************************************");
		if (getN() != periodo) {
			throw new Exception("El número de datos a analizar no es el adecuado. Se usan " + getN()
					+ " y se necesitan " + periodo);
		} else {
			System.out.println("Se tienen " + getN() + " y se usan " + periodo);
			for (int i = 0; i < getN(); i++) {
				System.out.print(getElement(i) + ", ");
			}
			System.out.println("");
		}

		System.out.println("media_sma = " + getMean());
		System.out.println("std_sma = " + getStandardDeviation());
		System.out.println("pendiente_sma = " + getPendiente());
		System.out.println("ratio_SMA = " + getRatioSMA());
		System.out.println("ratio_maxrelativo = " + getRatioMax());
		System.out.println("ratio_minrelativo = " + getRatioMin());
		System.out.println("curtosis = " + getKurtosis());
		System.out.println("skewness = " + getSkewness());
	}

	/**
	 * 
	 * @param periodo
	 * @param rellenarConInvalidos
	 * @return
	 */
	public HashMap<String, String> getParametros(final Integer periodo, final Boolean rellenarConInvalidos) {
		String periodoString = periodo.toString();
		HashMap<String, String> parametros = new HashMap<String, String>();
		if (rellenarConInvalidos) {
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.media_zzz_sma + periodoString, VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.std_zzz_sma + periodoString, VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.pendiente_zzz_sma + periodoString, VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.ratio_zzz_sma + periodoString, VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.ratio_zzz_sma + periodoString, VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.ratio_zzz_maxrelativo + periodoString,
					VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.ratio_zzz_minrelativo + periodoString,
					VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.curtosis_zzz + periodoString, VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.skewness_zzz + periodoString, VALOR_INVALIDO);
		} else {
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.media_zzz_sma + periodoString,
					String.valueOf(this.getMean()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.std_zzz_sma + periodoString,
					String.valueOf(this.getStandardDeviation()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.pendiente_zzz_sma + periodoString,
					String.valueOf(this.getPendiente()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.ratio_zzz_sma + periodoString,
					String.valueOf(this.getRatioSMA()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.ratio_zzz_sma + periodoString,
					String.valueOf(this.getRatioSMA()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.ratio_zzz_maxrelativo + periodoString,
					String.valueOf(this.getMax()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.ratio_zzz_minrelativo + periodoString,
					String.valueOf(this.getMin()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.curtosis_zzz + periodoString,
					String.valueOf(this.getKurtosis()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.skewness_zzz + periodoString,
					String.valueOf(this.getSkewness()));
		}
		return parametros;
	}

}
