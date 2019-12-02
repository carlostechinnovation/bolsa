package c30x.elaborados.construir;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class Estadisticas extends DescriptiveStatistics {

	private static final long serialVersionUID = 1L;

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

}
