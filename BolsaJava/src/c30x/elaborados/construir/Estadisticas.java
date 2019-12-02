package c30x.elaborados.construir;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class Estadisticas extends DescriptiveStatistics {

	private static final long serialVersionUID = 1L;

	/** Como si fuera la derivada, pero s�lo con el primer y �ltimo valores */
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
	 * Ratio, en porcentaje, entre el PRIMER dato y el M�XIMO del conjunto de datos.
	 * Puede tener valores negativos.
	 */
	public int getRatioMax() {
		return (int) Math.ceil(100 * (this.getElement(0) / this.getMax()));
	}

	/**
	 * Ratio, en porcentaje, entre el PRIMER dato y el M�NIMO del conjunto de datos.
	 * Puede tener valores negativos.
	 */
	public int getRatioMin() {
		return (int) Math.ceil(100 * (this.getElement(0) / this.getMin()));
	}

	/**
	 * Se imprime al log una validaci�n de los c�lculos de esta funci�n.
	 */
	public void debugValidacion(final Integer periodo) throws Exception {
		// VALIDACI�N DE ENTRADA
		// Si no se tienen todos los datos del periodo (por ejemplo, para una media de
		// 200 d�as, 200*7 valores hacia atr�s), lanzar� excepci�n
		System.out.println("********************************************************");
		if (getN() != periodo) {
			throw new Exception("El n�mero de datos a analizar no es el adecuado. Se usan " + getN()
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
