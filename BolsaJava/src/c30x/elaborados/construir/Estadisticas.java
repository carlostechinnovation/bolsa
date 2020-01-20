package c30x.elaborados.construir;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.HashMap;
import java.util.Locale;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.ResizableDoubleArray;

public class Estadisticas extends DescriptiveStatistics {

	private static final long serialVersionUID = 1L;

	public final static String VALOR_INVALIDO = "null";

	// Si a�ado m�s par�metros, debo modificar la constructora
	private HashMap<Integer, String> ordenNombresParametrosElaborados;

	public enum COMIENZO_NOMBRES_PARAMETROS_ELABORADOS {
		MEDIA_SMA_, STD_SMA_, PENDIENTE_SMA_, RATIO_SMA_, RATIO_MAXRELATIVO_, RATIO_MINRELATIVO_, CURTOSIS_, SKEWNESS_;
	}

	public enum FINAL_NOMBRES_PARAMETROS_ELABORADOS {
		_PRECIO, _VOLUMEN;
	}

	static Locale locale;
	static DecimalFormat df;

	public static void main(String[] args) {

		Estadisticas e1 = new Estadisticas();
		e1.addValue(1D);
		e1.addValue(1D);
		e1.addValue(1D);
		e1.addValue(1D);
		e1.addValue(1D);
		e1.addValue(1D);
		e1.addValue(1D);
		e1.addValue(1D);
		e1.addValue(1D);
		e1.addValue(1D);
		Estadisticas e5 = new Estadisticas();
		e5.addValue(1D);
		e5.addValue(1D);
		e5.addValue(5D);
		e5.addValue(1D);
		e5.addValue(1D);
		e5.addValue(1D);
		e5.addValue(1D);
		e5.addValue(1D);
		e5.addValue(1D);
		e5.addValue(1D);

		System.out.println("e1.getVariacionRelativaMaximaDePendiente(): " + e1.getVariacionRelativaMaximaDePendiente());
		System.out.println("e5.getVariacionRelativaMaximaDePendiente(): " + e5.getVariacionRelativaMaximaDePendiente());

	}

	/**
	 * 
	 */
	public Estadisticas() {
		ordenNombresParametrosElaborados = new HashMap<Integer, String>();
		ordenNombresParametrosElaborados.put(1, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_.toString());
		ordenNombresParametrosElaborados.put(2, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.STD_SMA_.toString());
		ordenNombresParametrosElaborados.put(3, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_SMA_.toString());
		ordenNombresParametrosElaborados.put(4, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_SMA_.toString());
		ordenNombresParametrosElaborados.put(5, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MAXRELATIVO_.toString());
		ordenNombresParametrosElaborados.put(6, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MINRELATIVO_.toString());
		ordenNombresParametrosElaborados.put(7, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.CURTOSIS_.toString());
		ordenNombresParametrosElaborados.put(8, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.SKEWNESS_.toString());

		locale = new Locale("en", "UK");
		df = (DecimalFormat) NumberFormat.getNumberInstance(locale);
		df.applyPattern("#0.#");
	}

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
	public void debugValidacion(final Integer numDatosGestionados) throws Exception {
		// VALIDACI�N DE ENTRADA
		// Si no se tienen todos los datos del periodo (por ejemplo, para una media de
		// 200 d�as, 200*7 valores hacia atr�s), lanzar� excepci�n
		System.out.println("********************************************************");
		if (getN() != numDatosGestionados) {
			throw new Exception("El n�mero de datos a analizar no es el adecuado. Se usan " + getN()
					+ " y se necesitan " + numDatosGestionados);
		} else {
			System.out.println("Se tienen " + getN() + " y se usan " + numDatosGestionados);
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
	 * @param periodo
	 * @param finalNombreParametro
	 * @param rellenarConInvalidos
	 * @return
	 */
	public HashMap<String, String> getParametros(final Integer periodo, final String finalNombreParametro,
			final Boolean rellenarConInvalidos) {

		String periodoString = periodo.toString();
		HashMap<String, String> parametros = new HashMap<String, String>();

		String media_sma = VALOR_INVALIDO;// default
		String std_sma = VALOR_INVALIDO;// default
		String pendiente_sma = VALOR_INVALIDO;// default
		String ratio_sma = VALOR_INVALIDO;// default
		String ratio_maxrelativo = VALOR_INVALIDO;// default
		String ratio_minrelativo = VALOR_INVALIDO;// default
		String kurtosis = VALOR_INVALIDO;// default
		String skewness = VALOR_INVALIDO;// default

		if (rellenarConInvalidos == false) {

			double d_media_sma = this.getMean();
			double d_std_sma = this.getStandardDeviation();
			double d_pendiente_sma = this.getPendiente();
			double d_ratio_sma = this.getRatioSMA();
			double d_ratio_maxrelativo = this.getMax();
			double d_ratio_minrelativo = this.getMin();
			double d_kurtosis = this.getKurtosis();
			double d_skewness = this.getSkewness();

			media_sma = Double.isNaN(d_media_sma) ? VALOR_INVALIDO : df.format(d_media_sma);
			std_sma = Double.isNaN(d_std_sma) ? VALOR_INVALIDO : df.format(d_std_sma);
			pendiente_sma = Double.isNaN(d_pendiente_sma) ? VALOR_INVALIDO : df.format(d_pendiente_sma);
			ratio_sma = Double.isNaN(d_ratio_sma) ? VALOR_INVALIDO : df.format(d_ratio_sma);
			ratio_maxrelativo = Double.isNaN(d_ratio_maxrelativo) ? VALOR_INVALIDO : df.format(d_ratio_maxrelativo);
			ratio_minrelativo = Double.isNaN(d_ratio_minrelativo) ? VALOR_INVALIDO : df.format(d_ratio_minrelativo);
			kurtosis = Double.isNaN(d_kurtosis) ? VALOR_INVALIDO : df.format(d_kurtosis);
			skewness = Double.isNaN(d_skewness) ? VALOR_INVALIDO : df.format(d_skewness);
		}

		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_ + periodoString + finalNombreParametro,
				media_sma);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.STD_SMA_ + periodoString + finalNombreParametro, std_sma);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_SMA_ + periodoString + finalNombreParametro,
				pendiente_sma);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_SMA_ + periodoString + finalNombreParametro,
				ratio_sma);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MAXRELATIVO_ + periodoString + finalNombreParametro,
				ratio_maxrelativo);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MINRELATIVO_ + periodoString + finalNombreParametro,
				ratio_minrelativo);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.CURTOSIS_ + periodoString + finalNombreParametro,
				kurtosis);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.SKEWNESS_ + periodoString + finalNombreParametro,
				skewness);

		return parametros;
	}

	/**
	 * @return the ordenNombresParametrosElaborados
	 */
	public HashMap<Integer, String> getOrdenNombresParametrosElaborados() {
		return ordenNombresParametrosElaborados;
	}

	/**
	 * @return Para un conjunto de valores, se obtienen sus diferencias en
	 *         porcentaje sin signo 
	 *         y_t=ABS((x_i+1 - x_i)/x_i). 
	 *         Se devuelve:
	 *         ABS(max(y_t)/media(y_t))
	 */
	public Double getVariacionRelativaMaximaDePendiente() {
		ResizableDoubleArray y = new ResizableDoubleArray();
		for (int i = 0; i < getN() - 1; i++) {
			y.addElement(Math.abs((getElement(i + 1) - getElement(i)) / getElement(i)));
		}
		Double max = 0D, average = 0D;
		for (int counter = 1; counter < y.getNumElements(); counter++) {
			if (y.getElement(counter) > max) {
				max = y.getElement(counter);
			}
			average += y.getElement(counter);
		}
		average = Double.valueOf(average / y.getNumElements());
		return Math.abs(max / average);
	}

}
