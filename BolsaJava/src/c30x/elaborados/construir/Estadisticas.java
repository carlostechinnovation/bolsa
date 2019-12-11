package c30x.elaborados.construir;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.HashMap;
import java.util.Locale;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

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
	 * 
	 * @param periodo
	 * @param rellenarConInvalidos
	 * @return
	 */
	public HashMap<String, String> getParametros(final Integer periodo, final String finalNombreParametro,
			final Boolean rellenarConInvalidos) {
		String periodoString = periodo.toString();
		HashMap<String, String> parametros = new HashMap<String, String>();
		if (rellenarConInvalidos) {
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_ + periodoString + finalNombreParametro,
					VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.STD_SMA_ + periodoString + finalNombreParametro,
					VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_SMA_ + periodoString + finalNombreParametro,
					VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_SMA_ + periodoString + finalNombreParametro,
					VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_SMA_ + periodoString + finalNombreParametro,
					VALOR_INVALIDO);
			parametros.put(
					COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MAXRELATIVO_ + periodoString + finalNombreParametro,
					VALOR_INVALIDO);
			parametros.put(
					COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MINRELATIVO_ + periodoString + finalNombreParametro,
					VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.CURTOSIS_ + periodoString + finalNombreParametro,
					VALOR_INVALIDO);
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.SKEWNESS_ + periodoString + finalNombreParametro,
					VALOR_INVALIDO);
		} else {

			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_ + periodoString + finalNombreParametro,
					df.format(this.getMean()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.STD_SMA_ + periodoString + finalNombreParametro,
					df.format(this.getStandardDeviation()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_SMA_ + periodoString + finalNombreParametro,
					df.format(this.getPendiente()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_SMA_ + periodoString + finalNombreParametro,
					df.format(this.getRatioSMA()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_SMA_ + periodoString + finalNombreParametro,
					df.format(this.getRatioSMA()));
			parametros.put(
					COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MAXRELATIVO_ + periodoString + finalNombreParametro,
					df.format(this.getMax()));
			parametros.put(
					COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MINRELATIVO_ + periodoString + finalNombreParametro,
					df.format(this.getMin()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.CURTOSIS_ + periodoString + finalNombreParametro,
					df.format(this.getKurtosis()));
			parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.SKEWNESS_ + periodoString + finalNombreParametro,
					df.format(this.getSkewness()));
		}
		return parametros;
	}

	/**
	 * @return the ordenNombresParametrosElaborados
	 */
	public HashMap<Integer, String> getOrdenNombresParametrosElaborados() {
		return ordenNombresParametrosElaborados;
	}

}
