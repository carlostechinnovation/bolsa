package c30x.elaborados.construir;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.Locale;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.ResizableDoubleArray;

/**
 * @author carloslinux
 *
 */
public class Estadisticas extends DescriptiveStatistics {

	private static final long serialVersionUID = 1L;

	public final static String VALOR_INVALIDO = "null";
	public final static int VALOR_FAKE = 0;
	public final static double NUM100 = 100.0D;
	public final static double NUM1K = 1000.0D;
	public final static double NUM1M = 1000000.0D;

	// Si anhado mas parametros, debo modificar la constructora
	private HashMap<Integer, String> ordenNombresParametrosElaborados;

	// Otros menos útiles:
//	PENDIENTE_SMA_SEGUNDO_, PENDIENTE_1M_SMA_SEGUNDO_, PENDIENTE_2M_SMA_SEGUNDO_
//	RATIO_MAXRELATIVO_SEGUNDO_, RATIO_SMA_SEGUNDO_, STD_SMA_, PENDIENTE_1M_SMA_, 
	public enum COMIENZO_NOMBRES_PARAMETROS_ELABORADOS {
		MEDIA_SMA_, PENDIENTE_SMA_, PENDIENTE_2M_SMA_, RATIO_SMA_, RATIO_MAXRELATIVO_, RATIO_MINRELATIVO_,
		RATIO_MINRELATIVO_SEGUNDO_, RATIO_U_SMA_, RATIO_U_MAXRELATIVO_, RATIO_U_MINRELATIVO_, CURTOSIS_, SKEWNESS_;
	}

//	Otros menos útiles: 
//  _CLOSEHIGH, _CLOSELOW, _OPENLOW,
	public enum FINAL_NOMBRES_PARAMETROS_ELABORADOS {
		_VOLUMEN, _CLOSE, _HIGH, _LOW, _OPEN, _CLOSEOPEN, _OPENHIGH, _HIGHLOW;
	}

	public enum OTROS_PARAMS_ELAB {
		DIAS_HASTA_FIN_MES, DIAS_HASTA_FIN_TRIMESTRE;
	}

	static Locale locale;
	static DecimalFormat df;

	public static void main(String[] args) {

		Estadisticas e5 = new Estadisticas();
		e5.addValue(1D);
		e5.addValue(10D);
		e5.addValue(NUM100);
		e5.addValue(NUM1K);
		System.out.println("Valores: " + e5.toString());
		System.out.println("e5.getPendienteRelativa(): " + e5.getPendienteRelativa());
//		System.out.println("e5.getPendienteRelativaSegundo(): " + e5.getPendienteRelativaSegundo());
		System.out.println("e5.getPendienteRelativa1M(): " + e5.getPendienteRelativa1M());
//		System.out.println("e5.getPendienteRelativa1MSegundo(): " + e5.getPendienteRelativa1MSegundo());
		System.out.println("e5.getPendienteRelativa2M(): " + e5.getPendienteRelativa2M());
//		System.out.println("e5.getPendienteRelativa2MSegundo(): " + e5.getPendienteRelativa2MSegundo());
		System.out.println("e5.getRatioSMA(): " + e5.getRatioSMA());
		System.out.println("e5.getRatioSMASegundo(): " + e5.getRatioSMASegundo());
		System.out.println("e5.getRatioMax(): " + e5.getRatioMax());
		System.out.println("e5.getRatioMin(): " + e5.getRatioMin());
		System.out.println("e5.getRatioMaxSegundo(): " + e5.getRatioMaxSegundo());
		System.out.println("e5.getRatioMinSegundo(): " + e5.getRatioMinSegundo());
		System.out.println("e5.getRatioUltimoSMA(): " + e5.getRatioUltimoSMA());
		System.out.println("e5.getRatioUltimoMax(): " + e5.getRatioUltimoMax());
		System.out.println("e5.getRatioUltimoMin(): " + e5.getRatioUltimoMin());
		System.out.println("e5.getKurtosis(): " + e5.getKurtosis());
		System.out.println("e5.getSkewness(): " + e5.getSkewness());

	}

	/**
	 * Se muestran los valores separados por ;
	 */
	@Override
	public String toString() {
		String salida = "";
		for (int i = 0; i < this.getN() - 1; i++)
			salida += this.getElement(i) + "; ";
		salida += this.getElement((int) this.getN() - 1);
		return salida;
	}

	/**
	 * Constructora
	 */
	public Estadisticas() {
		ordenNombresParametrosElaborados = new HashMap<Integer, String>();
		// otros nó útiles:
//		ordenNombresParametrosElaborados.put(4, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_SMA_SEGUNDO_.toString());
//		ordenNombresParametrosElaborados.put(6, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_1M_SMA_SEGUNDO_.toString());
//		ordenNombresParametrosElaborados.put(8, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_2M_SMA_SEGUNDO_.toString());
//		ordenNombresParametrosElaborados.put(2, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.STD_SMA_.toString());
//		ordenNombresParametrosElaborados.put(4, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_1M_SMA_.toString());
//		ordenNombresParametrosElaborados.put(7, COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_SMA_SEGUNDO_.toString());
//		ordenNombresParametrosElaborados.put(10,
//				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MAXRELATIVO_SEGUNDO_.toString());

		ordenNombresParametrosElaborados.put(ordenNombresParametrosElaborados.size() + 1,
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_.toString());
		ordenNombresParametrosElaborados.put(ordenNombresParametrosElaborados.size() + 1,
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_SMA_.toString());
		ordenNombresParametrosElaborados.put(ordenNombresParametrosElaborados.size() + 1,
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_2M_SMA_.toString());
		ordenNombresParametrosElaborados.put(ordenNombresParametrosElaborados.size() + 1,
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_SMA_.toString());
		ordenNombresParametrosElaborados.put(ordenNombresParametrosElaborados.size() + 1,
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MAXRELATIVO_.toString());
		ordenNombresParametrosElaborados.put(ordenNombresParametrosElaborados.size() + 1,
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MINRELATIVO_.toString());
		ordenNombresParametrosElaborados.put(ordenNombresParametrosElaborados.size() + 1,
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MINRELATIVO_SEGUNDO_.toString());
		ordenNombresParametrosElaborados.put(ordenNombresParametrosElaborados.size() + 1,
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_U_SMA_.toString());
		ordenNombresParametrosElaborados.put(ordenNombresParametrosElaborados.size() + 1,
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_U_MAXRELATIVO_.toString());
		ordenNombresParametrosElaborados.put(ordenNombresParametrosElaborados.size() + 1,
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_U_MINRELATIVO_.toString());
		ordenNombresParametrosElaborados.put(ordenNombresParametrosElaborados.size() + 1,
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.CURTOSIS_.toString());
		ordenNombresParametrosElaborados.put(ordenNombresParametrosElaborados.size() + 1,
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.SKEWNESS_.toString());

		locale = new Locale("en", "UK");
		df = (DecimalFormat) NumberFormat.getNumberInstance(locale);
		df.applyPattern("#0.#");
	}

	/**
	 * Media.
	 */
	public double getMedia() {
		return this.getMean();
	}

	/**
	 * Desviación estándar.
	 */
	public double getStd() {
		return this.getStandardDeviation();
	}

	/**
	 * En porcentaje. Como si fuera la derivada, pero solo con el primer y ultimo
	 * valores
	 */
	public double getPendienteRelativa() {
		double salida = NUM100 * (this.getElement((int) (this.getN() - 1)) - this.getElement(0))
				/ (this.getN() * this.getElement(0));
		// Para evitar infinitos, asumimos estos valores como infinito
		if (salida > NUM1M) {
			salida = NUM1M;
		} else if (salida < (-1.0) * NUM1M) {
			salida = (-1.0) * NUM1M;
		}
		return salida;
	}

	/**
	 * En porcentaje. Como si fuera la derivada, pero solo con el SEGUNDO y ultimo
	 * valores
	 */
	public double getPendienteRelativaSegundo() {
		double salida = NUM100 * (this.getElement((int) (this.getN() - 1)) - this.getElement(1))
				/ (this.getN() * this.getElement(1));
		// Para evitar infinitos, asumimos estos valores como infinito
		if (salida > NUM1M) {
			salida = NUM1M;
		} else if (salida < (-1.0) * NUM1M) {
			salida = (-1.0) * NUM1M;
		}
		return salida;
	}

	/**
	 * En porcentaje. Como si fuera la derivada, de la primera mitad de valores,
	 * pero solo con el primer y ultimo valores.
	 */
	public double getPendienteRelativa1M() {
		double salida = NUM100 * (this.getElement((int) (Math.ceil(this.getN() / 2.0 - 1))) - this.getElement(0))
				/ ((Math.ceil(this.getN() / 2.0 - 1)) * this.getElement(0));
		// Para evitar infinitos, asumimos estos valores como infinito
		if (salida > NUM1M) {
			salida = NUM1M;
		} else if (salida < (-1.0) * NUM1M) {
			salida = (-1.0) * NUM1M;
		}
		return salida;
	}

	/**
	 * En porcentaje. Como si fuera la derivada, de la primera mitad de valores,
	 * pero solo con el SEGUNDO y ultimo valores.
	 */
	public double getPendienteRelativa1MSegundo() {
		double salida = NUM100 * (this.getElement((int) (Math.ceil(this.getN() / 2.0 - 1))) - this.getElement(1))
				/ ((Math.ceil(this.getN() / 2.0 - 1)) * this.getElement(1));
		// Para evitar infinitos, asumimos estos valores como infinito
		if (salida > NUM1M) {
			salida = NUM1M;
		} else if (salida < (-1.0) * NUM1M) {
			salida = (-1.0) * NUM1M;
		}
		return salida;
	}

	/**
	 * En porcentaje. Como si fuera la derivada, de la segunda mitad de valores,
	 * pero solo con el primer y ultimo valores.
	 */
	public double getPendienteRelativa2M() {
		double salida = NUM100
				* (this.getElement((int) (this.getN() - 1)) - this.getElement((int) ((int) this.getN() / 2.0)))
				/ (Math.ceil(this.getN() / 2.0 - 1)) * this.getElement((int) ((int) this.getN() / 2.0));
		// Para evitar infinitos, asumimos estos valores como infinito
		if (salida > NUM1M) {
			salida = NUM1M;
		} else if (salida < (-1.0) * NUM1M) {
			salida = (-1.0) * NUM1M;
		}
		return salida;
	}

	/**
	 * En porcentaje. Como si fuera la derivada, de la segunda mitad de valores,
	 * pero solo con el SEGUNDO y ultimo valores.
	 */
	public double getPendienteRelativa2MSegundo() {
		double salida = NUM100
				* (this.getElement((int) (this.getN() - 1)) - this.getElement((int) ((int) this.getN() / 2.0 - 1)))
				/ (Math.ceil(this.getN() / 2.0 - 1)) * this.getElement((int) ((int) this.getN() / 2.0 - 1));
		// Para evitar infinitos, asumimos estos valores como infinito
		if (salida > NUM1M) {
			salida = NUM1M;
		} else if (salida < (-1.0) * NUM1M) {
			salida = (-1.0) * NUM1M;
		}
		return salida;
	}

	/**
	 * Ratio, en porcentaje, entre el PRIMER dato y la MEDIA del conjunto de datos.
	 * Puede tener valores negativos.
	 */
	public int getRatioSMA() {
		return (int) Math.round(NUM100 * (this.getElement(0) / getMean()));
	}

	/**
	 * Ratio, en porcentaje, entre el SEGUNDO dato y la MEDIA del conjunto de datos.
	 * Puede tener valores negativos.
	 */
	public int getRatioSMASegundo() {
		int salida = VALOR_FAKE;
		if (this.getN() > 1)
			salida = (int) Math.round(NUM100 * (this.getElement(1) / getMean()));
		return salida;
	}

	/**
	 * Ratio, en porcentaje, entre el PRIMER dato y el MAXIMO del conjunto de datos.
	 * Puede tener valores negativos.
	 */
	public int getRatioMax() {
		return (int) Math.round(NUM100 * (this.getElement(0) / this.getMax()));
	}

	/**
	 * Ratio, en porcentaje, entre el PRIMER dato y el MINIMO del conjunto de datos.
	 * Puede tener valores negativos.
	 */
	public int getRatioMin() {
		return (int) Math.round(NUM100 * (this.getElement(0) / this.getMin()));
	}

	/**
	 * Ratio, en porcentaje, entre el SEGUNDO dato y el MAXIMO del conjunto de
	 * datos. Puede tener valores negativos.
	 */
	public int getRatioMaxSegundo() {
		int salida = VALOR_FAKE;
		if (this.getN() > 1)
			salida = (int) Math.round(NUM100 * (this.getElement(1) / this.getMax()));
		return salida;
	}

	/**
	 * Ratio, en porcentaje, entre el SEGUNDO dato y el MINIMO del conjunto de
	 * datos. Puede tener valores negativos.
	 */
	public int getRatioMinSegundo() {
		return (int) Math.round(NUM100 * (this.getElement(1) / this.getMin()));
	}

	/**
	 * Ratio, en porcentaje, entre el ULTIMO dato y la MEDIA del conjunto de datos.
	 * Puede tener valores negativos.
	 */
	public int getRatioUltimoSMA() {
		int salida = VALOR_FAKE;
		if (this.getN() > 1)
			salida = (int) Math.round(NUM100 * (this.getElement((int) getN() - 1) / getMean()));
		return salida;
	}

	/**
	 * Ratio, en porcentaje, entre el ULTIMO dato y el MAXIMO del conjunto de datos.
	 * Puede tener valores negativos.
	 */
	public int getRatioUltimoMax() {
		int salida = VALOR_FAKE;
		if (this.getN() > 1)
			salida = (int) Math.round(NUM100 * (this.getElement((int) getN() - 1) / this.getMax()));
		return salida;
	}

	/**
	 * Ratio, en porcentaje, entre el ULTIMO dato y el MINIMO del conjunto de datos.
	 * Puede tener valores negativos.
	 */
	public int getRatioUltimoMin() {
		return (int) Math.round(NUM100 * (this.getElement((int) getN() - 1) / this.getMin()));
	}

	/**
	 * Se imprime al log una validacion de los calculos de esta funcion.
	 */
	public void debugValidacion(final Integer numDatosGestionados) throws Exception {
		// VALIDACION DE ENTRADA
		// Si no se tienen todos los datos del periodo (por ejemplo, para una media de
		// 200 dias, 200*7 valores hacia atras), lanzara excepcion
		System.out.println("********************************************************");
		if (getN() != numDatosGestionados) {
			throw new Exception("El numero de datos a analizar no es el adecuado. Se usan " + getN()
					+ " y se necesitan " + numDatosGestionados);
		} else {
			System.out.println("Se tienen " + getN() + " y se usan " + numDatosGestionados);
			for (int i = 0; i < getN(); i++) {
				System.out.print(getElement(i) + ", ");
			}
			System.out.println("");
		}

		System.out.println("media_sma = " + getMedia());
		System.out.println("std_sma = " + getStd());
		System.out.println("pendiente_sma = " + getPendienteRelativa());
//		System.out.println("pendiente_sma_segundo = " + getPendienteRelativaSegundo());
		System.out.println("pendiente_1m_sma = " + getPendienteRelativa1M());
//		System.out.println("pendiente_1m_sma_segundo = " + getPendienteRelativa1MSegundo());
		System.out.println("pendiente_2m_sma = " + getPendienteRelativa2M());
//		System.out.println("pendiente_2m_sma_segundo = " + getPendienteRelativa2MSegundo());
		System.out.println("ratio_SMA = " + getRatioSMA());
		System.out.println("ratio_SMASegundo = " + getRatioSMASegundo());
		System.out.println("ratio_maxrelativo = " + getRatioMax());
		System.out.println("ratio_minrelativo = " + getRatioMin());
		System.out.println("ratio_maxrelativoSegundo = " + getRatioMaxSegundo());
		System.out.println("ratio_minrelativoSegundo = " + getRatioMinSegundo());
		System.out.println("ratio_UltimoSMA = " + getRatioUltimoSMA());
		System.out.println("ratio_Ultimomaxrelativo = " + getRatioUltimoMax());
		System.out.println("ratio_Ultimominrelativo = " + getRatioUltimoMin());
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
		String pendiente_sma_segundo = VALOR_INVALIDO;// default
		String pendiente_1m_sma = VALOR_INVALIDO;// default
		String pendiente_1m_sma_segundo = VALOR_INVALIDO;// default
		String pendiente_2m_sma = VALOR_INVALIDO;// default
		String pendiente_2m_sma_segundo = VALOR_INVALIDO;// default
		String ratio_sma = VALOR_INVALIDO;// default
		String ratio_smaSegundo = VALOR_INVALIDO;// default
		String ratio_maxrelativo = VALOR_INVALIDO;// default
		String ratio_minrelativo = VALOR_INVALIDO;// default
		String ratio_maxrelativoSegundo = VALOR_INVALIDO;// default
		String ratio_minrelativoSegundo = VALOR_INVALIDO;// default
		String ratio_usma = VALOR_INVALIDO;// default
		String ratio_umaxrelativo = VALOR_INVALIDO;// default
		String ratio_uminrelativo = VALOR_INVALIDO;// default
		String kurtosis = VALOR_INVALIDO;// default
		String skewness = VALOR_INVALIDO;// default

		if (rellenarConInvalidos == false) {

			double d_media_sma = this.getMedia();
			double d_std_sma = this.getStd();
			double d_pendiente_sma = this.getPendienteRelativa();
//			double d_pendiente_sma_segundo = this.getPendienteRelativaSegundo();
			double d_pendiente_1m_sma = this.getPendienteRelativa1M();
//			double d_pendiente_1m_sma_segundo = this.getPendienteRelativa1MSegundo();
			double d_pendiente_2m_sma = this.getPendienteRelativa2M();
//			double d_pendiente_2m_sma_segundo = this.getPendienteRelativa2MSegundo();
			double d_ratio_sma = this.getRatioSMA();
			double d_ratio_smaSegundo = this.getRatioSMASegundo();
			double d_ratio_maxrelativo = this.getRatioMax();
			double d_ratio_minrelativo = this.getRatioMin();
			double d_ratio_maxrelativoSegundo = this.getRatioMaxSegundo();
			double d_ratio_minrelativoSegundo = this.getRatioMinSegundo();
			double d_ratio_usma = this.getRatioUltimoSMA();
			double d_ratio_umaxrelativo = this.getRatioUltimoMax();
			double d_ratio_uminrelativo = this.getRatioUltimoMin();
			double d_kurtosis = this.getKurtosis();
			double d_skewness = this.getSkewness();

			media_sma = Double.isNaN(d_media_sma) ? VALOR_INVALIDO : df.format(d_media_sma);
			std_sma = Double.isNaN(d_std_sma) ? VALOR_INVALIDO : df.format(d_std_sma);
			pendiente_sma = Double.isNaN(d_pendiente_sma) ? VALOR_INVALIDO : df.format(d_pendiente_sma);
//			pendiente_sma_segundo = Double.isNaN(d_pendiente_sma_segundo) ? VALOR_INVALIDO : df.format(d_pendiente_sma_segundo);
			pendiente_1m_sma = Double.isNaN(d_pendiente_1m_sma) ? VALOR_INVALIDO : df.format(d_pendiente_1m_sma);
//			pendiente_1m_sma_segundo = Double.isNaN(d_pendiente_1m_sma_segundo) ? VALOR_INVALIDO : df.format(d_pendiente_1m_sma_segundo);
			pendiente_2m_sma = Double.isNaN(d_pendiente_2m_sma) ? VALOR_INVALIDO : df.format(d_pendiente_2m_sma);
//			pendiente_2m_sma_segundo = Double.isNaN(d_pendiente_2m_sma_segundo) ? VALOR_INVALIDO : df.format(d_pendiente_2m_sma_segundo);
			ratio_sma = Double.isNaN(d_ratio_sma) ? VALOR_INVALIDO : df.format(d_ratio_sma);
			ratio_smaSegundo = Double.isNaN(d_ratio_smaSegundo) ? VALOR_INVALIDO : df.format(d_ratio_smaSegundo);
			ratio_maxrelativo = Double.isNaN(d_ratio_maxrelativo) ? VALOR_INVALIDO : df.format(d_ratio_maxrelativo);
			ratio_minrelativo = Double.isNaN(d_ratio_minrelativo) ? VALOR_INVALIDO : df.format(d_ratio_minrelativo);
			ratio_maxrelativoSegundo = Double.isNaN(d_ratio_maxrelativoSegundo) ? VALOR_INVALIDO
					: df.format(d_ratio_maxrelativoSegundo);
			ratio_minrelativoSegundo = Double.isNaN(d_ratio_minrelativoSegundo) ? VALOR_INVALIDO
					: df.format(d_ratio_minrelativoSegundo);
			ratio_usma = Double.isNaN(d_ratio_usma) ? VALOR_INVALIDO : df.format(d_ratio_usma);
			ratio_umaxrelativo = Double.isNaN(d_ratio_umaxrelativo) ? VALOR_INVALIDO : df.format(d_ratio_umaxrelativo);
			ratio_uminrelativo = Double.isNaN(d_ratio_uminrelativo) ? VALOR_INVALIDO : df.format(d_ratio_uminrelativo);
			// Se pone un 0 si la kurtosis no es válida. Para menos de 4 valores, será
			// siempre inválido. Se pone 0 en vez de null para que no se descarten todas las
			// filas. El clasificador no lo usará, ya que no lo considerará útil
			kurtosis = Double.isNaN(d_kurtosis) ? "0" : df.format(d_kurtosis);
			skewness = Double.isNaN(d_skewness) ? VALOR_INVALIDO : df.format(d_skewness);
		}

		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.MEDIA_SMA_ + periodoString + finalNombreParametro,
				media_sma);
//		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.STD_SMA_ + periodoString + finalNombreParametro, std_sma);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_SMA_ + periodoString + finalNombreParametro,
				pendiente_sma);
//		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_SMA_SEGUNDO_ + periodoString + finalNombreParametro,
//				pendiente_sma_segundo);
//		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_1M_SMA_ + periodoString + finalNombreParametro,
//				pendiente_1m_sma);
//		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_1M_SMA_SEGUNDO_ + periodoString + finalNombreParametro,
//				pendiente_1m_sma_segundo);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_2M_SMA_ + periodoString + finalNombreParametro,
				pendiente_2m_sma);
//		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.PENDIENTE_2M_SMA_SEGUNDO_ + periodoString + finalNombreParametro,
//				pendiente_2m_sma_segundo);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_SMA_ + periodoString + finalNombreParametro,
				ratio_sma);
//		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_SMA_SEGUNDO_ + periodoString + finalNombreParametro,
//				ratio_smaSegundo);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MAXRELATIVO_ + periodoString + finalNombreParametro,
				ratio_maxrelativo);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MINRELATIVO_ + periodoString + finalNombreParametro,
				ratio_minrelativo);
//		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MAXRELATIVO_SEGUNDO_ + periodoString
//				+ finalNombreParametro, ratio_maxrelativoSegundo);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_MINRELATIVO_SEGUNDO_ + periodoString
				+ finalNombreParametro, ratio_minrelativoSegundo);
		parametros.put(COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_U_SMA_ + periodoString + finalNombreParametro,
				ratio_usma);
		parametros.put(
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_U_MAXRELATIVO_ + periodoString + finalNombreParametro,
				ratio_umaxrelativo);
		parametros.put(
				COMIENZO_NOMBRES_PARAMETROS_ELABORADOS.RATIO_U_MINRELATIVO_ + periodoString + finalNombreParametro,
				ratio_uminrelativo);
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
	 *         porcentaje sin signo y_t=ABS(x_i+1 - x_i). Se devuelve:
	 *         ABS(max(y_t)/average(y_t))
	 */
	public Double getVariacionRelativaMaxima() {
		ResizableDoubleArray y = new ResizableDoubleArray();
		for (int i = 0; i < getN() - 1; i++) {
			y.addElement(Math.abs(getElement(i + 1) - getElement(i)));
		}
		Double max = 0D, average = 0D;
		for (int counter = 0; counter < y.getNumElements(); counter++) {
			if (max < y.getElement(counter)) {
				max = y.getElement(counter);
			}
			average += y.getElement(counter);
		}
		Double n = Double.valueOf(y.getNumElements());
		average = average / n;
		return Math.abs(max / average);
	}

	/**
	 * Calcula el ultimo dia del MES
	 * 
	 * @param date
	 * @return
	 */
	public static Calendar calcularUltimoDiaDelMes(Date date) {
		Calendar cal = Calendar.getInstance();
		cal.setTime(date);
		cal.set(Calendar.DAY_OF_MONTH, cal.getActualMaximum(Calendar.DAY_OF_MONTH));
		return cal;
	}

	/**
	 * Calcula el ultimo dia del TRIMESTRE
	 * 
	 * @param date
	 * @return
	 */
	public static Calendar calcularUltimoDiaDelTrimestre(Date date) {
		Calendar cal = Calendar.getInstance();
		cal.setTime(date);
		cal.set(Calendar.DAY_OF_MONTH, 1);
		cal.set(Calendar.MONTH, cal.get(Calendar.MONTH) / 3 * 3 + 2);
		cal.set(Calendar.DAY_OF_MONTH, cal.getActualMaximum(Calendar.DAY_OF_MONTH));
		return cal;
	}

	/**
	 * Calcula el ultimo dia del AÑO
	 * 
	 * @param date
	 * @return
	 */
	public static Calendar calcularUltimoDiaDelAnio(Date date) {
		Calendar cal = Calendar.getInstance();
		cal.setTime(date);
		cal.set(Calendar.DAY_OF_MONTH, 31);
		cal.set(Calendar.MONTH, Calendar.DECEMBER);
		return cal;
	}

	/**
	 * @param currentTime
	 * @param endDateTime
	 * @return
	 */
	public static int restarTiempos(long currentTime, long endDateTime) {

		Calendar endDateCalendar;
		Calendar currentDayCalendar;

		// expiration day
		endDateCalendar = Calendar.getInstance();
		endDateCalendar.setTimeInMillis(endDateTime);
		endDateCalendar.set(Calendar.MILLISECOND, 0);
		endDateCalendar.set(Calendar.MINUTE, 0);
		endDateCalendar.set(Calendar.HOUR, 0);

		// current day
		currentDayCalendar = Calendar.getInstance();
		currentDayCalendar.setTimeInMillis(currentTime);
		currentDayCalendar.set(Calendar.MILLISECOND, 0);
		currentDayCalendar.set(Calendar.MINUTE, 0);
		currentDayCalendar.set(Calendar.HOUR, 0);

		int remainingDays = Math
				.round((float) (endDateCalendar.getTimeInMillis() - currentDayCalendar.getTimeInMillis())
						/ (24 * 60 * 60 * 1000));

		return remainingDays;
	}

}
