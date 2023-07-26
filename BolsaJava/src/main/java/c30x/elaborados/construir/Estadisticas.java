package c30x.elaborados.construir;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.Locale;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.ResizableDoubleArray;
import org.apache.log4j.Logger;

/**
 * Utilidad que calcula estadisticas a partir de un dataset de entrada.
 *
 */
public class Estadisticas extends DescriptiveStatistics {

	private static final long serialVersionUID = 1L;

	public final static String VALOR_INVALIDO = "null";
	public final static int VALOR_FAKE = 0;
	public final static float NUM100 = 100.0F;
	public final static float NUM1K = 1000.0F;
	public final static float NUM1M = 1000000.0F;

	private static final DecimalFormat df4decimales = new DecimalFormat("0.0000");

	static Logger MY_LOGGER = Logger.getLogger(Estadisticas.class);

	// Si anhado mas parametros, debo modificar la constructora
	private HashMap<Integer, String> ordenNombresParametrosElaborados;

	public enum PREFIJOS_ELAB {
		STD_SMA_, PENDIENTE_1M_SMA_, RATIO_SMA_SEGUNDO_, RATIO_MAXRELATIVO_SEGUNDO_, PENDIENTE_SMA_SEGUNDO_,
		PENDIENTE_1M_SMA_SEGUNDO_, PENDIENTE_2M_SMA_SEGUNDO_, MEDIA_SMA_, PENDIENTE_SMA_, PENDIENTE_2M_SMA_, RATIO_SMA_,
		RATIO_MAXRELATIVO_, RATIO_MINRELATIVO_, RATIO_MINRELATIVO_SEGUNDO_, RATIO_U_SMA_, RATIO_U_MAXRELATIVO_,
		RATIO_U_MINRELATIVO_, CURTOSIS_, SKEWNESS_, FASEWYCKOFF_, EMA_, MACD_, RSI14_, VARREL_;
	}

//	Otros menos útiles: 
//  _CLOSEHIGH, _CLOSELOW, _OPENLOW, _HIGH, _LOW, _OPEN, _CLOSEOPEN,
	public enum FINAL_NOMBRES_PARAMETROS_ELABORADOS {
		_VOLUMEN, _CLOSE, _OPENHIGH, _HIGHLOW;
	}

	public enum OTROS_PARAMS_ELAB {
		DIAS_HASTA_FIN_MES, DIAS_HASTA_FIN_TRIMESTRE;
	}

	static Locale locale;
	static DecimalFormat df;

	public static void main(String[] args) {

		Estadisticas e5 = new Estadisticas();
		e5.addValue(10D);
		e5.addValue(11D);
		e5.addValue(12D);
		e5.addValue(16D);
		e5.addValue(21D);
		e5.addValue(19D);
		System.out.println("Valores: " + e5.toString());
		System.out.println("e5.getMedia(): " + e5.getMedia());
		System.out.println("e5.getPendienteRelativa(): " + e5.getPendienteRelativa());
		System.out.println("e5.getPendienteRelativaSegundo(): " + e5.getPendienteRelativaSegundo());
		System.out.println("e5.getPendienteRelativa1M(): " + e5.getPendienteRelativa1M());
		System.out.println("e5.getPendienteRelativa1MSegundo(): " + e5.getPendienteRelativa1MSegundo());
		System.out.println("e5.getPendienteRelativa2M(): " + e5.getPendienteRelativa2M());
		System.out.println("e5.getPendienteRelativa2MSegundo(): " + e5.getPendienteRelativa2MSegundo());
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
		System.out.println("e5.getFaseWyckoff(): " + e5.getFaseWyckoff());
		System.out.println("e5.getVariacionRelativaMaxima(): " + e5.getVariacionRelativaMaxima());
		System.out.println("e5.getRatioEMA(): " + e5.getRatioEMA());
		System.out.println("e5.getRatioMACDMitadPeriodo(): " + e5.getRatioMACDMitadPeriodo());
		System.out.println("e5.getRsi14(): " + e5.getRsi14());
		System.out.println("e5.getVariacionRelativaMaxima(): " + e5.getVariacionRelativaMaxima());
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

		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.STD_SMA_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.PENDIENTE_1M_SMA_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.RATIO_SMA_SEGUNDO_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.RATIO_MAXRELATIVO_SEGUNDO_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.PENDIENTE_SMA_SEGUNDO_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.PENDIENTE_1M_SMA_SEGUNDO_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.PENDIENTE_2M_SMA_SEGUNDO_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.MEDIA_SMA_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.PENDIENTE_SMA_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.PENDIENTE_2M_SMA_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.RATIO_SMA_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.RATIO_MAXRELATIVO_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.RATIO_MINRELATIVO_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.RATIO_MINRELATIVO_SEGUNDO_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.RATIO_U_SMA_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.RATIO_U_MAXRELATIVO_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.RATIO_U_MINRELATIVO_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.CURTOSIS_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.SKEWNESS_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.FASEWYCKOFF_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.EMA_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.MACD_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.RSI14_);
		incluirParametroCebecera(ordenNombresParametrosElaborados, PREFIJOS_ELAB.VARREL_);

		locale = new Locale("en", "UK");
		df = (DecimalFormat) NumberFormat.getNumberInstance(locale);
		df.applyPattern("#0.#");
	}

	/**
	 * @param mapa
	 * @param item
	 */
	public void incluirParametroCebecera(HashMap<Integer, String> mapa, Object item) {
		mapa.put(ordenNombresParametrosElaborados.size() + 1, item.toString());
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
	 * Ratio, en porcentaje de MILLÓN, entre el PRIMER dato y la MEDIA del conjunto
	 * de datos. Puede tener valores negativos.
	 */
	public int getRatioSMA() {
		return (int) Math.round(NUM1M * (this.getElement(0) / getMean()));
	}

	/**
	 * Ratio, en porcentaje de MILLÓN, entre el SEGUNDO dato y la MEDIA del conjunto
	 * de datos. Puede tener valores negativos.
	 */
	public int getRatioSMASegundo() {
		int salida = VALOR_FAKE;
		if (this.getN() > 1)
			salida = (int) Math.round(NUM1M * (this.getElement(1) / getMean()));
		return salida;
	}

	/**
	 * Ratio, en porcentaje de MILLÓN, entre el PRIMER dato y el MAXIMO del conjunto
	 * de datos. Puede tener valores negativos.
	 */
	public int getRatioMax() {
		return (int) Math.round(NUM1M * (this.getElement(0) / this.getMax()));
	}

	/**
	 * Ratio, en porcentajede MILLÓN, entre el PRIMER dato y el MINIMO del conjunto
	 * de datos. Puede tener valores negativos.
	 */
	public int getRatioMin() {
		return (int) Math.round(NUM1M * (this.getElement(0) / this.getMin()));
	}

	/**
	 * Ratio, en porcentajede MILLÓN, entre el SEGUNDO dato y el MAXIMO del conjunto
	 * de datos. Puede tener valores negativos.
	 */
	public int getRatioMaxSegundo() {
		int salida = VALOR_FAKE;
		if (this.getN() > 1)
			salida = (int) Math.round(NUM1M * (this.getElement(1) / this.getMax()));
		return salida;
	}

	/**
	 * Ratio, en porcentaje de MILLÓN, entre el SEGUNDO dato y el MINIMO del
	 * conjunto de datos. Puede tener valores negativos.
	 */
	public int getRatioMinSegundo() {
		int salida = VALOR_FAKE;
		if (this.getN() > 1)
			salida = (int) Math.round(NUM1M * (this.getElement(1) / this.getMin()));
		return salida;
	}

	/**
	 * Ratio, en porcentaje de MILLÓN, entre el ULTIMO dato y la MEDIA del conjunto
	 * de datos. Puede tener valores negativos.
	 */
	public int getRatioUltimoSMA() {
		int salida = VALOR_FAKE;
		if (this.getN() > 1)
			salida = (int) Math.round(NUM1M * (this.getElement((int) getN() - 1) / getMean()));
		return salida;
	}

	/**
	 * Ratio, en porcentaje de MILLÓN, entre el ULTIMO dato y el MAXIMO del conjunto
	 * de datos. Puede tener valores negativos.
	 */
	public int getRatioUltimoMax() {
		int salida = VALOR_FAKE;
		if (this.getN() > 1)
			salida = (int) Math.round(NUM1M * (this.getElement((int) getN() - 1) / this.getMax()));
		return salida;
	}

	/**
	 * Ratio, en porcentaje de MILLÓN, entre el ULTIMO dato y el MINIMO del conjunto
	 * de datos. Puede tener valores negativos.
	 */
	public int getRatioUltimoMin() {
		return (int) Math.round(NUM1M * (this.getElement((int) getN() - 1) / this.getMin()));
	}

	/**
	 * Ratio, en porcentaje en MILLÓN, entre la pendiente de la segunda mitad de
	 * datos, y la pendiente de la primera mitad.
	 */
	public int getFaseWyckoff() {
		int salida = VALOR_FAKE;
		double salidaTemp = VALOR_FAKE;
		if (this.getN() > 4) {
			double numerador, denominador;
			double primero_1m, ultimo_1m, primero_2m, ultimo_2m;
			primero_1m = this.getElement(0);
			ultimo_1m = this.getElement((int) ((int) this.getN() / 2.0 - 1));
			primero_2m = this.getElement((int) (Math.floor(this.getN() / 2.0)));
			ultimo_2m = this.getElement((int) (Math.ceil(this.getN() - 1)));
			numerador = (ultimo_2m - primero_2m) / primero_2m;
			denominador = (ultimo_1m - primero_1m) / primero_1m;
			salidaTemp = numerador / denominador;
		}
		// Para evitar infinitos, asumimos estos valores como infinito
		if (salidaTemp > NUM1M) {
			salidaTemp = NUM1M;
		} else if (salidaTemp < (-1.0) * NUM1M) {
			salidaTemp = (-1.0) * NUM1M;
		}
		salida = (int) Math.round(NUM1M * salidaTemp);
		return salida;

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
		System.out.println("pendiente_sma_segundo = " + getPendienteRelativaSegundo());
		System.out.println("pendiente_1m_sma = " + getPendienteRelativa1M());
		System.out.println("pendiente_1m_sma_segundo = " + getPendienteRelativa1MSegundo());
		System.out.println("pendiente_2m_sma = " + getPendienteRelativa2M());
		System.out.println("pendiente_2m_sma_segundo = " + getPendienteRelativa2MSegundo());
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
		System.out.println("faseWyckoff = " + getFaseWyckoff());
		System.out.println("ema = " + getRatioEMA());
		System.out.println("macd = " + getRatioMACDMitadPeriodo());
		System.out.println("rsi14 = " + getRsi14());
		System.out.println("VariacionRelativaMaxima = " + getVariacionRelativaMaxima());
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

		incluirParamValor(parametros, PREFIJOS_ELAB.STD_SMA_, periodoString, finalNombreParametro, getStd(),
				rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.PENDIENTE_1M_SMA_, periodoString, finalNombreParametro,
				getPendienteRelativa1M(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.RATIO_SMA_SEGUNDO_, periodoString, finalNombreParametro,
				getRatioSMASegundo(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.RATIO_MAXRELATIVO_SEGUNDO_, periodoString, finalNombreParametro,
				getRatioMaxSegundo(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.PENDIENTE_SMA_SEGUNDO_, periodoString, finalNombreParametro,
				getPendienteRelativaSegundo(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.PENDIENTE_1M_SMA_SEGUNDO_, periodoString, finalNombreParametro,
				getPendienteRelativa1MSegundo(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.PENDIENTE_2M_SMA_SEGUNDO_, periodoString, finalNombreParametro,
				getPendienteRelativa2MSegundo(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.MEDIA_SMA_, periodoString, finalNombreParametro, getMedia(),
				rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.PENDIENTE_SMA_, periodoString, finalNombreParametro,
				getPendienteRelativa(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.PENDIENTE_2M_SMA_, periodoString, finalNombreParametro,
				getPendienteRelativa2M(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.RATIO_SMA_, periodoString, finalNombreParametro, getRatioSMA(),
				rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.RATIO_MAXRELATIVO_, periodoString, finalNombreParametro,
				getRatioMax(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.RATIO_MINRELATIVO_, periodoString, finalNombreParametro,
				getRatioMin(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.RATIO_MINRELATIVO_SEGUNDO_, periodoString, finalNombreParametro,
				getRatioMinSegundo(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.RATIO_U_SMA_, periodoString, finalNombreParametro,
				getRatioUltimoSMA(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.RATIO_U_MAXRELATIVO_, periodoString, finalNombreParametro,
				getRatioUltimoMax(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.RATIO_U_MINRELATIVO_, periodoString, finalNombreParametro,
				getRatioUltimoMin(), rellenarConInvalidos);

		// Se pone un 0 si la kurtosis no es válida. Para menos de 4 valores, será
		// siempre inválido. Se pone 0 en vez de null para que no se descarten todas las
		// filas. El clasificador no lo usará, ya que no lo considerará útil
		incluirParamValor(parametros, PREFIJOS_ELAB.CURTOSIS_, periodoString, finalNombreParametro,
				Double.isNaN(getKurtosis()) ? 0 : getKurtosis(), rellenarConInvalidos);

		incluirParamValor(parametros, PREFIJOS_ELAB.SKEWNESS_, periodoString, finalNombreParametro, getSkewness(),
				rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.FASEWYCKOFF_, periodoString, finalNombreParametro, getFaseWyckoff(),
				rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.EMA_, periodoString, finalNombreParametro, getRatioEMA(),
				rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.MACD_, periodoString, finalNombreParametro,
				getRatioMACDMitadPeriodo(), rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.RSI14_, periodoString, finalNombreParametro, getRsi14(),
				rellenarConInvalidos);
		incluirParamValor(parametros, PREFIJOS_ELAB.VARREL_, periodoString, finalNombreParametro,
				getVariacionRelativaMaxima(), rellenarConInvalidos);

		return parametros;
	}

	/**
	 * @param parametros
	 * @param prefijo
	 * @param periodo
	 * @param finalNombreParametro
	 * @param valor
	 */
	public void incluirParamValor(HashMap<String, String> parametros, Object prefijo, String periodo,
			String finalNombreParametro, Object valor, Boolean rellenarConInvalidos) {

		if (rellenarConInvalidos) {
			parametros.put(prefijo + periodo + finalNombreParametro, VALOR_INVALIDO);
		} else {

			if (valor instanceof Double && Double.isNaN((double) valor)) {
				parametros.put(prefijo + periodo + finalNombreParametro, VALOR_INVALIDO);

			} else if ((valor instanceof Double && !Double.isNaN((double) valor))
					|| valor instanceof Float && !Float.isNaN((float) valor)) {
				parametros.put(prefijo + periodo + finalNombreParametro, df4decimales.format(valor));

			} else {
				parametros.put(prefijo + periodo + finalNombreParametro, String.valueOf(valor));
			}

		}

	}

	/**
	 * @return the ordenNombresParametrosElaborados
	 */
	public HashMap<Integer, String> getOrdenNombresParametrosElaborados() {
		return ordenNombresParametrosElaborados;
	}

	/**
	 * Incremento máximo respecto al incremento medio.
	 * 
	 * @return Para un conjunto de valores, se obtienen sus diferencias en
	 *         porcentaje sin signo y_t=ABS(x_i+1 - x_i). Se devuelve:
	 *         ABS(max(y_t)/average(y_t))
	 */
	public Double getVariacionRelativaMaxima() {

		// INCREMENTOS ABSOLUTOS
		ResizableDoubleArray incrementos = new ResizableDoubleArray(1);
		for (int i = 0; i < getN() - 1; i++) {
			incrementos.addElement(Math.abs(getElement(i + 1) - getElement(i))); // INCREMENTO ABSOLUTO
		}

		// INCREMENTO MAXIMO
		Double max = 0D, average = 0D;
		for (int counter = 0; counter < incrementos.getNumElements(); counter++) {
			if (max < incrementos.getElement(counter)) {
				max = incrementos.getElement(counter);
			}
			average += incrementos.getElement(counter); // acumular, para despues hacer la media
		}

		Double n = Double.valueOf(incrementos.getNumElements()); // N elementos
		average = average / n; // INCREMENTO MEDIO
//		System.out.println("max: "+max);
//		System.out.println("average: "+average);
		return Math.abs(max / average); // INCREMENTO MAXIMO RESPECTO AL MEDIO
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

	/**
	 * Ratio Exponential Moving Average, en porcentaje de MILLÓN
	 */
	public int getRatioEMA() {
		return getRatioEMAPeriodo((int) this.getN());
	}

	/**
	 * Ratio Exponential Moving Average por periodo, en porcentaje de MILLÓN
	 */
	public int getRatioEMAPeriodo(int period) {

		double[] prices = this.getValues();

		double[] periodSma;
		double smoothingConstant;
		double[] periodEma;

		smoothingConstant = 2d / (period + 1);

		periodSma = new double[prices.length];
		periodEma = new double[prices.length];

		SimpleMovingAverage sma = new SimpleMovingAverage();

		int salida = VALOR_FAKE;

		try {
			for (int i = (period - 1); i < prices.length; i++) {
				double[] slice = Arrays.copyOfRange(prices, 0, i + 1);

				double[] smaResults = sma.calculate(slice, period).getSMA();

				periodSma[i] = smaResults[smaResults.length - 1];

				if (i == (period - 1)) {
					periodEma[i] = periodSma[i];
				} else if (i > (period - 1)) {
					// Formula: (Close - EMA(previous day)) x multiplier +
					// EMA(previous day)
					periodEma[i] = (prices[i] - periodEma[i - 1]) * smoothingConstant + periodEma[i - 1];
				}

				periodEma[i] = NumberFormatter.round(periodEma[i]);
			}

			// Se toma sólo el último valor del periodo
			salida = (int) Math.round(NUM1M * (periodEma[prices.length - 1]));

		} catch (Exception e) {
			MY_LOGGER.debug(e.getMessage());
		}

		return salida;
	}

	/**
	 * Ratio MACD calculado según: MACD =
	 * EMA(datos,floor(periodo/2))-EMA(datos,periodo) En porcentaje de MILLÓN.
	 * 
	 */
	public int getRatioMACDMitadPeriodo() {
		return getRatioEMAPeriodo((int) Math.floor(this.getN() / 2)) - getRatioEMAPeriodo((int) this.getN());
	}

	/**
	 * 
	 * @return RSI de 14 elementos.
	 */
	public double getRsi14() {

		int period = 14;
		EMA ema = new EMA(2 * period - 1);

		// fill 'up' and 'down' table - 'up' when today prize is bigger than yesterday,
		// 'down' when today is lower than yesterday
		final double[] up = new double[(int) getN() - 1];
		final double[] down = new double[(int) getN() - 1];
		for (int i = 0; i < (int) getN() - 1; i++) {
			if (getElement(i) > getElement(i + 1)) {
				up[i] = getElement(i) - getElement(i + 1);
				down[i] = 0;
			}
			if (getElement(i) < getElement(i + 1)) {
				down[i] = Math.abs(getElement(i) - getElement(i + 1));
				up[i] = 0;
			}
		}

		// count EMA for up and down tables
		final int emaLength = (int) getN() - 2 * period;
		double[] rsis = new double[0];
		if (emaLength > 0) {
			final double[] emus = new double[emaLength];
			final double[] emds = new double[emaLength];
			ema.count(up, 0, emus);
			ema.count(down, 0, emds);

			// count RSI with RSI recursive formula
			rsis = new double[emaLength];
			for (int i = 0; i < rsis.length; i++) {
				rsis[i] = 100 - (100 / (double) (1 + emus[i] / emds[i]));
			}
		}

		double salida = 0;
		if (rsis.length > 0)
			salida = rsis[rsis.length - 1];

		return salida;
	}
}
