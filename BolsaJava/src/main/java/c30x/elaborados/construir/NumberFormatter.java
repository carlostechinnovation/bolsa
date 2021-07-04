package c30x.elaborados.construir;

import java.math.BigDecimal;

public class NumberFormatter {

	private NumberFormatter() {
	}

	public static double round(double value) {
		return NumberFormatter.round(value, 2);
	}

	public static double round(double value, int numberOfDigitsAfterDecimalPoint) {
		BigDecimal bigDecimal = new BigDecimal(value);
		bigDecimal = bigDecimal.setScale(numberOfDigitsAfterDecimalPoint, BigDecimal.ROUND_HALF_UP);
		return bigDecimal.doubleValue();
	}

}