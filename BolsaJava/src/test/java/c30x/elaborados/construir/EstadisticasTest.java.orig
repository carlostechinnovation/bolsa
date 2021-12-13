package c30x.elaborados.construir;

import static org.junit.Assert.assertTrue;

import java.util.HashMap;

import org.junit.Before;
import org.junit.Test;

public class EstadisticasTest {

	@Before
	public void preparar() {

	}

	@Test
	public void getPendienteRelativaTest() {
		Estadisticas modelo = new Estadisticas();
		modelo.addValue(5.5D);
		modelo.addValue(7.0D);
		modelo.addValue(25.0D); // Este valor de pico intermedio no lo detecta la pendiente
		modelo.addValue(15.0D);
		double out = modelo.getPendienteRelativa();

		double esperado = 100.0D * (15.0D - 5.5D) / (5.5D * 4);
		assertTrue(out == esperado);
	}

	@Test
	public void getRatioMaxTest() {
		Estadisticas modelo = new Estadisticas();
		modelo.addValue(5.5D);// Vela 0
		modelo.addValue(7.0D);
		modelo.addValue(25.0D); // Máximo
		modelo.addValue(15.0D);
		double out = modelo.getRatioMax();

		double esperado = Estadisticas.NUM1M * (5.5D / 25.0D);
		assertTrue(out == esperado);
	}

	@Test
	public void getVariacionRelativaMaximaTest() {
		Estadisticas modelo = new Estadisticas();
		modelo.addValue(1.02D);// Vela 0
		modelo.addValue(1.04D);
		modelo.addValue(1.09D);
		modelo.addValue(1.15D);// Máximo
		double out = modelo.getVariacionRelativaMaxima();

		// Incrementos --> 0.02, 0.05, 0.06
		// Incremento maximo/medio = 0.06 / ((0.02 + 0.05 + 0.06)/3) = 1.3846

		double esperado = 1.3846153846153817D;
		assertTrue(out == esperado);
	}

	@Test
	public void incluirParamValorTest() {
		Estadisticas modelo = new Estadisticas();
		HashMap<String, String> parametros = new HashMap<String, String>();
		String prefijo = "PREFIJO_", periodo = "50D";
		String finalNombreParametro = "_SUFIJO";
		Boolean rellenarConInvalidos = false;

		int valorInt = 1;
		modelo.incluirParamValor(parametros, prefijo, periodo, finalNombreParametro, valorInt, rellenarConInvalidos);

		double valorDouble = 1;
		modelo.incluirParamValor(parametros, prefijo, periodo, finalNombreParametro, valorDouble, rellenarConInvalidos);

		double valorDoubleNaN = Double.NaN;
		modelo.incluirParamValor(parametros, prefijo, periodo, finalNombreParametro, valorDoubleNaN,
				rellenarConInvalidos);
	}

}
