package c30x.elaborados.construir;

import static org.junit.Assert.assertTrue;

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

		double esperado = 100.0D * (5.5D / 25.0D);
		assertTrue(out == esperado);
	}

}
