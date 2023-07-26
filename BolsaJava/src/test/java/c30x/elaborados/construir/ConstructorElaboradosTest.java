package c30x.elaborados.construir;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Scanner;

import org.junit.Before;
import org.junit.Test;

public class ConstructorElaboradosTest {

	@Before
	public void preparar() {

	}

	@Test
	public void calcularTargetTest() throws IOException {

		ClassLoader classLoader = getClass().getClassLoader();
		File file = new File(classLoader.getResource("elaborados_test1.txt").getFile());
		Scanner myReader = new Scanner(file);
		HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada = new HashMap<Integer, HashMap<String, String>>();
		while (myReader.hasNextLine()) {
			String cadena = myReader.nextLine().trim();
			if (cadena != null && !cadena.isEmpty()) {
				procesarFilaFichero1(cadena, datosEmpresaEntrada);
			}
		}
		myReader.close();

		String empresa = "AACG";
		Integer ANTIGUEDAD_ESTUDIADA = 20; // ANTIGUEDAD ESTUDIADA (su target será 1 ó 0 mirando las X+M velas más
		// recientes (futuras)
		Integer S = 10;
		Integer X = 10;
		Integer R = 10;
		Integer M = 1;
		Integer F = 10;
		Integer B = 10;
		Double umbralMaximo = 5.0D;
		Double umbralMinimo = 0.0D;

		String targetCalculado = ConstructorElaborados.calcularTarget(empresa, datosEmpresaEntrada,
				ANTIGUEDAD_ESTUDIADA, S, X, R, M, F, B, umbralMaximo, umbralMinimo);
		String targetEsperado = "1";

		assertTrue(targetCalculado.equals(targetEsperado));
	}

	/**
	 * FUNCION AUXILIAR - Procesa una linea del fichero de entrada de test.
	 * 
	 * @param cadena              Fila en la que aparece: antiguedad y
	 *                            featuresClaveValor
	 * @param datosEmpresaEntrada Mapa ya creado al que se anhaden los features
	 *                            procesados
	 */
	private void procesarFilaFichero1(String cadena, HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada) {

		Integer antiguedad = Integer.valueOf(cadena.substring(0, cadena.indexOf("=")));
		String featuresClaveValor = cadena.substring(1 + cadena.indexOf("{"), cadena.indexOf("}"));
		String[] featuresClaveValorPartes = featuresClaveValor.split(",");

		HashMap<String, String> mapaFeatures = new HashMap<String, String>(0);
		for (String fcv : featuresClaveValorPartes) {
			String[] claveyvalor = fcv.trim().split("=");
			mapaFeatures.put(claveyvalor[0], claveyvalor[1]);
		}
		datosEmpresaEntrada.put(antiguedad, mapaFeatures);

		// Pintamos los 50 precios close más recientes
		if (antiguedad < 30) {
			System.out.println("(ConstructorElaboradosTest.procesarFilaFichero1): antiguedad=" + antiguedad
					+ " -->close=" + mapaFeatures.get("close"));
		}

	}

}
