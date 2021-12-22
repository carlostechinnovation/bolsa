package c30x.elaborados.construir;

import java.io.File;
import java.io.IOException;

import org.junit.Before;
import org.junit.Test;

import c10X.brutos.EstaticosFinvizDescargarYParsear;

public class EstaticosFinvizDescargarYParsearTest {
	@Before
	public void preparar() {

	}

	@Test
	public void limpiarDuplicadosEnFicheroDesconocidosTest() throws IOException {

		ClassLoader classLoader = getClass().getClassLoader();
		File file = new File(classLoader.getResource("desconocidos_test.txt").getFile());

		int contadorListaLimpia = EstaticosFinvizDescargarYParsear.limpiarDuplicadosEnFicheroDesconocidos(
				file.getAbsolutePath(), "/tmp/EstaticosFinvizDescargarYParsearTest.salida.txt");

		assert (contadorListaLimpia == 4);

	}
}
