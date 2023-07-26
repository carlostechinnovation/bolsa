package c30x.elaborados.construir;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;

import c10X.brutos.BrutosUtils;
import c10X.brutos.EstaticosFinvizDescargarYParsear;
import c10X.brutos.FinvizNoticiasEmpresa;

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

	@Test
	public void parsearFinviz1Test() throws Exception {

		String idEmpresa = "BBIO";
		Map<String, String> mapaExtraidos = new HashMap<String, String>();
		List<String> operacionesInsidersLimpias = new ArrayList<String>();
		FinvizNoticiasEmpresa noticias = new FinvizNoticiasEmpresa(BrutosUtils.MERCADO_NQ, idEmpresa);

		ClassLoader classLoader = getClass().getClassLoader();
		File file = new File(classLoader.getResource("FZ_NASDAQ_" + idEmpresa + ".html").getFile());
		String rutaHtmlBruto = file.getAbsolutePath();

		EstaticosFinvizDescargarYParsear.parsearFinviz1(BrutosUtils.MERCADO_NQ, idEmpresa, rutaHtmlBruto, mapaExtraidos,
				operacionesInsidersLimpias, noticias);

		assert (mapaExtraidos.size() > 0);
		assert (noticias.mapa.size() > 0);

		// en el dia 8-dic-2021 deben aparecen 2 noticias:
		List<String> noticias8dic2021 = noticias.mapa.get(20211208);
		assert (noticias8dic2021.size() == 2);

	}

}
