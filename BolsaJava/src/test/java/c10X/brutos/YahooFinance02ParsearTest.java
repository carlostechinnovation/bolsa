package c10X.brutos;

import java.io.IOException;

import org.json.simple.JSONArray;
import org.junit.Before;
import org.junit.Test;

public class YahooFinance02ParsearTest {

	@Before
	public void preparar() {

	}

	@Test
	public void detectarAnomaliasGigantesTest() throws IOException {

		JSONArray listaPreciosDia1 = new JSONArray();
		listaPreciosDia1.add(0, "empresa1");
		listaPreciosDia1.add(0, "empresa1");

		JSONArray listaPreciosDia2 = new JSONArray();
		listaPreciosDia1.add(0, "empresa1");

//		assert (YahooFinance02Parsear.detectarAnomaliasGigantes("empresa1", listaPreciosDia1,
//				listaPreciosDia2) == true);

		// TODO pendiente implementar...

	}

}
