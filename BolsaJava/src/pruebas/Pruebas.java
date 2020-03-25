package pruebas;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import c10X.brutos.EstaticoNasdaqModelo;

/**
 * @author casa
 *
 */
public class Pruebas {

	public Pruebas() {
	}

	public static void main(String[] args) {

		// ----------------------------------------------------
		String filaPredicha = "AAOI|0|NASDAQ|2020|1|2|21|30|0|12.53|12.43|12.49|12.43|11.9|0.3|-0.1|105.0|12.5|11.4|-0.9|0.4|0.0|0.0|0.0|0.0|0.0|0.0|||11.6|0.4|-0.0|108.0|12.5|11.0|-0.6|0.5|0.0|0.0|0.0|0.0|0.0|0.0|||11.1|0.5|0.0|113.0|12.5|10.2|-0.2|0.3|0.0|0.0|0.0|0.0|0.0|0.0|||10.8|0.7|0.0|117.0|12.5|9.2|-0.4|0.1|0.0|0.0|0.0|0.0|0.0|0.0||||0";
		String[] partes = filaPredicha.split("\\|");
		String tiempoVelaPredicha = partes[3] + "|" + partes[4] + "|" + partes[5] + "|" + partes[6] + "|" + partes[7];
		System.out.println(tiempoVelaPredicha);

		// ----------------------------------------------------
		List<EstaticoNasdaqModelo> lista = new ArrayList<EstaticoNasdaqModelo>();
		lista.add(new EstaticoNasdaqModelo("Aasdf", "", "", "", "", "", "", ""));
		lista.add(new EstaticoNasdaqModelo("Basdfs", "", "", "", "", "", "", ""));
		lista.add(new EstaticoNasdaqModelo("ZZZasdfs", "", "", "", "", "", "", ""));
		lista.add(new EstaticoNasdaqModelo("Casdfsdf", "", "", "", "", "", "", ""));
		Collections.sort(lista);
		Collections.reverse(lista);

		// ----------------------------------------------------
		String fechaEarningsStr = "Jan 28 AMC"; // BMO=(Before Market Open), AMC=(After Market Close)
		String[] earningsFechaPartes = fechaEarningsStr.split(" ");
		boolean earningsImpactaEnDiaSiguiente = false;
		if (fechaEarningsStr.contains("AMC")) {
			earningsImpactaEnDiaSiguiente = false;

		} else if (fechaEarningsStr.contains("BMO")) {
			earningsImpactaEnDiaSiguiente = false;
		}

		// ----------------------------------------------------

		String a = String.valueOf(Long.valueOf("1234567890") / 1000000);
		System.out.println(a);
	}

}
