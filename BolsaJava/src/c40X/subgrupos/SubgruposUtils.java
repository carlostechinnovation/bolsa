package c40X.subgrupos;

public class SubgruposUtils {

	public static final String DIR_SUBGRUPOS = "/bolsa/pasado/datasets/";
	//Cobertura mínima en tanto por 100
	public static final String MIN_COBERTURA_CLUSTER="80";
	
	/**
	 * 
	 * @param caracterABuscar
	 * @param numOcurrencias
	 * @param entrada
	 * @return
	 */
	public static String recortaPrimeraParteDeString(final Character caracterABuscar, final Integer numOcurrencias,
			final String entrada) {
		String salida = "";
		Integer vecesEncontradas = 0;
		for (int i = 0; i < entrada.length(); i++) {
			if (entrada.charAt(i) == caracterABuscar) {
				vecesEncontradas++;
			}
			if (vecesEncontradas == numOcurrencias) {
				salida = entrada.substring(i + 1);
				break;
			}
		}
		return salida;
	}

}
