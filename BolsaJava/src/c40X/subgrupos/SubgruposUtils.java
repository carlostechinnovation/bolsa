package c40X.subgrupos;

public class SubgruposUtils {

	public static final String DIR_SUBGRUPOS = "/bolsa/pasado/subgrupos/";
	// Cobertura mínima en tanto por 100
	public static final String MIN_COBERTURA_CLUSTER = "60";
	public static final String MIN_EMPRESAS_POR_CLUSTER = "10";

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

	/**
	 * Posición del carácter en el String pasado como parámetro, en la ocurrencia
	 * indicada como parámetro.
	 * 
	 * @param caracterABuscar
	 * @param numOcurrencias
	 * @param entrada
	 * @return
	 */
	public static Integer indiceDeAparicion(final Character caracterABuscar, final Integer numOcurrencias,
			final String entrada) {
		Integer salida = -1;
		Integer vecesEncontradas = 0;
		for (int i = 0; i < entrada.length(); i++) {
			if (entrada.charAt(i) == caracterABuscar) {
				vecesEncontradas++;
			}
			if (vecesEncontradas == numOcurrencias) {
				salida = i;
				break;
			}
		}
		return salida;
	}

	/**
	 * Devuelve el trozo final de la cadena de entrada que esté a la derecha del
	 * string buscado, en la aparición número "numOcurrenciasDesdeElFinal" desde el
	 * final de la cadena.
	 * 
	 * @param caracterABuscar
	 * @param numOcurrenciasDesdeElFinal
	 * @param entrada
	 * @return
	 */
	public static String recortaUltimaParteDeString(final Character caracterABuscar,
			final Integer numOcurrenciasDesdeElFinal, final String entrada) {
		Integer indice = SubgruposUtils.indiceDeAparicion(caracterABuscar,
				(int) entrada.chars().filter(ch -> ch == caracterABuscar).count() - numOcurrenciasDesdeElFinal + 1,
				entrada);
		String salida = entrada.substring(indice + 1);
		return salida;
	}

}
