package pruebas;

/**
 * @author casa
 *
 */
public class Pruebas {

	public Pruebas() {
	}

	public static void main(String[] args) {

		String hola = "hol|a";
		int numeroCamposActual = 0;
		if (hola != null & !hola.isEmpty() && hola.contains("|")) {
			numeroCamposActual = hola.split("\\|").length;
			System.out.println(numeroCamposActual);
		}

	}

}
