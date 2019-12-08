package pruebas;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * @author casa
 *
 */
public class Pruebas {

	public Pruebas() {
	}

	public static void main(String[] args) {

		long segundosDesde1970 = 1575386791 * 1000L;
		System.out.println(segundosDesde1970);
		Date currentDate = new Date(segundosDesde1970);

		long currentDateTime = System.currentTimeMillis();
		System.out.println(currentDateTime);

		DateFormat df = new SimpleDateFormat("yyyy|MM|dd|HH|mm");
		String cadena = df.format(currentDate);

//		String cadena = (anio + 1970) + "|" + mes + "|" + dia + "|" + hora + "|" + minuto + "|";
		System.out.println(cadena);

		String cadenaconAntiguedad = "124" + "|" + cadena;
		int indexPrimerPipe = cadenaconAntiguedad.indexOf("|");
		String antiguedad = cadenaconAntiguedad.substring(0, indexPrimerPipe);
		String fechaStr = cadenaconAntiguedad.substring(indexPrimerPipe + 1);
		System.out.println(antiguedad);
		System.out.println(fechaStr);
	}

}
