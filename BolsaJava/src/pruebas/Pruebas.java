package pruebas;

import java.io.File;
import java.text.DateFormat;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

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

		Locale locale = new Locale("en", "UK");
		DecimalFormat df2 = (DecimalFormat) NumberFormat.getNumberInstance(locale);
		df2.applyPattern("#0.#######");
		Float numeroF1 = 9.75F;
		Float numeroF2 = Float.valueOf(numeroF1) / 1000000F;
		String numStr3 = df2.format(numeroF2);
		System.out.println("Entrada=" + numeroF1 + " --> Salida=" + numStr3);

		String dirSubgrupoOut = "/bolsa/pasado/subgrupos/SG_0/";
		System.out.println("Creando la carpeta del subgrupo con ID=0 en: " + dirSubgrupoOut);
		File dirSubgrupoOutFile = new File(dirSubgrupoOut);
		boolean dirCreadoBien = dirSubgrupoOutFile.mkdir();
		if (!dirCreadoBien) {
			System.err.println("Error al crear la carpeta: " + dirSubgrupoOut);
		}

	}

}
