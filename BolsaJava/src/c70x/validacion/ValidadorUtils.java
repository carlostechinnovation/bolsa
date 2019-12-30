package c70x.validacion;

public class ValidadorUtils {

	public final static Integer HORAS_AL_DIA = 7;

	public static final String PATH_PREDICHO = "/bolsa/futuro/subgrupos/";
	public static final String PATH_VALIDACION = "/bolsa/validacion/";
	// Parametros del TARGET (subida del S% en precio de close, tras X velas, y no
	// cae mas de un R% dentro de las siguientes M velas posteriores)
	public static final Integer S = 10;
	public static final Integer X = 4 * HORAS_AL_DIA;
	public static final Integer R = 5;
	public static final Integer M = 1 * HORAS_AL_DIA;

}
