package c30x.elaborados.construir;

public class ElaboradosUtils {

	public final static Integer HORAS_AL_DIA = 7;

	public static final String DIR_ELABORADOS = "/bolsa/pasado/elaborados/";
	// Parametros del TARGET (subida del S% en precio de close, tras X velas, y no
	// cae mas de un R% dentro de las siguientes M velas posteriores)
	public static final Integer S = 10;
	public static final Integer X = 4 * HORAS_AL_DIA;
	public static final Integer R = 5;
	public static final Integer M = 1 * HORAS_AL_DIA;
}
