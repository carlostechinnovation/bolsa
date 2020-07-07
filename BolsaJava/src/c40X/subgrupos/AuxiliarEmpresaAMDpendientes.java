package c40X.subgrupos;

import java.io.Serializable;

/**
 * Clase con ANIO,MES,DIA, PENDIENTE_HIGH_3D,PENDIENTE_HIGH_7D,
 * PENDIENTE_LOW_3D,PENDIENTE_LOW_7D
 *
 */
public class AuxiliarEmpresaAMDpendientes implements Serializable {

	private static final long serialVersionUID = 1L;

	public static String SG_HIGH_PENDIENTE3D = "SG_HIGH_PENDIENTE3D";
	public static String SG_HIGH_PENDIENTE7D = "SG_HIGH_PENDIENTE7D";
	public static String SG_LOW_PENDIENTE3D = "SG_LOW_PENDIENTE3D";
	public static String SG_LOW_PENDIENTE7D = "SG_LOW_PENDIENTE7D";

	Integer anio, mes, dia;
	Float highPendiente3D, highPendiente7D;
	Float lowPendiente3D, lowPendiente7D;

	public AuxiliarEmpresaAMDpendientes(Integer anio, Integer mes, Integer dia, Float highPendiente3D,
			Float highPendiente7D, Float lowPendiente3D, Float lowPendiente7D) {
		super();
		this.anio = anio;
		this.mes = mes;
		this.dia = dia;
		this.highPendiente3D = highPendiente3D;
		this.highPendiente7D = highPendiente7D;
		this.lowPendiente3D = lowPendiente3D;
		this.lowPendiente7D = lowPendiente7D;
	}

	@Override
	public String toString() {
		return "anio|mes|dia|" + SG_HIGH_PENDIENTE3D + "|" + SG_HIGH_PENDIENTE7D + "|" + SG_LOW_PENDIENTE3D + "|"
				+ SG_LOW_PENDIENTE7D + " = " + anio + "|" + String.format("%02d", mes) + "|"
				+ String.format("%02d", dia) + "|" + highPendiente3D + "|" + highPendiente7D + "|" + lowPendiente3D
				+ "|" + lowPendiente7D;
	}

	/**
	 * @return
	 */
	public static String getCabeceraSinAMD() {
		return "|" + SG_HIGH_PENDIENTE3D + "|" + SG_HIGH_PENDIENTE7D + "|" + SG_LOW_PENDIENTE3D + "|"
				+ SG_LOW_PENDIENTE7D;
	}

	/**
	 * 
	 */
	public String getDatosParaCSVSinAMD() {
		return "|" + String.format("%.6f", highPendiente3D) + "|" + String.format("%.6f", highPendiente7D) + "|"
				+ String.format("%.6f", lowPendiente3D) + "|" + String.format("%.6f", lowPendiente7D);
	}

	/**
	 * @return
	 */
	public static String getDatosParaCSVSinAMDnulo() {
		return "|" + "|" + "|" + "|";
	}

}
