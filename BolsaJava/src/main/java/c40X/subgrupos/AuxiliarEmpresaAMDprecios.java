package c40X.subgrupos;

/**
 * Clase con ANIO,MES,DIA,HIGH,LOW de una empresa
 *
 */
public class AuxiliarEmpresaAMDprecios implements Comparable {

	public Integer ordenCreacion;
	public Integer anio, mes, dia;
	public Float high, low;

	public AuxiliarEmpresaAMDprecios(Integer ordenCreacion, Integer anio, Integer mes, Integer dia, Float high,
			Float low) {
		super();
		this.ordenCreacion = ordenCreacion;
		this.anio = anio;
		this.mes = mes;
		this.dia = dia;
		this.high = high;
		this.low = low;
	}

	@Override
	public String toString() {
		return "[ordenCreacion|anio|mes|dia|high|low =>" + ordenCreacion + "|" + anio + "|" + mes + "|" + dia + "|"
				+ high + "|" + low + "]";
	}

	@Override
	public int compareTo(Object otro) {

		int salida = 0;// default

		if (otro instanceof AuxiliarEmpresaAMDprecios && this != null && this.ordenCreacion != null && otro != null
				&& ((AuxiliarEmpresaAMDprecios) otro).ordenCreacion != null) {
			salida = this.ordenCreacion.compareTo(((AuxiliarEmpresaAMDprecios) otro).ordenCreacion);
		}
		return salida;
	}

}
