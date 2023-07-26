package c10X.brutos;

import java.io.Serializable;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;

/**
 * Dato dinámico sobre una operacion hecha por un insider de una empresa. Dato
 * traido de FINVIZ.
 *
 */
public class OperacionInsiderFinvizModelo implements Serializable {

	private static final long serialVersionUID = 1L;

	public static final String VENTA = "Sale";
	public static final String COMPRA = "Buy";
	public static final String OPTION_EXERCISE = "Option Exercise";

	public Calendar fecha;
	public String tipooperacion;
	public Long importe;

	public static final SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd");

	public OperacionInsiderFinvizModelo(String fecha, String tipooperacion, String importe) throws ParseException {
		super();

		Calendar fechaIn = Calendar.getInstance();
		fechaIn.setTime(sdf.parse(fecha));
		this.fecha = fechaIn;

		this.tipooperacion = tipooperacion;

		if (tipooperacion != null) {
			if (tipooperacion.equalsIgnoreCase(VENTA)) {
				// Signo NEGATIVO porque es una venta (y esto nos servirá para tratar todas las
				// operaciones a la vez)
				this.importe = Long.valueOf("-" + importe.replace(",", ""));

			} else if (tipooperacion.equalsIgnoreCase(COMPRA)) {
				// POSITIVO: COMPRAS y OPTION_EXERCISE (que normalmente son compras)
				this.importe = Long.valueOf(importe.replace(",", ""));

			} else {
				// OPTION_EXERCISE: indeterminado. No sabemos si son compras o ventas
				this.importe = null;
			}
		}

	}

	public OperacionInsiderFinvizModelo() {
		super();
		fecha = null;
		tipooperacion = "";
		importe = null;
	}

	@Override
	public String toString() {
		return "OIFM [fecha=" + sdf.format(fecha.getTime()) + ", tipooperacion=" + tipooperacion + ", importe="
				+ importe + "]";
	}

}
