package c10X.brutos;

import java.io.Serializable;

/**
 * Dato dinámico sobre una operacion hecha por un insider de una empresa. Dato
 * traido de FINVIZ.
 *
 */
public class OperacionInsiderFinvizModelo implements Serializable {

	private static final long serialVersionUID = 1L;

	public String fecha;
	public String tipooperacion;
	public String importe;

	public OperacionInsiderFinvizModelo(String fecha, String tipooperacion, String importe) {
		super();
		this.fecha = fecha;
		this.tipooperacion = tipooperacion;
		this.importe = importe;
	}

	public OperacionInsiderFinvizModelo() {
		super();
		fecha = "";
		tipooperacion = "";
		importe = "";
	}

}
