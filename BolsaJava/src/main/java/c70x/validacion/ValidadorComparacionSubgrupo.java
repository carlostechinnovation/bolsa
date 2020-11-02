package c70x.validacion;

import java.io.Serializable;

import c30x.elaborados.construir.Estadisticas;

public class ValidadorComparacionSubgrupo implements Serializable {

	private static final long serialVersionUID = 1L;

	public String empresaPredicha, empresaValidacion, antiguedadValidacion, fechaPredicha, fechaValidacion;
	public Integer indicePredicha, indiceValidacion;

	public Float aciertosTargetUnoSubgrupo, fallosTargetUnoSubgrupo, totalTargetUnoEnSubgrupo;
	public Integer antiguedadFutura;
	public Estadisticas performanceClose, performanceCloseAcertados, performanceCloseFallados;
	public Double closeValidacionFutura, closeValidacionActual;
	public Double mediaRendimientoClose, mediaRendimientoCloseAcertados, mediaRendimientoCloseFallados;
	public Double stdRendimientoClose, stdRendimientoCloseAcertados, stdRendimientoCloseFallados;
	public Boolean acertado;

	public ValidadorComparacionSubgrupo() {
		super();

		aciertosTargetUnoSubgrupo = 0F;
		fallosTargetUnoSubgrupo = 0F;
		totalTargetUnoEnSubgrupo = 0F;
		performanceClose = new Estadisticas();
		performanceCloseAcertados = new Estadisticas();
		performanceCloseFallados = new Estadisticas();
	}

	/**
	 * @return
	 */
	protected Double calcularRentabilidad() {
		return (closeValidacionFutura - closeValidacionActual) / closeValidacionActual;
	}

}
