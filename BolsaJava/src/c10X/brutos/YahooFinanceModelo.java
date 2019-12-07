package c10X.brutos;

public class YahooFinanceModelo {

	String mercado = BrutosUtils.NULO;
	String empresa = BrutosUtils.NULO;
	String anio = BrutosUtils.NULO;
	String mes = BrutosUtils.NULO;
	String dia = BrutosUtils.NULO;
	String hora = BrutosUtils.NULO;
	String minuto = BrutosUtils.NULO;
	String volumen = BrutosUtils.NULO;
	String high = BrutosUtils.NULO;
	String low = BrutosUtils.NULO;
	String close = BrutosUtils.NULO;
	String open = BrutosUtils.NULO;

	public YahooFinanceModelo(String mercado, String empresa, String anio, String mes, String dia, String hora,
			String minuto, String volumen, String high, String low, String close, String open) {
		super();
		this.mercado = mercado;
		this.empresa = empresa;
		this.anio = anio;
		this.mes = mes;
		this.dia = dia;
		this.hora = hora;
		this.minuto = minuto;
		this.volumen = volumen;
		this.high = high;
		this.low = low;
		this.close = close;
		this.open = open;
	}

}
