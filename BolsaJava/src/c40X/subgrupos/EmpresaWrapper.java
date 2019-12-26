package c40X.subgrupos;

import java.io.File;
import java.util.HashMap;

import org.apache.commons.math3.ml.clustering.Clusterable;

public class EmpresaWrapper implements Clusterable {
	private double[] points;
	private File fichero;
	private HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada;
	private Double valorClustering;

	public EmpresaWrapper(Empresa empresa) {
		this.fichero = empresa.getFichero();
		this.datosEmpresaEntrada = empresa.getDatosEmpresaEntrada();
		HashMap<Integer, HashMap<String, String>> datosAux = empresa.getDatosEmpresaEntrada();
		// Por simplicidad, para seleccionar las posiciones para calcular el clustering
		// (distancias entre empresas),
		// tomo sólo los datos de antigüedad=1 (la fila de antigüedad = 0 suele estar en un periodo incompleto de tiempo)
		HashMap<String, String> parametrosAux = datosAux.get(1);

//        De entre estos parámetros, debo escoger MANUALMENTE los que creo que son más útiles para clusterizar las empresas
//        empresa|antiguedad|mercado|anio|mes|dia|hora|minuto|volumen|high|low|close|open|Insider Own|Debt/Eq|
//        P/E|Dividend %|Employees|Inst Own|Market Cap|MEDIA_SMA_21_PRECIO|STD_SMA_21_PRECIO|PENDIENTE_SMA_21_PRECIO|
//        RATIO_SMA_21_PRECIO|RATIO_MAXRELATIVO_21_PRECIO|RATIO_MINRELATIVO_21_PRECIO|CURTOSIS_21_PRECIO|SKEWNESS_21_PRECIO|
//        MEDIA_SMA_21_VOLUMEN|STD_SMA_21_VOLUMEN|PENDIENTE_SMA_21_VOLUMEN|RATIO_SMA_21_VOLUMEN|RATIO_MAXRELATIVO_21_VOLUMEN|
//        RATIO_MINRELATIVO_21_VOLUMEN|CURTOSIS_21_VOLUMEN|SKEWNESS_21_VOLUMEN|MEDIA_SMA_49_PRECIO|STD_SMA_49_PRECIO|
//        PENDIENTE_SMA_49_PRECIO|RATIO_SMA_49_PRECIO|RATIO_MAXRELATIVO_49_PRECIO|RATIO_MINRELATIVO_49_PRECIO|
//        CURTOSIS_49_PRECIO|SKEWNESS_49_PRECIO|MEDIA_SMA_49_VOLUMEN|STD_SMA_49_VOLUMEN|PENDIENTE_SMA_49_VOLUMEN|
//        RATIO_SMA_49_VOLUMEN|RATIO_MAXRELATIVO_49_VOLUMEN|RATIO_MINRELATIVO_49_VOLUMEN|CURTOSIS_49_VOLUMEN|
//        SKEWNESS_49_VOLUMEN|MEDIA_SMA_140_PRECIO|STD_SMA_140_PRECIO|PENDIENTE_SMA_140_PRECIO|RATIO_SMA_140_PRECIO|
//        RATIO_MAXRELATIVO_140_PRECIO|RATIO_MINRELATIVO_140_PRECIO|CURTOSIS_140_PRECIO|SKEWNESS_140_PRECIO|
//        MEDIA_SMA_140_VOLUMEN|STD_SMA_140_VOLUMEN|PENDIENTE_SMA_140_VOLUMEN|RATIO_SMA_140_VOLUMEN|
//        RATIO_MAXRELATIVO_140_VOLUMEN|RATIO_MINRELATIVO_140_VOLUMEN|CURTOSIS_140_VOLUMEN|SKEWNESS_140_VOLUMEN|
//        MEDIA_SMA_350_PRECIO|STD_SMA_350_PRECIO|PENDIENTE_SMA_350_PRECIO|RATIO_SMA_350_PRECIO|RATIO_MAXRELATIVO_350_PRECIO|
//        RATIO_MINRELATIVO_350_PRECIO|CURTOSIS_350_PRECIO|SKEWNESS_350_PRECIO|MEDIA_SMA_350_VOLUMEN|STD_SMA_350_VOLUMEN|
//        PENDIENTE_SMA_350_VOLUMEN|RATIO_SMA_350_VOLUMEN|RATIO_MAXRELATIVO_350_VOLUMEN|RATIO_MINRELATIVO_350_VOLUMEN|
//        CURTOSIS_350_VOLUMEN|SKEWNESS_350_VOLUMEN|TARGET

//        Escojo (PENDIENTE DE VOLVERLO A ANALIZAR) los siguientes, sumados sin ponderar, 
//        ya que ambas están centradas en 0, y lo habitual para ambas es: Some says for skewness (−1,1) 
//        and (−2,2) for kurtosis is an acceptable range for being normally distributed-
//        Se pueden añadir tantas variables como se quiera
//		this.points = new double[] { Double.valueOf(parametrosAux.get("CURTOSIS_140_VOLUMEN")),
//				Double.valueOf(parametrosAux.get("SKEWNESS_140_VOLUMEN")) };
//		this.valorClustering = Double.valueOf(parametrosAux.get("CURTOSIS_140_VOLUMEN"))
//				+ Double.valueOf(parametrosAux.get("SKEWNESS_140_VOLUMEN"));
//		this.points = new double[] { Double.valueOf(parametrosAux.get("Market Cap")) };
//		this.valorClustering = Double.valueOf(parametrosAux.get("Market Cap"));
		this.points = new double[] { Double.valueOf(parametrosAux.get("STD_SMA_140_PRECIO")) };
	}

	public double[] getPoint() {
		return points;
	}

	public double[] getPoints() {
		return points;
	}

	public void setPoints(double[] points) {
		this.points = points;
	}

	public File getFichero() {
		return fichero;
	}

	public void setFichero(File fichero) {
		this.fichero = fichero;
	}

	public HashMap<Integer, HashMap<String, String>> getDatosEmpresaEntrada() {
		return datosEmpresaEntrada;
	}

	public void setDatosEmpresaEntrada(HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada) {
		this.datosEmpresaEntrada = datosEmpresaEntrada;
	}

	public Double getValorClustering() {
		return valorClustering;
	}

	public void setValorClustering(Double valorClustering) {
		this.valorClustering = valorClustering;
	}

}