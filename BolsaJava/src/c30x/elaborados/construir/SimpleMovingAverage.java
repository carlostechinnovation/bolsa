package c30x.elaborados.construir;

import java.util.ArrayList;

public class SimpleMovingAverage {

	// queue used to store list so that we get the average
	private ArrayList<Float> dataset = new ArrayList<Float>();
	private float sum;

	public SimpleMovingAverage() {
	}

	public void addData(float num) {
		dataset.add(num);

	}

	public String getDataToString() {
		String devuelto = "";
		for (int x = 0; x < dataset.size(); x++) {
			devuelto += dataset.get(x).toString() + ", ";
		}
		return devuelto;

	}

	// function to calculate mean
	public float getMean(Integer tamanoObligatorio) throws Exception {
		sum = 0;
		for (Float valor : dataset) {
			sum += valor;
		}
		if (tamanoObligatorio != dataset.size())
			throw new Exception("Conjunto de datos NO válido para realizar el cálculo");
		return sum / dataset.size();
	}

}
