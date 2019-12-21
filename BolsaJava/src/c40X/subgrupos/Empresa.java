package c40X.subgrupos;

import java.io.File;
import java.util.HashMap;

public class Empresa {

	File fichero;
	HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada;

	/**
	 * 
	 * @param fichero
	 * @param datosEmpresaEntrada
	 */
	public Empresa(File fichero, HashMap<Integer, HashMap<String, String>> datosEmpresaEntrada) {
		super();
		this.fichero = fichero;
		this.datosEmpresaEntrada = datosEmpresaEntrada;
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

}
