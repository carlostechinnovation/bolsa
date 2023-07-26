package c10X.brutos;

/**
 * Datos estaticos del NASDAQ
 *
 */
public class EstaticoNasdaqModelo implements Comparable<EstaticoNasdaqModelo> {

	public String symbol = "";
	public String name = "";
	public String lastSale = "";
	public String marketCap = "";
	public String ipoYear = "";
	public String sector = "";
	public String industry = "";
	public String summaryQuote = "";
	public Boolean prioritario = false;

	/**
	 * @param symbol
	 * @param name
	 * @param lastSale
	 * @param marketCap
	 * @param ipoYear
	 * @param sector
	 * @param industry
	 * @param summaryQuote
	 */
	public EstaticoNasdaqModelo(String symbol, String name, String lastSale, String marketCap, String ipoYear,
			String sector, String industry, String summaryQuote, int prioritario) {
		super();
		this.symbol = symbol;
		this.name = name;
		this.lastSale = lastSale;
		this.marketCap = marketCap;
		this.ipoYear = ipoYear;
		this.sector = sector;
		this.industry = industry;
		this.summaryQuote = summaryQuote;
		this.prioritario = (prioritario == 1) ? true : false;
	}

	@Override
	public String toString() {
		return "EstaticoNasdaqModelo [symbol=" + symbol + ", name=" + name + ", lastSale=" + lastSale + ", marketCap="
				+ marketCap + ", ipoYear=" + ipoYear + ", sector=" + sector + ", industry=" + industry
				+ ", summaryQuote=" + summaryQuote + ", prioritario=" + prioritario + "]";
	}

	public String getSymbol() {
		return symbol;
	}

	@Override
	public int compareTo(EstaticoNasdaqModelo otro) {

		if (this.symbol.equalsIgnoreCase(BrutosUtils.NASDAQ_REFERENCIA)) { // Prioritario maximo
			return -1;

		} else if (otro.symbol.equalsIgnoreCase(BrutosUtils.NASDAQ_REFERENCIA)) { // Prioritario maximo
			return 1;

		} else if (this.prioritario && !otro.prioritario) {
			return -1;
		} else if (!this.prioritario && otro.prioritario) {
			return 1;
		} else {
			return this.symbol.compareTo(otro.symbol);
		}
	}

}
