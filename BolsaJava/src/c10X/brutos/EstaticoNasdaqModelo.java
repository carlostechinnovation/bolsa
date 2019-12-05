package c10X.brutos;

/**
 * Datos estáticos del NASDAQ
 *
 */
public class EstaticoNasdaqModelo {

	public String symbol = "";
	public String name = "";
	public String lastSale = "";
	public String marketCap = "";
	public String ipoYear = "";
	public String sector = "";
	public String industry = "";
	public String summaryQuote = "";

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
	public EstaticoNasdaqModelo(String symbol, String name, String lastSale, String marketCap, String ipoYear, String sector,
			String industry, String summaryQuote) {
		super();
		this.symbol = symbol;
		this.name = name;
		this.lastSale = lastSale;
		this.marketCap = marketCap;
		this.ipoYear = ipoYear;
		this.sector = sector;
		this.industry = industry;
		this.summaryQuote = summaryQuote;
	}

	@Override
	public String toString() {
		return "EstaticosNasdaq [symbol=" + symbol + ", name=" + name + ", lastSale=" + lastSale + ", marketCap="
				+ marketCap + ", ipoYear=" + ipoYear + ", sector=" + sector + ", industry=" + industry
				+ ", summaryQuote=" + summaryQuote + "]";
	}

}
