package c10X.brutos;

public class Otros02NasdaqTicker {

	private String symbol = "";
	private String name = "";
	private String lastSale = "";
	private String marketCap = "";
	private String ipoYear = "";
	private String sector = "";
	private String industry = "";
	private String summaryQuote = "";

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
	public Otros02NasdaqTicker(String symbol, String name, String lastSale, String marketCap, String ipoYear,
			String sector, String industry, String summaryQuote) {
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
		return "Otros02NasdaqTicker [symbol=" + symbol + ", name=" + name + ", lastSale=" + lastSale + ", marketCap="
				+ marketCap + ", ipoYear=" + ipoYear + ", sector=" + sector + ", industry=" + industry
				+ ", summaryQuote=" + summaryQuote + "]";
	}

}
