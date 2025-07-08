# Global AI Market Terminal

**A robust, AI-powered market analysis dashboard**  
Supports global stocks, ETFs, crypto, FOREX, and more using only free, non-subscription APIs and data sources.

---

## Features

- **Global Ticker Support:** Search by company name or ticker (auto-resolves from a CSV file of 25,000+ world tickers)
- **Market Information:** Live prices, price charts, signals, trend rationale, and news for any supported ticker
- **Suggestions:** News, events, and AI-based sector coverage for stocks, AI, energy, pharma, defense, tech, and more
- **Peer Comparison:** Find sector peers for most S&P 500 and many global stocks
- **Watchlist:** Add/remove companies for quick tracking
- **Financial Health:** View company financials, ratios, and statements
- **Options Analytics:** View options expiry dates (for US tickers only)
- **Macro Insights:** Global macro news from leading free sources
- **News Explorer:** Search latest market news for any company, asset, or theme

---

## Quick Start

1. **Clone this repo** or copy all files to your working directory.

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Download the global tickers file:**  
   - Get a world tickers CSV:  
     [Sample AlphaVantage CSV (free, requires free API key)](https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo)
   - Save/rename it as `global_tickers.csv` in the same directory as `app.py`.

4. **Launch the app:**
    ```
    streamlit run app.py
    ```

---

## File Structure

- `app.py` — main Streamlit dashboard
- `requirements.txt` — dependencies
- `global_tickers.csv` — list of 25k+ company tickers (keep up to date for best results)
- `readme.md` — this file

---

## Notes

- **Free data only:** This app uses Yahoo Finance, Google News, and other free APIs. Some features may be limited for obscure or delisted tickers.
- **Options data:** Options analytics are only for US tickers, due to data availability.
- **Ticker lookup:** If a company isn't found, update `global_tickers.csv` with the correct symbol/name.

---

## FAQ

**Q: The app can't find my ticker/company!**  
A: Update or expand `global_tickers.csv` for better coverage, or check if the ticker is delisted.

**Q: Why do some tabs say "no data"?**  
A: Some features are limited by free API coverage. The app will always handle gracefully, with no crashes.

**Q: How do I add more tickers?**  
A: Simply append rows to `global_tickers.csv` with `Symbol,Name,Exchange` (at minimum Symbol and Name).

---

**Enjoy your global market dashboard!**

