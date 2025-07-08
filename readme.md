# AI-Powered Market Analyzer

A Bloomberg-style Streamlit dashboard for live global market analysis, stock/ETF/crypto/FX/futures tracking, news aggregation, peer comparison, financial health, options analytics, macro trends, and more—no paid data feeds required.

## Features

- **Market Information:**  
  Enter company name or ticker, auto-resolves to correct symbol. Get price, signal (buy/hold/sell), rationale, 7-day forecast, trading strategy, and real-time news.
- **Suggestions:**  
  AI-curated companies/sectors to watch (AI, Tech, Nuclear, Healthcare, Pharma, Clean Energy, Defense, Canadian, etc.) with supporting news and FDA feeds.
- **Peer Comparison:**  
  Compare tickers against sector peers.
- **Watchlist:**  
  Add/remove tickers and track live prices/market cap.
- **Financial Health:**  
  See key financial ratios for any ticker.
- **Options Analytics:**  
  Explore available option chains and get AI-based suggestions.
- **Macro Insights:**  
  Monitor major indices, commodities, rates, and more.
- **News Explorer:**  
  Scrape and aggregate news from Google News, Yahoo Finance RSS, FDA, Reddit/StockTwits for any ticker, sector, or keyword.

## Installation

1. Clone/download this repo and move to the directory.
2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:

    ```bash
    streamlit run app.py
    ```

## Usage

- Use the sidebar to navigate between tabs.
- Type company names or tickers in input boxes (auto-resolves tickers for most public companies worldwide, including Canadian/US/ETFs/crypto).
- Add or remove tickers from your Watchlist tab.
- All news headlines are clickable links. News, signals, and forecasts update automatically.
- No paid market data feeds or API keys required.

## Notes

- For optimal performance, use Python 3.10+.
- No user authentication—if you deploy publicly, add security as needed.
- The Suggestions tab is frozen (as per client spec) to preserve AI picks and sector coverage.
- All features are modular: you can add, remove, or customize tabs as needed.
- If you see "No news found", the ticker or sector may have limited English news coverage at this time.

## License

MIT (or your preferred open-source license)

---

