# Global AI Market Watcher

An advanced Streamlit dashboard for worldwide stocks, ETFs, crypto, forex, and news—no paid feeds or keys required.

## Features

- Global company/ticker search & selection on every tab
- Market Information: Price, signal, trend, news, 7-day forecast (Prophet)
- AI-driven Suggestions by sector/industry (editable list)
- Peer Comparison, Watchlist, Financial Health, Options Analytics
- Real-time news (Google News, Yahoo Finance RSS, FDA)
- Modular—add/remove tabs as needed

## Setup

1. Clone repo
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run:
    ```bash
    streamlit run app.py
    ```

## Notes

- No authentication (add if deploying publicly)
- Handles global markets, Canadian tickers (SHOP.TO, RY.TO), UK, Asia, etc.
- Suggestions tab sector companies can be edited as needed

## License

MIT or your preferred license.
