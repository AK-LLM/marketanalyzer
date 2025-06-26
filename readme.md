# Global Market Watcher

A free, plug-and-play Streamlit app for global stocks, ETFs, crypto, FOREX, and commodities.  
AI-powered sector suggestions, company headlines, buy/sell/hold logic, and Prophet-based 7-day price forecast.

## Features

- Any ticker, worldwide: Stocks, ETFs, Crypto, FOREX, Commodities
- Auto company name match, clean dashboard
- RSI, MACD, technical analysis and "Buy/Hold/Sell" logic
- 7-day Prophet price forecast chart
- Trading strategy suggestions (Market, Limit, Stop, Call/Put Options)
- Sector-based global suggestions: AI, Nuclear, Tech, Healthcare, Defense, Pharma
- Real-time news scraping for every suggestion (Google News, Yahoo, FDA)
- Dividers and clear layout, mobile-friendly

## How to Run

1. **Clone this repo**:
    ```
    git clone https://github.com/yourusername/market-watcher.git
    cd market-watcher
    ```
2. **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```
3. **Start the app**:
    ```
    streamlit run app.py
    ```

## Cloud Deployment

- Works on [Streamlit Community Cloud](https://streamlit.io/cloud) for free.
- No database required for basic features!

## To Do

- Social scraping (Reddit/X) with official APIs (add in future)
- User accounts, notification triggers (add Supabase or DB backend if needed)
- Smarter AI model integrations

---

_Questions or enhancements? PRs and issues welcome!_
