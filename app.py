import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import feedparser
from datetime import datetime
from prophet import Prophet
from bs4 import BeautifulSoup

# --------- UTILITY FUNCTIONS ---------
def format_large_number(n):
    if pd.isnull(n): return "N/A"
    abs_n = abs(n)
    if abs_n >= 1_000_000_000: return f"{n/1_000_000_000:.2f}B"
    elif abs_n >= 1_000_000: return f"{n/1_000_000:.2f}M"
    elif abs_n >= 1_000: return f"{n/1_000:.2f}K"
    return f"{n:.2f}"

def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker

def get_tickers_from_name(name, count=10):
    # Yahoo's API can do fuzzy search, but for simplicity let's use yfinance search module
    try:
        import yfinance
        tickers = yfinance.utils.get_tickers(name)
        return tickers[:count]
    except Exception:
        return []

def get_peers(ticker):
    # This should use sector data, for now a static map
    sector_map = {
        "AAPL": ["MSFT", "GOOGL", "AMZN"],
        "MSFT": ["AAPL", "GOOGL", "AMZN"],
        "GOOGL": ["AAPL", "MSFT", "AMZN"],
        "NVDA": ["AMD", "QCOM", "AVGO"],
        "BNS": ["TD", "RY", "BMO"],
        "SHOP": ["AMZN", "BABA", "SQ"],
        "TD": ["RY", "BNS", "BMO"],
    }
    return sector_map.get(ticker.upper(), ["MSFT", "AAPL"])

def get_peer_comparison(ticker):
    peers = [ticker.upper()] + get_peers(ticker)
    data = []
    for t in peers:
        try:
            info = yf.Ticker(t).info
            name = info.get("shortName") or t
            market_cap = info.get("marketCap")
            pe = info.get("trailingPE")
            pb = info.get("priceToBook")
            div = info.get("dividendYield")
            data.append({
                "Ticker": t,
                "Company": name,
                "Market Cap": format_large_number(market_cap),
                "P/E": f"{pe:.2f}" if pe else "N/A",
                "P/B": f"{pb:.2f}" if pb else "N/A",
                "Dividend Yield": f"{100*div:.2f}%" if div else "N/A",
            })
        except Exception:
            data.append({"Ticker": t, "Company": "N/A", "Market Cap": "N/A", "P/E": "N/A", "P/B": "N/A", "Dividend Yield": "N/A"})
    return pd.DataFrame(data)

def get_news(ticker, count=5):
    headlines = []
    try:
        # Google News Scraping
        url = f"https://news.google.com/search?q={ticker}+stock"
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.select('article h3 a')[:count]:
            text = a.text.strip()
            href = "https://news.google.com" + a['href'][1:]
            headlines.append((text, href))
    except Exception:
        pass
    try:
        # Yahoo Finance RSS
        rss = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US")
        for entry in rss.entries[:count]:
            headlines.append((entry.title, entry.link))
    except Exception:
        pass
    # Deduplicate
    seen = set()
    uniq = []
    for txt, link in headlines:
        if txt not in seen:
            uniq.append((txt, link))
            seen.add(txt)
    return uniq[:count] if uniq else [("No news found.", "")]

def get_sector_news(sector_keywords, count=10):
    results = []
    for keyword in sector_keywords:
        results.extend(get_news(keyword, count=2))
    # Deduplicate and limit
    seen = set()
    uniq = []
    for txt, link in results:
        if txt not in seen:
            uniq.append((txt, link))
            seen.add(txt)
    return uniq[:count]

def get_fda_approvals(count=5):
    try:
        rss_url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
        feed = feedparser.parse(rss_url)
        return [(entry.title, entry.link) for entry in feed.entries[:count]]
    except:
        return [("No FDA data available", "")]

def get_reddit_news(ticker, count=5):
    # Simplified - in production use praw or another Reddit API wrapper (with key)
    return [(f"Reddit post about {ticker} (simulated)", "") for _ in range(count)]

def get_signal_and_rationale(df, info, news):
    if df is None or df.empty:
        return "N/A", "No data available."
    # Simple logic: combine technicals, P/E, and news
    trend = "Bullish" if df['Close'].iloc[-1] > df['Close'].mean() else "Bearish"
    last_pe = info.get("trailingPE", None)
    pe_explain = f"P/E Ratio is {last_pe:.2f}" if last_pe else "No P/E data"
    news_sentiment = "Positive" if any("up" in n[0].lower() for n in news) else "Neutral"
    if trend == "Bullish" and news_sentiment == "Positive":
        return "Buy", f"Price above average; news sentiment is positive. {pe_explain}."
    elif trend == "Bearish" and news_sentiment != "Positive":
        return "Sell", f"Price below average; news not positive. {pe_explain}."
    else:
        return "Hold", f"Mixed signals from price and news. {pe_explain}."

def forecast_prices(df, ticker):
    if df is None or df.empty:
        return None
    data = df[['Close']].copy().reset_index()
    data.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(7)

# --------- SESSION STATE INIT ---------
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = ["AAPL", "MSFT", "NVDA"]

# --------- PAGE NAV ---------
st.set_page_config(page_title="AI-Powered Market Analyzer", layout="wide")
tabs = [
    "Market Information", "Suggestions", "Peer Comparison", "Watchlist", "Financial Health",
    "Options Analytics", "Macro Insights", "News Explorer"
]
page = st.sidebar.radio("Navigation", tabs)

# --------- MARKET INFORMATION TAB ---------
if page == "Market Information":
    st.title("Market Information")
    search_term = st.text_input("Type company name or ticker:", "Apple")
    # Try to resolve to ticker(s)
    ticker_options = []
    if search_term and len(search_term) > 1:
        # Try as ticker first
        ticker_options.append(search_term.upper())
        # Then try fuzzy search for tickers
        try:
            for t in get_tickers_from_name(search_term):
                if t not in ticker_options:
                    ticker_options.append(t)
        except Exception:
            pass
    ticker = st.selectbox("Choose Ticker:", ticker_options, index=0) if ticker_options else "AAPL"

    # Download data and display
    try:
        df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        info = yf.Ticker(ticker).info
        name = get_company_name(ticker)
        st.header(f"{name} ({ticker.upper()})")

        if df is not None and not df.empty and "Close" in df.columns and not df["Close"].empty:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
            st.metric("Market Cap", format_large_number(info.get("marketCap")))
            st.metric("P/E Ratio", f"{info.get('trailingPE'):.3f}" if info.get("trailingPE") else "N/A")
        else:
            st.warning("No price data available for this ticker. Please check the ticker symbol or try another.")
            st.stop()

        # Signal and Rationale
        news = get_news(ticker, count=5)
        signal, rationale = get_signal_and_rationale(df, info, news)
        st.metric("Signal", signal)
        st.write("**Rationale:**", rationale)
        # 7-day forecast
        st.subheader("7-Day Price Trajectory")
        forecast = forecast_prices(df, ticker)
        if forecast is not None and not forecast.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(shape='linear')))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for forecast.")

        # Trading strategy suggestion
        st.subheader("Trading Strategy Suggestion")
        if signal == "Buy":
            st.success("Consider a Market or Limit Buy Order. Stop Loss just below recent support.")
        elif signal == "Sell":
            st.error("Consider a Stop Loss or Put Option. Review for bearish catalysts.")
        else:
            st.info("No strong bias. Hold, or set alerts for volatility.")

        # News Sentiment (deduped)
        st.subheader("News Sentiment")
        for n, url in news:
            if url:
                st.markdown(f"- [{n}]({url})")
            else:
                st.write("-", n)
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# --------- SUGGESTIONS TAB ---------
elif page == "Suggestions":
    st.title("AI Company Suggestions")
    st.write("Latest trending and high-impact companies across sectors...")
    # Frozen robust version, as requested
    sector_map = {
        "AI": ["NVDA", "SMCI", "AMD", "GOOGL", "MSFT", "Baidu"],
        "Tech": ["AAPL", "AVGO", "AMZN", "TSLA", "META", "SHOP"],
        "Defense": ["NOC", "LMT", "BWXT", "PLTR", "RTX", "GD"],
        "Healthcare Tech": ["UNH", "XBI", "TDOC", "CNC"],
        "Pharmaceuticals": ["PFE", "MRNA", "JNJ", "LLY", "BNTX"],
        "Nuclear": ["SMR", "BWXT", "CAMECO", "XOM"],
        "Biotech": ["ARKG", "REGN", "VRTX", "CRSP"],
        "Clean Energy": ["ICLN", "ENPH", "FSLR", "PLUG"],
        "Canadian": ["SHOP", "BNS", "RY", "TD", "ENB", "CNQ"],
    }
    for sector, tickers in sector_map.items():
        st.subheader(f"🔷 {sector} Sector")
        for t in tickers:
            name = get_company_name(t)
            st.markdown(f"**{name} ({t})**")
            news_items = get_news(t, count=2)
            for n, url in news_items:
                if url:
                    st.markdown(f"- [{n}]({url})")
                else:
                    st.write("-", n)
    st.divider()
    st.markdown("### 📰 Sector-wide News")
    for sector, tickers in sector_map.items():
        news = get_sector_news(tickers, count=2)
        st.subheader(f"🗞️ {sector}")
        for n, url in news:
            if url:
                st.markdown(f"- [{n}]({url})")
            else:
                st.write("-", n)
    st.subheader("🧪 FDA Drug Approval News")
    for n, url in get_fda_approvals():
        if url:
            st.markdown(f"- [{n}]({url})")
        else:
            st.write("-", n)

# --------- PEER COMPARISON TAB ---------
elif page == "Peer Comparison":
    st.title("Peer Comparison")
    peer_ticker = st.text_input("Enter Ticker for Peer Comparison:", "AAPL", key="peer_comp_ticker")
    if peer_ticker:
        try:
            comp_df = get_peer_comparison(peer_ticker)
            st.dataframe(comp_df)
            st.info("Compare key metrics: Market Cap, P/E, P/B, Dividend Yield")
        except Exception as e:
            st.error(f"Error: {e}")

# --------- WATCHLIST TAB ---------
elif page == "Watchlist":
    st.title("Watchlist")
    st.write("Add tickers to your watchlist and monitor them here.")
    add_ticker = st.text_input("Add ticker:", "", key="add_ticker")
    if st.button("Add to Watchlist"):
        t = add_ticker.upper().strip()
        if t and t not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(t)
    remove_ticker = st.selectbox("Remove ticker:", st.session_state['watchlist'])
    if st.button("Remove from Watchlist"):
        st.session_state['watchlist'] = [t for t in st.session_state['watchlist'] if t != remove_ticker]
    # Show watchlist
    data = []
    for t in st.session_state['watchlist']:
        info = yf.Ticker(t).info
        data.append({
            "Ticker": t,
            "Price": info.get("currentPrice", "N/A"),
            "Market Cap": format_large_number(info.get("marketCap"))
        })
    st.dataframe(pd.DataFrame(data))

# --------- FINANCIAL HEALTH TAB ---------
elif page == "Financial Health":
    st.title("Financial Health Dashboard")
    ticker = st.text_input("Enter ticker:", "AAPL", key="fh_ticker")
    if ticker:
        try:
            info = yf.Ticker(ticker).info
            st.metric("Current Ratio", f"{info.get('currentRatio', 'N/A')}")
            st.metric("Debt/Equity", f"{info.get('debtToEquity', 'N/A')}")
            st.metric("Return on Equity", f"{info.get('returnOnEquity', 'N/A')}")
        except Exception as e:
            st.error(f"Error: {e}")

# --------- OPTIONS ANALYTICS TAB ---------
elif page == "Options Analytics":
    st.title("Options Analytics")
    ticker = st.text_input("Enter ticker for options analytics:", "AAPL", key="opt_ticker")
    if ticker:
        try:
            tk = yf.Ticker(ticker)
            opts = tk.options
            st.write("Available expiration dates:", opts)
            if opts:
                expiry = st.selectbox("Select expiration date:", opts)
                chain = tk.option_chain(expiry)
                st.write("Calls", chain.calls.head())
                st.write("Puts", chain.puts.head())
                # AI-based suggestion (simple logic)
                st.subheader("AI Trading Suggestion")
                if not chain.calls.empty and chain.calls['impliedVolatility'].mean() > 0.5:
                    st.info("High IV detected: Consider selling covered calls or puts for premium income.")
                elif not chain.calls.empty:
                    st.success("Moderate IV: Consider buying call options for bullish exposure.")
        except Exception as e:
            st.error(f"Error: {e}")

# --------- MACRO INSIGHTS TAB ---------
elif page == "Macro Insights":
    st.title("Macro Insights")
    st.write("Key economic indicators and world indices:")
    # Demo: S&P 500, Nasdaq, FTSE, DAX, Nikkei, Gold, Oil, USD Index
    indices = ["^GSPC", "^IXIC", "^FTSE", "^GDAXI", "^N225", "GC=F", "CL=F", "DX-Y.NYB"]
    data = []
    for idx in indices:
        try:
            df = yf.download(idx, period="5d")
            price = df['Close'].iloc[-1]
            data.append({"Symbol": idx, "Price": f"{price:,.2f}"})
        except:
            data.append({"Symbol": idx, "Price": "N/A"})
    st.dataframe(pd.DataFrame(data))
    st.info("Indices: S&P500 (^GSPC), Nasdaq (^IXIC), FTSE (^FTSE), DAX (^GDAXI), Nikkei (^N225), Gold (GC=F), Oil (CL=F), USD Index (DX-Y.NYB)")

# --------- NEWS EXPLORER TAB ---------
elif page == "News Explorer":
    st.title("News Explorer")
    st.write("Explore the latest financial news from multiple sources:")
    tick = st.text_input("Enter ticker or keyword for news:", "AAPL", key="news_ticker")
    if tick:
        for n, url in get_news(tick, count=10):
            if url:
                st.markdown(f"- [{n}]({url})")
            else:
                st.write("-", n)
    st.subheader("FDA Drug Approval News")
    for n, url in get_fda_approvals():
        if url:
            st.markdown(f"- [{n}]({url})")
        else:
            st.write("-", n)

    st.subheader("Reddit Market News (Simulated)")
    for n, url in get_reddit_news(tick, count=5):
        st.write("-", n)

# --------- END OF APP ---------
