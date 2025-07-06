import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import feedparser
import time

# ------ Helper function: universal company search/ticker resolution ------
@st.cache_data(show_spinner=False)
def search_company(query):
    """Returns a list of (name, ticker, type, exchange) tuples matching the query."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&lang=en"
    try:
        results = requests.get(url, timeout=5).json().get("quotes", [])
        found = []
        for item in results:
            name = item.get("shortname") or item.get("longname") or item.get("name") or ""
            ticker = item.get("symbol") or ""
            exch = item.get("exchange") or ""
            type_ = item.get("quoteType") or ""
            if ticker:
                found.append((name, ticker, type_, exch))
        return found
    except Exception:
        return []

# ------ Data fetchers ------
@st.cache_data(show_spinner=False)
def get_data(ticker, period="1y"):
    try:
        ticker = ticker.upper().strip()
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data.empty or "Close" not in data.columns:
            raise ValueError("No valid data found for ticker.")
        return data
    except Exception as e:
        raise ValueError(f"Failed to fetch data for ticker: {str(e)}")

def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except:
        return ticker

def get_financials(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        fin = ticker_obj.financials
        bal = ticker_obj.balance_sheet
        cf = ticker_obj.cashflow
        return fin, bal, cf
    except:
        return None, None, None

def get_earnings_calendar(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        return cal.transpose() if not cal.empty else None
    except:
        return None

# ------ Technicals & Analytics ------
def calculate_indicators(df):
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def generate_signals(df):
    df["Trend"] = np.where(df["SMA_50"] > df["SMA_200"], 1, -1)
    df["Buy"] = (df["RSI"] < 30) & (df["MACD"] > df["Signal_Line"]) & (df["Trend"] == 1)
    df["Sell"] = (df["RSI"] > 70) & (df["MACD"] < df["Signal_Line"]) & (df["Trend"] == -1)
    df["Signal"] = np.where(df["Buy"], "Buy", np.where(df["Sell"], "Sell", "Hold"))
    return df

# ------ News & Sentiment ------
def get_news_sources(ticker_or_name):
    """
    Aggregates news from Google News, Yahoo Finance, Reuters, MarketWatch, CNBC, SeekingAlpha, Finviz, Reddit, FDA, SEDAR, EDGAR.
    """
    news = []
    # Google News
    try:
        url = f"https://news.google.com/search?q={ticker_or_name}"
        html = requests.get(url, timeout=7).text
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("article h3 a")[:4]:
            headline = a.text.strip()
            link = "https://news.google.com" + a["href"][1:] if a["href"].startswith(".") else a["href"]
            news.append((headline, link, "Google News"))
    except Exception:
        pass
    # Yahoo Finance RSS
    try:
        feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_or_name}&region=US&lang=en-US")
        for entry in feed.entries[:4]:
            news.append((entry.title, entry.link, "Yahoo Finance"))
    except Exception:
        pass
    # Reuters
    try:
        url = f"https://www.reuters.com/companies/{ticker_or_name}/news"
        html = requests.get(url, timeout=7).text
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("a.StoryCollection__story__headline__2vP6H")[:2]:
            headline = a.text.strip()
            link = "https://www.reuters.com" + a["href"] if a["href"].startswith("/") else a["href"]
            news.append((headline, link, "Reuters"))
    except Exception:
        pass
    # MarketWatch
    try:
        url = f"https://www.marketwatch.com/investing/stock/{ticker_or_name.lower()}"
        html = requests.get(url, timeout=7).text
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("div.element.element--article article h3 a")[:2]:
            headline = a.text.strip()
            link = a["href"]
            news.append((headline, link, "MarketWatch"))
    except Exception:
        pass
    # Seeking Alpha
    try:
        url = f"https://seekingalpha.com/symbol/{ticker_or_name}"
        html = requests.get(url, timeout=7).text
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("a.symbol_article")[:2]:
            headline = a.text.strip()
            link = "https://seekingalpha.com" + a["href"] if a["href"].startswith("/") else a["href"]
            news.append((headline, link, "SeekingAlpha"))
    except Exception:
        pass
    # Finviz
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker_or_name}"
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=7).text
        soup = BeautifulSoup(html, "html.parser")
        for td in soup.select("td.fullview-news-outer a")[:2]:
            headline = td.text.strip()
            link = td["href"]
            news.append((headline, link, "Finviz"))
    except Exception:
        pass
    # Reddit
    try:
        feed = feedparser.parse(f"https://www.reddit.com/r/stocks/search.rss?q={ticker_or_name}&sort=new&t=month&restrict_sr=on")
        for entry in feed.entries[:2]:
            news.append((entry.title, entry.link, "Reddit"))
    except Exception:
        pass
    # FDA (for pharma/health)
    try:
        fda_feed = feedparser.parse("https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml")
        for entry in fda_feed.entries[:2]:
            if ticker_or_name.lower() in entry.title.lower() or ticker_or_name.lower() in entry.summary.lower():
                news.append((entry.title, entry.link, "FDA"))
    except Exception:
        pass
    # SEDAR (Canada filings)
    if ticker_or_name.endswith(".TO") or ticker_or_name.endswith(".V"):
        sedar_link = f"https://www.sedarplus.ca/search/company/{ticker_or_name.split('.')[0]}"
        news.append((f"SEDAR filings for {ticker_or_name}", sedar_link, "SEDAR"))
    # EDGAR (US filings)
    if "." not in ticker_or_name and len(ticker_or_name) < 8:
        edgar_link = f"https://www.sec.gov/edgar/browse/?CIK={ticker_or_name}&owner=exclude"
        news.append((f"EDGAR filings for {ticker_or_name}", edgar_link, "EDGAR"))

    # Deduplicate by headline
    deduped = []
    seen = set()
    for (h, l, src) in news:
        if h not in seen:
            deduped.append((h, l, src))
            seen.add(h)
    return deduped[:10]

def summarize_sentiment(news):
    # Simple rule-based: +1 for positive, -1 for negative, else 0
    pos_kw = ["beat", "record", "growth", "up", "surge", "strong", "upgrade", "outperform"]
    neg_kw = ["miss", "loss", "decline", "down", "weak", "downgrade", "underperform"]
    score = 0
    for (headline, _, _) in news:
        headline_l = headline.lower()
        if any(kw in headline_l for kw in pos_kw):
            score += 1
        if any(kw in headline_l for kw in neg_kw):
            score -= 1
    if score > 1:
        return "Positive 🟢"
    elif score < -1:
        return "Negative 🔴"
    else:
        return "Mixed/Neutral 🟡"

# ------ Forecasting ------
def forecast_prices(df, ticker):
    df = df[["Close"]].copy().reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    # For now, we don't add regressors to Prophet for free news/market data
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].tail(7)

# ------ Layout ------
st.set_page_config(page_title="AI-Powered Market Analyzer", layout="wide")
st.title("🌏 Market Analyzer Terminal")

tabs = st.tabs([
    "Market Information",
    "Suggestions",
    "Peer Comparison",
    "Watchlist",
    "Financial Health",
    "Options Analytics",
    "Macro Insights",
    "News Explorer"
])

# ------ Market Information ------
with tabs[0]:
    st.header("Market Information")
    # Company/ticker search with autocomplete
    search = st.text_input("🔍 Search company name or ticker:", value="Apple")
    search_results = []
    ticker = None
    if search:
        search_results = search_company(search)
        if search_results:
            names = [f"{name} ({ticker}) [{exch}]" for (name, ticker, type_, exch) in search_results]
            selected = st.selectbox("Select company/ticker:", names)
            idx = names.index(selected)
            ticker = search_results[idx][1]
        else:
            st.warning("No matches found.")
    if ticker:
        name = get_company_name(ticker)
        try:
            df = get_data(ticker)
            df = calculate_indicators(df)
            df = generate_signals(df)
            forecast = forecast_prices(df, ticker)
            news_items = get_news_sources(ticker)
            sentiment = summarize_sentiment(news_items)

            # --- Display
            st.subheader(f"{name} ({ticker})")
            st.metric("Current Price", f"${float(df['Close'].iloc[-1]):.2f}")
            st.metric("Signal", df["Signal"].iloc[-1])
            st.metric("Trend", "📈 Bullish" if df["Trend"].iloc[-1] == 1 else "📉 Bearish")
            st.metric("News Sentiment", sentiment)
            with st.expander("📰 News (click to expand)", expanded=True):
                for h, l, src in news_items:
                    st.markdown(f"- [{h}]({l}) <sub>({src})</sub>", unsafe_allow_html=True)
            st.markdown("#### 7-Day Price Forecast")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="7-Day Forecast", line=dict(shape="linear")))
            st.plotly_chart(fig, use_container_width=True)

            # --- Rationale explanation
            st.markdown("**Signal Rationale:**")
            rationale = []
            if df["Signal"].iloc[-1] == "Buy":
                if df["RSI"].iloc[-1] < 30 and df["MACD"].iloc[-1] > df["Signal_Line"].iloc[-1] and df["Trend"].iloc[-1] == 1:
                    rationale.append("Oversold RSI, bullish MACD crossover, and uptrend detected.")
                if sentiment.startswith("Positive"):
                    rationale.append("Recent news is positive.")
            elif df["Signal"].iloc[-1] == "Sell":
                if df["RSI"].iloc[-1] > 70 and df["MACD"].iloc[-1] < df["Signal_Line"].iloc[-1] and df["Trend"].iloc[-1] == -1:
                    rationale.append("Overbought RSI, bearish MACD crossover, and downtrend detected.")
                if sentiment.startswith("Negative"):
                    rationale.append("Recent news is negative.")
            else:
                rationale.append("Mixed technicals and neutral news.")

            # Compare forecast
            if (df["Signal"].iloc[-1] == "Buy" and forecast["yhat"].iloc[-1] < df['Close'].iloc[-1]):
                rationale.append("Note: Despite buy signal, 7-day forecast is flat/downward.")
            elif (df["Signal"].iloc[-1] == "Sell" and forecast["yhat"].iloc[-1] > df['Close'].iloc[-1]):
                rationale.append("Note: Despite sell signal, 7-day forecast is upward.")

            st.info(" ".join(rationale))

            # --- Trading Strategy Suggestion
            st.markdown("**Suggested Trading Strategy:**")
            strat = ""
            if df["Signal"].iloc[-1] == "Buy":
                strat = "Market or limit order buy; consider protective stop loss below support. For options: long call or bull call spread if volatility is reasonable."
            elif df["Signal"].iloc[-1] == "Sell":
                strat = "Sell/short; consider stop buy to limit risk. For options: long put or bear put spread."
            else:
                strat = "Hold, or consider neutral options (e.g., iron condor) if expecting rangebound movement."
            st.success(strat)

        except Exception as e:
            st.error(f"Error: {e}")

# ------ Suggestions ------
with tabs[1]:
    st.header("AI Suggested Companies to Watch")
    # Sectors and top names (static demo; can be enhanced with scraping)
    ai_suggestions = {
        "AI & Tech": [
            ("Apple Inc.", "AAPL"),
            ("NVIDIA Corp.", "NVDA"),
            ("Super Micro Computer", "SMCI"),
            ("Microsoft", "MSFT"),
            ("Google (Alphabet)", "GOOGL"),
            ("Broadcom", "AVGO"),
            ("Shopify", "SHOP.TO"),
            ("OpenText", "OTEX.TO"),
            ("BlackBerry", "BB.TO"),
        ],
        "Defense & Aerospace": [
            ("Northrop Grumman", "NOC"),
            ("Lockheed Martin", "LMT"),
            ("Magellan Aerospace", "MAL.TO"),
            ("CAE Inc.", "CAE.TO"),
            ("Palantir", "PLTR"),
            ("BWX Technologies", "BWXT"),
        ],
        "Nuclear & Clean Energy": [
            ("Cameco", "CCO.TO"),
            ("BWX Technologies", "BWXT"),
            ("Constellation Energy", "CEG"),
            ("NextEra Energy", "NEE"),
            ("ICLN ETF", "ICLN"),
        ],
        "Healthcare & Pharma": [
            ("Pfizer", "PFE"),
            ("Moderna", "MRNA"),
            ("Knight Therapeutics", "GUD.TO"),
            ("WELL Health Tech", "WELL.TO"),
            ("Sienna Senior Living", "SIA.TO"),
            ("XBI ETF", "XBI"),
        ],
        "Financials": [
            ("Royal Bank of Canada", "RY.TO"),
            ("TD Bank", "TD.TO"),
            ("Scotiabank", "BNS.TO"),
            ("CIBC", "CM.TO"),
            ("Bank of Montreal", "BMO.TO"),
        ],
        "Biotech": [
            ("ARK Genomic Revolution ETF", "ARKG"),
            ("BioNTech", "BNTX"),
            ("Gilead Sciences", "GILD"),
        ],
        "Other Global Leaders": [
            ("Tesla", "TSLA"),
            ("Amazon", "AMZN"),
            ("Enbridge", "ENB.TO"),
            ("Suncor", "SU.TO"),
            ("Canadian National Railway", "CNR.TO"),
            ("iShares TSX 60 ETF", "XIU.TO"),
        ]
    }

    for sector, companies in ai_suggestions.items():
        st.subheader(sector)
        cols = st.columns(2)
        for i, (name, ticker) in enumerate(companies):
            with cols[i % 2]:
                st.markdown(f"**{name}** (`{ticker}`)")
                news_items = get_news_sources(ticker)
                if news_items:
                    for h, l, src in news_items[:3]:
                        st.markdown(f"- [{h}]({l}) <sub>({src})</sub>", unsafe_allow_html=True)
                else:
                    st.write("No news available.")
        st.divider()

    st.header("Sector & Market News")
    # Broader sector-wide news aggregation
    sectors = [
        "technology", "artificial intelligence", "nuclear", "defense", "pharmaceutical", "healthcare",
        "aerospace", "energy", "mining", "banking", "transportation"
    ]
    for s in sectors:
        st.markdown(f"#### {s.title()} News")
        news_items = get_news_sources(s)
        if news_items:
            for h, l, src in news_items[:3]:
                st.markdown(f"- [{h}]({l}) <sub>({src})</sub>", unsafe_allow_html=True)
        else:
            st.write("No news found for this sector.")
        st.divider()

# ------ Peer Comparison ------
with tabs[2]:
    st.header("Peer Comparison")
    peer_search = st.text_input("Search/add peer company or ticker:", value="NVIDIA")
    peer_results = []
    if peer_search:
        peer_results = search_company(peer_search)
    selected_peers = st.multiselect(
        "Select companies/tickers to compare:",
        options=[f"{name} ({ticker}) [{exch}]" for (name, ticker, type_, exch) in peer_results] if peer_results else [],
        default=[],
    )
    tickers = [r.split("(")[-1].split(")")[0] for r in selected_peers]
    if tickers:
        dfs = []
        for t in tickers:
            try:
                d = get_data(t, period="1y")
                d = calculate_indicators(d)
                dfs.append((t, d))
            except Exception as e:
                st.warning(f"{t}: {e}")
        if dfs:
            fig = go.Figure()
            for t, d in dfs:
                fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name=t))
            st.plotly_chart(fig, use_container_width=True)

# ------ Watchlist ------
with tabs[3]:
    st.header("Watchlist (Session Only)")
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []
    # Add by name or ticker
    add = st.text_input("Add to watchlist by name or ticker:", value="Tesla")
    add_results = []
    if add:
        add_results = search_company(add)
        if add_results:
            add_select = st.selectbox("Select to add:", [f"{name} ({ticker}) [{exch}]" for (name, ticker, type_, exch) in add_results])
            ticker_add = add_results[[f"{name} ({ticker}) [{exch}]" for (name, ticker, type_, exch) in add_results].index(add_select)][1]
            if st.button(f"Add {ticker_add} to Watchlist"):
                if ticker_add not in st.session_state["watchlist"]:
                    st.session_state["watchlist"].append(ticker_add)
    st.markdown("#### Your Watchlist")
    for w in st.session_state["watchlist"]:
        col1, col2 = st.columns([4,1])
        with col1:
            st.write(f"{get_company_name(w)} ({w})")
        with col2:
            if st.button(f"Remove {w}", key=w):
                st.session_state["watchlist"].remove(w)
    if st.session_state["watchlist"]:
        for w in st.session_state["watchlist"]:
            try:
                d = get_data(w, period="1mo")
                d = calculate_indicators(d)
                st.line_chart(d["Close"], height=100)
                news = get_news_sources(w)
                st.caption(" ".join([f"{h} ({src})" for h, _, src in news[:2]]))
            except:
                pass

# ------ Financial Health ------
with tabs[4]:
    st.header("Financial Health")
    search = st.text_input("Company or ticker:", value="Royal Bank of Canada")
    search_results = []
    ticker = None
    if search:
        search_results = search_company(search)
        if search_results:
            names = [f"{name} ({ticker}) [{exch}]" for (name, ticker, type_, exch) in search_results]
            selected = st.selectbox("Select company/ticker:", names)
            idx = names.index(selected)
            ticker = search_results[idx][1]
    if ticker:
        fin, bal, cf = get_financials(ticker)
        st.subheader(f"Financials for {get_company_name(ticker)} ({ticker})")
        if fin is not None and not fin.empty:
            st.markdown("**Income Statement**")
            st.dataframe(fin)
        if bal is not None and not bal.empty:
            st.markdown("**Balance Sheet**")
            st.dataframe(bal)
        if cf is not None and not cf.empty:
            st.markdown("**Cash Flow Statement**")
            st.dataframe(cf)
        # Line chart for yearly revenue/profit trends
        try:
            income_df = fin.T if fin is not None else None
            if income_df is not None and not income_df.empty:
                chart_data = pd.DataFrame()
                for field in ["Total Revenue", "Gross Profit", "EBITDA", "Net Income"]:
                    if field in income_df.columns:
                        chart_data[field] = income_df[field]
                if not chart_data.empty:
                    chart_data = chart_data.sort_index()
                    chart_data.index = chart_data.index.astype(str)
                    fig = go.Figure()
                    for col in chart_data.columns:
                        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data[col], mode="lines+markers", name=col))
                    fig.update_layout(
                        xaxis_title="Year",
                        yaxis_title="Amount (USD or CAD)",
                        title="Financial Trends",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Chart not available: {e}")

# ------ Options Analytics ------
with tabs[5]:
    st.header("Options Analytics")
    search = st.text_input("Optionable company/ticker:", value="MSFT")
    search_results = []
    ticker = None
    if search:
        search_results = search_company(search)
        if search_results:
            names = [f"{name} ({ticker}) [{exch}]" for (name, ticker, type_, exch) in search_results]
            selected = st.selectbox("Select:", names)
            idx = names.index(selected)
            ticker = search_results[idx][1]
    if ticker:
        try:
            tkr = yf.Ticker(ticker)
            expiries = tkr.options
            st.write(f"Available Expiries: {', '.join(expiries)}")
            selected_expiry = st.selectbox("Choose expiry:", expiries)
            opt_chain = tkr.option_chain(selected_expiry)
            st.markdown("**Calls**")
            st.dataframe(opt_chain.calls)
            st.markdown("**Puts**")
            st.dataframe(opt_chain.puts)
            st.markdown("**Strategy Suggestion:**")
            if len(opt_chain.calls) > 0 and len(opt_chain.puts) > 0:
                st.info("Bullish: Buy calls or bull call spread. Bearish: Buy puts or bear put spread. Neutral: Iron condor/strangle.")
        except Exception as e:
            st.warning(f"Could not load options data: {e}")

# ------ Macro Insights ------
with tabs[6]:
    st.header("Macro Insights")
    try:
        macro_tickers = ["^GSPC", "^TSX", "^VIX", "SPY", "XIU.TO", "QQQ", "GLD", "USDCAD=X"]
        dfs = {}
        for m in macro_tickers:
            dfs[m] = get_data(m, period="1mo")
        cols = st.columns(4)
        for i, m in enumerate(macro_tickers):
            with cols[i % 4]:
                st.metric(m, f"{dfs[m]['Close'].iloc[-1]:.2f}")
                st.line_chart(dfs[m]["Close"], height=100)
    except:
        st.info("Some macro data not available.")

# ------ News Explorer ------
with tabs[7]:
    st.header("News Explorer")
    search = st.text_input("Sector/keyword or company/ticker:", value="Nuclear")
    if search:
        news_items = get_news_sources(search)
        sentiment = summarize_sentiment(news_items)
        st.metric("Sentiment", sentiment)
        if news_items:
            for h, l, src in news_items:
                st.markdown(f"- [{h}]({l}) <sub>({src})</sub>", unsafe_allow_html=True)
        else:
            st.write("No news found.")

# ------ End of script ------
