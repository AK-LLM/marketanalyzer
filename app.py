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

st.set_page_config(page_title="AI-Powered Market Analyzer", layout="wide")

# ========== UTILS ==========

@st.cache_data
def get_data(ticker, period='1y'):
    ticker = ticker.upper().strip()
    if '-' in ticker and not ticker.endswith('-USD'):
        ticker += '-USD'
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty or 'Close' not in data.columns:
        raise ValueError("No valid data found for ticker.")
    return data

def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('longName') or info.get('shortName') or ticker
    except:
        return ticker

def calculate_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def generate_signals(df):
    df['Trend'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    df['Buy'] = (df['RSI'] < 30) & (df['MACD'] > df['Signal_Line']) & (df['Trend'] == 1)
    df['Sell'] = (df['RSI'] > 70) & (df['MACD'] < df['Signal_Line']) & (df['Trend'] == -1)
    df['Signal'] = np.where(df['Buy'], 'Buy', np.where(df['Sell'], 'Sell', 'Hold'))
    return df

def signal_rationale(signal, trend, df):
    reasons = []
    if signal == "Buy":
        reasons.append("RSI below 30 (oversold), positive MACD crossover, and short-term average above long-term.")
    elif signal == "Sell":
        reasons.append("RSI above 70 (overbought), negative MACD crossover, and short-term average below long-term.")
    else:
        reasons.append("No strong buy/sell triggers: technicals indicate sideways/consolidation.")
    if trend == 1:
        reasons.append("Short-term average (SMA50) above long-term (SMA200): bullish underlying trend.")
    else:
        reasons.append("Short-term average (SMA50) below long-term (SMA200): bearish trend dominates.")
    if not df.empty:
        reasons.append(f"Latest RSI: {df['RSI'].iloc[-1]:.2f}, MACD: {df['MACD'].iloc[-1]:.2f}")
    return " ".join(reasons)

def get_news(ticker):
    try:
        url = f"https://news.google.com/search?q={ticker}+stock"
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, 'html.parser')
        return [item.text for item in soup.select('article h3 a')][:5]
    except:
        return ["No news found."]

def get_news_sources(keyword):
    news = []
    # Google News
    try:
        url = f"https://news.google.com/search?q={keyword}"
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.select('article h3 a'):
            headline = a.text
            link = "https://news.google.com" + a['href'][1:] if a['href'].startswith('.') else a['href']
            news.append((headline, link, "Google News"))
    except: pass
    # Yahoo Finance RSS
    try:
        feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={keyword}&region=US&lang=en-US")
        for entry in feed.entries[:3]:
            news.append((entry.title, entry.link, "Yahoo Finance"))
    except: pass
    # Reuters
    try:
        feed = feedparser.parse(f"https://feeds.reuters.com/reuters/{keyword.lower()}News")
        for entry in feed.entries[:2]:
            news.append((entry.title, entry.link, "Reuters"))
    except: pass
    return news

def get_fda_approvals():
    try:
        rss_url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
        feed = feedparser.parse(rss_url)
        return [(entry.title, entry.link, "FDA") for entry in feed.entries[:5]]
    except:
        return []

def forecast_prices(df, ticker):
    hist = df[['Close']].copy().reset_index()
    hist.columns = ['ds', 'y']
    hist['ds'] = pd.to_datetime(hist['ds'])
    hist = hist.drop_duplicates(subset='ds')
    hist = hist.dropna()
    if len(hist) < 30:
        forecast_df = pd.DataFrame({"ds": pd.date_range(datetime.today(), periods=7), "yhat": [hist['y'].iloc[-1]]*7})
        return forecast_df
    model = Prophet(daily_seasonality=True)
    model.fit(hist)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(7)

def get_peers(ticker):
    """Get tickers in the same sector for peer comparison."""
    try:
        sector = yf.Ticker(ticker).info.get('sector')
        sp500 = yf.Tickers("^GSPC").tickers
        peers = []
        for peer in sp500:
            try:
                info = yf.Ticker(peer).info
                if info.get('sector') == sector and peer != ticker:
                    peers.append(peer)
            except:
                continue
        return peers[:5]
    except:
        return []

def get_basic_financials(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "Market Cap": info.get("marketCap"),
            "P/E Ratio": info.get("trailingPE"),
            "EPS": info.get("trailingEps"),
            "Dividend Yield": info.get("dividendYield"),
            "Profit Margin": info.get("profitMargins"),
            "Revenue": info.get("totalRevenue"),
            "Gross Profit": info.get("grossProfits"),
            "EBITDA": info.get("ebitda"),
            "Net Income": info.get("netIncomeToCommon"),
        }
    except:
        return {}

def get_balance_sheet(ticker):
    try:
        return yf.Ticker(ticker).balance_sheet
    except:
        return pd.DataFrame()

def get_options(ticker):
    try:
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return pd.DataFrame(), None
        calls = t.option_chain(expiries[0]).calls
        puts = t.option_chain(expiries[0]).puts
        return calls, puts
    except:
        return pd.DataFrame(), pd.DataFrame()

def get_macro_data():
    # Fetch data for S&P 500, Dow, Nasdaq, TSX, FTSE, DAX, gold, oil, EUR/USD, BTC
    indices = {
        "S&P 500": "^GSPC", "Dow": "^DJI", "Nasdaq": "^IXIC",
        "TSX": "^GSPTSE", "FTSE": "^FTSE", "DAX": "^GDAXI",
        "Gold": "GC=F", "Oil": "CL=F", "EUR/USD": "EURUSD=X", "BTC/USD": "BTC-USD"
    }
    df = {}
    for k, v in indices.items():
        try:
            val = yf.Ticker(v).history(period='1d')['Close'].iloc[-1]
        except:
            val = np.nan
        df[k] = val
    return df

# ========== MAIN APP ==========

tabs = st.tabs([
    "Market Information", "Suggestions", "Peer Comparison", "Watchlist",
    "Financial Health", "Options Analytics", "Macro Insights", "News Explorer"
])

# ------ Market Information ------
with tabs[0]:
    st.header("Market Information")
    ticker_input = st.text_input("Enter any ticker (Stock, ETF, Crypto, FOREX, Commodity):", "AAPL")
    if ticker_input:
        try:
            df = get_data(ticker_input)
            df = calculate_indicators(df)
            df = generate_signals(df)
            name = get_company_name(ticker_input)
            signal = str(df['Signal'].iloc[-1]) if not df['Signal'].empty else "N/A"
            trend = "Bullish" if df['Trend'].iloc[-1] == 1 else "Bearish"
            st.subheader(f"{name} ({ticker_input.upper()})")
            st.metric("Signal", signal)
            st.metric("Trend", trend)
            st.caption(signal_rationale(signal, df['Trend'].iloc[-1], df))
            st.subheader("7-Day Price Forecast")
            forecast = forecast_prices(df.copy(), ticker_input)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode="lines+markers", name="Forecast"))
            fig.update_layout(xaxis_title="Date", yaxis_title="Forecast Price")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("News Sentiment")
            news = get_news_sources(ticker_input)
            if news:
                for h, l, src in news:
                    st.markdown(f"- [{h}]({l}) <sub>({src})</sub>", unsafe_allow_html=True)
            else:
                st.write("No news found.")
        except Exception as e:
            st.error(f"Error: {e}")

# ------ Suggestions ------
with tabs[1]:
    st.header("AI Suggested Companies to Watch")
    sectors = {
        "AI": ["NVDA", "MSFT", "GOOGL", "AMD", "SMCI"],
        "Tech": ["AAPL", "AMZN", "AVGO", "TSLA", "META"],
        "Defense": ["NOC", "LMT", "BWXT", "PLTR", "RTX"],
        "Healthcare Tech": ["UNH", "XBI", "TDOC", "DXCM"],
        "Pharmaceuticals": ["PFE", "MRNA", "JNJ", "LLY", "BNTX"],
        "Nuclear": ["SMR", "BWXT", "LEU"],
        "Biotech": ["ARKG", "CRSP", "NTLA"],
        "Clean Energy": ["ICLN", "ENPH", "SEDG"],
        "Canadian AI/Tech": ["SHOP.TO", "GOOG.TO", "AI.V", "BB.TO"],
        "Canadian Banks": ["RY.TO", "TD.TO", "BNS.TO", "BMO.TO"],
    }
    for sector, tickers in sectors.items():
        st.subheader(f"🔷 {sector} Sector")
        for t in tickers:
            name = get_company_name(t)
            st.markdown(f"**{name} ({t})**")
            news_items = get_news_sources(t)
            for h, l, src in news_items[:2]:
                st.markdown(f"- [{h}]({l}) <sub>({src})</sub>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📰 Sector/Market Breaking News")
    for sector in sectors:
        st.subheader(f"{sector} News")
        items = []
        for t in sectors[sector]:
            items.extend(get_news_sources(t))
        seen = set()
        filtered = []
        for h, l, src in items:
            if h not in seen:
                seen.add(h)
                filtered.append((h, l, src))
        if filtered:
            for h, l, src in filtered[:5]:
                st.markdown(f"- [{h}]({l}) <sub>({src})</sub>", unsafe_allow_html=True)
        else:
            st.write("No news found for this sector.")
    st.divider()
    st.subheader("🧪 FDA Drug Approval News")
    for h, l, src in get_fda_approvals():
        st.markdown(f"- [{h}]({l}) <sub>({src})</sub>", unsafe_allow_html=True)

# ------ Peer Comparison ------
with tabs[2]:
    st.header("Peer Comparison")
    ticker = st.text_input("Enter Ticker for Peer Comparison:", "AAPL")
    if ticker:
        try:
            peers = get_peers(ticker)
            tickers = [ticker] + peers if peers else [ticker]
            data = {}
            for t in tickers:
                df = get_data(t, period='1mo')
                data[t] = df['Close']
            df_comp = pd.DataFrame(data)
            st.line_chart(df_comp)
            st.write("Peers:", ", ".join(peers) if peers else "No peers found.")
        except Exception as e:
            st.error(f"Error: {e}")

# ------ Watchlist ------
with tabs[3]:
    st.header("Watchlist")
    watchlist = st.session_state.setdefault('watchlist', [])
    add = st.text_input("Add ticker to watchlist:")
    if add and add not in watchlist:
        watchlist.append(add)
    remove = st.selectbox("Remove ticker:", options=[""]+watchlist)
    if st.button("Remove") and remove and remove in watchlist:
        watchlist.remove(remove)
    st.write("Current Watchlist:", ", ".join(watchlist))
    for t in watchlist:
        try:
            df = get_data(t, period="1mo")
            st.write(f"**{t}** last close: {df['Close'].iloc[-1]:.2f}")
        except:
            st.write(f"Could not load {t}")

# ------ Financial Health ------
with tabs[4]:
    st.header("Financial Health")
    ticker = st.text_input("Enter Ticker for Financials:", "AAPL")
    if ticker:
        info = get_basic_financials(ticker)
        st.write("Key Financials:")
        st.json(info)
        bs = get_balance_sheet(ticker)
        if not bs.empty:
            st.write("Balance Sheet:")
            st.dataframe(bs)
        else:
            st.write("No balance sheet data found.")

# ------ Options Analytics ------
with tabs[5]:
    st.header("Options Analytics")
    ticker = st.text_input("Enter Ticker for Options Analytics:", "AAPL")
    if ticker:
        calls, puts = get_options(ticker)
        if not calls.empty:
            st.write("Calls:")
            st.dataframe(calls)
        else:
            st.write("No call options found.")
        if not puts.empty:
            st.write("Puts:")
            st.dataframe(puts)
        else:
            st.write("No put options found.")

# ------ Macro Insights ------
with tabs[6]:
    st.header("Macro Insights")
    st.write("Major Global Indices & Commodities")
    data = get_macro_data()
    st.table(pd.DataFrame.from_dict(data, orient='index', columns=["Last Price"]))

# ------ News Explorer ------
with tabs[7]:
    st.header("News Explorer")
    search = st.text_input("Sector/keyword or company/ticker:", value="Nuclear")
    if search:
        news_items = get_news_sources(search)
        if news_items:
            for h, l, src in news_items:
                st.markdown(f"- [{h}]({l}) <sub>({src})</sub>", unsafe_allow_html=True)
        else:
            st.write("No news found.")
