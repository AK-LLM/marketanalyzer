import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
import datetime
from prophet import Prophet
import feedparser
from bs4 import BeautifulSoup
import time

# ------------- WIDEST POSSIBLE GLOBAL TICKER + NAME RESOLUTION -------------
@st.cache_data(show_spinner=False)
def get_ticker_list():
    url = "https://raw.githubusercontent.com/ranaroussi/yfinance-extra/master/data/all_tickers.json"
    try:
        tickers_df = pd.read_json(url)
        tickers_df = tickers_df.dropna(subset=['symbol', 'name'])
        tickers_df.columns = [c.capitalize() for c in tickers_df.columns]
        return tickers_df
    except Exception:
        # Fallback to most common
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "BABA", "BRK-A", "V", "JPM"]
        names = ["Apple", "Microsoft", "Alphabet", "Tesla", "NVIDIA", "Amazon", "Meta Platforms", "Alibaba", "Berkshire Hathaway", "Visa", "JP Morgan Chase"]
        return pd.DataFrame({"Symbol": symbols, "Name": names})

# ------------- TICKER RESOLVE FUNCTION -------------
def search_tickers(name_or_ticker, tickers_df):
    name_or_ticker = name_or_ticker.strip().lower()
    mask = (tickers_df["Name"].str.lower().str.contains(name_or_ticker)) | (tickers_df["Symbol"].str.lower() == name_or_ticker)
    results = tickers_df[mask].reset_index(drop=True)
    return results

def get_company_name(symbol, tickers_df):
    try:
        return tickers_df[tickers_df["Symbol"].str.upper() == symbol.upper()]["Name"].values[0]
    except:
        try:
            return yf.Ticker(symbol).info.get('longName') or yf.Ticker(symbol).info.get('shortName') or symbol
        except:
            return symbol

# ------------- YFINANCE DATA FETCHING -------------
def get_data(ticker, period='1y'):
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data.empty or 'Close' not in data.columns:
            raise ValueError("No valid data found for ticker.")
        return data
    except Exception as e:
        raise ValueError(f"Failed to fetch data for ticker: {str(e)}")

# ------------- SIGNALS + INDICATORS -------------
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

def get_signal_explanation(df):
    if df.empty:
        return "No sufficient data for signals."
    reasons = []
    if df['Buy'].iloc[-1]:
        reasons.append("Oversold (RSI < 30), bullish crossover (MACD > Signal), and positive trend (SMA50 > SMA200).")
    elif df['Sell'].iloc[-1]:
        reasons.append("Overbought (RSI > 70), bearish crossover (MACD < Signal), and negative trend (SMA50 < SMA200).")
    else:
        reasons.append("Neutral: No strong buy/sell indicators. Market may be consolidating.")
    return " ".join(reasons)

# ------------- 7 DAY FORECASTING -------------
def forecast_prices(df, ticker):
    df = df[['Close']].copy().reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    model = Prophet(daily_seasonality=True)
    try:
        model.fit(df)
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat']].tail(7)
    except Exception:
        # fallback
        return pd.DataFrame({'ds': pd.date_range(df['ds'].max(), periods=7, freq="D"), 'yhat': np.nan})

# ------------- FINANCIALS -------------
def get_financials(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        fin = yf_ticker.financials
        bal = yf_ticker.balance_sheet
        cash = yf_ticker.cashflow
        return fin, bal, cash
    except Exception:
        return None, None, None

# ------------- PEERS (BEST EFFORT: INDUSTRY) -------------
def get_peers(ticker):
    try:
        info = yf.Ticker(ticker).info
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        tickers_df = get_ticker_list()
        # Try to match companies with same sector or industry
        mask = (tickers_df['Name'].str.lower().str.contains(industry.lower())) | \
               (tickers_df['Name'].str.lower().str.contains(sector.lower()))
        peers = tickers_df[mask]
        return peers.head(10)
    except Exception:
        return pd.DataFrame()

# ------------- OPTIONS ANALYTICS (BASIC CHAIN) -------------
def get_options_chain(ticker):
    try:
        t = yf.Ticker(ticker)
        dates = t.options
        if not dates:
            return None
        opt = t.option_chain(dates[0])
        return opt.calls, opt.puts
    except Exception:
        return None, None

def suggest_trading_strategy(df):
    if df.empty:
        return "No data for strategy suggestion."
    signal = df['Signal'].iloc[-1]
    if signal == "Buy":
        return "Consider: Buy stock, Bull Call Spread, or Long Call Option (if available)."
    elif signal == "Sell":
        return "Consider: Sell/Short, Bear Put Spread, or Long Put Option."
    else:
        return "Hold: No strong strategy, consider protective stop-loss orders."

# ------------- MACRO INSIGHTS (BASIC LIVE INDICES) -------------
def get_macro_indices():
    indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DAX": "^GDAXI", "Nikkei": "^N225", "FTSE": "^FTSE", "Hang Seng": "^HSI"}
    out = []
    for name, symbol in indices.items():
        try:
            price = yf.Ticker(symbol).history(period="1d")["Close"].iloc[-1]
        except Exception:
            price = np.nan
        out.append((name, symbol, price))
    return out

# ------------- NEWS SOURCING (MULTI-FEED) -------------
def get_news(ticker, company_name=None):
    news_items = []
    try:
        # Google News
        url = f"https://news.google.com/rss/search?q={ticker}+stock"
        feed = feedparser.parse(url)
        news_items += [(entry.title, entry.link) for entry in feed.entries[:5]]
    except:
        pass
    try:
        # Yahoo Finance
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        news_items += [(entry.title, entry.link) for entry in feed.entries[:5]]
    except:
        pass
    # Try company name search
    if company_name:
        try:
            url = f"https://news.google.com/rss/search?q={company_name}"
            feed = feedparser.parse(url)
            news_items += [(entry.title, entry.link) for entry in feed.entries[:3]]
        except:
            pass
    # Deduplicate
    seen = set()
    filtered = []
    for t, l in news_items:
        if l not in seen:
            filtered.append((t, l))
            seen.add(l)
    return filtered

def get_sector_news(sector_keywords):
    # Scrapes Google News and Yahoo Finance RSS for sector-wide news
    sector_news = []
    for kw in sector_keywords:
        try:
            url = f"https://news.google.com/rss/search?q={kw}"
            feed = feedparser.parse(url)
            sector_news += [(entry.title, entry.link) for entry in feed.entries[:3]]
        except:
            pass
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={kw}&region=US&lang=en-US"
            feed = feedparser.parse(url)
            sector_news += [(entry.title, entry.link) for entry in feed.entries[:2]]
        except:
            pass
    # Deduplicate
    seen = set()
    filtered = []
    for t, l in sector_news:
        if l not in seen:
            filtered.append((t, l))
            seen.add(l)
    return filtered[:12]

def get_fda_approvals():
    try:
        rss_url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
        feed = feedparser.parse(rss_url)
        return [(entry.title, entry.link) for entry in feed.entries[:5]]
    except:
        return []

# ----------------------------- STREAMLIT UI START --------------------------
st.set_page_config(page_title="Global Market Analyzer", layout="wide")
st.markdown("<style>footer {visibility: hidden;} .reportview-container .main .block-container{padding-top:2rem;}</style>", unsafe_allow_html=True)

TABS = [
    "Market Information",
    "Suggestions",
    "Peer Comparison",
    "Watchlist",
    "Financial Health",
    "Options Analytics",
    "Macro Insights",
    "News Explorer"
]

tab_idx = st.sidebar.radio("Navigation", TABS, horizontal=True)
tickers_df = get_ticker_list()

# --- Shared State for Ticker (for cross-tab use) ---
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = "AAPL"

# ----------- MARKET INFORMATION TAB -----------
if tab_idx == 0:
    st.title("Market Information")
    company_query = st.text_input("Type company name or ticker:")
    matches = search_tickers(company_query, tickers_df) if company_query else pd.DataFrame()
    selected_symbol = st.session_state.selected_symbol

    if not matches.empty:
        symbol = st.selectbox("Choose Ticker:", matches["Symbol"] + " - " + matches["Name"], index=0)
        if symbol:
            symbol = symbol.split(" - ")[0]
            st.session_state.selected_symbol = symbol
    else:
        symbol = st.session_state.selected_symbol

    if symbol:
        try:
            name = get_company_name(symbol, tickers_df)
            st.header(f"{name} ({symbol})")
            df = get_data(symbol)
            df = calculate_indicators(df)
            df = generate_signals(df)
            st.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
            st.metric("Signal", df['Signal'].iloc[-1])
            st.metric("Trend", "📈 Bullish" if df['Trend'].iloc[-1]==1 else "📉 Bearish")
            st.caption("Rationale/Explanation: " + get_signal_explanation(df))
            st.subheader("7-Day Price Forecast ↪")
            forecast = forecast_prices(df, symbol)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='7D Forecast', line_shape='linear'))
            st.plotly_chart(fig, use_container_width=True)
            # Financials Chart
            fin, bal, cash = get_financials(symbol)
            if fin is not None and not fin.empty:
                st.subheader("Financial Performance (Annual)")
                metrics = ["Total Revenue", "Gross Profit", "Ebitda", "Net Income"]
                fin = fin.T
                fin.index = fin.index.strftime('%Y')
                fig2 = go.Figure()
                for m in metrics:
                    if m in fin.columns:
                        fig2.add_trace(go.Scatter(x=fin.index, y=fin[m], name=m, mode='lines+markers'))
                fig2.update_layout(yaxis_title="USD", xaxis_title="Year")
                st.plotly_chart(fig2, use_container_width=True)
            # News Sentiment
            st.subheader("News Sentiment")
            news = get_news(symbol, name)
            for title, link in news:
                st.markdown(f"- [{title}]({link})")
        except Exception as e:
            st.warning(f"No price data available for this ticker. Please check the ticker symbol or try another. ({e})")
    else:
        st.info("Type a company name and select a valid ticker from the dropdown.")

# ----------- SUGGESTIONS TAB -----------
elif tab_idx == 1:
    st.title("AI Market Suggestions")
    ai_sectors = {
        "AI/Tech": ["NVDA", "SMCI", "AMD", "MSFT", "GOOGL", "AVGO", "ASML", "TSM", "INTC"],
        "Nuclear": ["BWXT", "SMR", "URA"],
        "Defense & Aerospace": ["NOC", "LMT", "RTX", "BA", "PLTR", "GD"],
        "Healthcare": ["UNH", "JNJ", "PFE", "CVS", "MRK"],
        "Healthcare Tech": ["TDOC", "ISRG", "DOCS", "HCA"],
        "Pharma & Biotech": ["MRNA", "VRTX", "REGN", "BNTX", "XBI"],
        "Canadian Market": ["SHOP.TO", "TD.TO", "BNS.TO", "ENB.TO", "CNQ.TO"],
        "ETFs/Global": ["QQQ", "SPY", "ICLN", "ARKK", "ARKG", "VTI", "EFA", "EWJ"]
    }
    st.markdown("#### Companies to Watch")
    for sector, tickers in ai_sectors.items():
        st.markdown(f"**{sector}**")
        for t in tickers:
            name = get_company_name(t, tickers_df)
            st.markdown(f"- **{name} ({t})**")
    st.divider()
    st.markdown("#### Sector Breaking News / Trend News")
    sector_news_keywords = ["AI", "Nuclear", "Defense", "Healthcare", "Pharma", "Aerospace", "ETF", "Technology", "Energy", "Canada"]
    sector_news = get_sector_news(sector_news_keywords)
    for title, link in sector_news:
        st.markdown(f"- [{title}]({link})")
    st.divider()
    st.markdown("#### Regulatory News (FDA Drug Approvals, etc.)")
    for title, link in get_fda_approvals():
        st.markdown(f"- [{title}]({link})")

# ----------- PEER COMPARISON TAB -----------
elif tab_idx == 2:
    st.title("Peer Comparison")
    symbol = st.text_input("Enter Ticker for Peer Comparison:", st.session_state.selected_symbol)
    if symbol:
        try:
            name = get_company_name(symbol, tickers_df)
            peers = get_peers(symbol)
            st.markdown(f"##### Peer Companies for {name} ({symbol}):")
            st.dataframe(peers[["Symbol", "Name"]].reset_index(drop=True))
            # Show metrics
            data = []
            for idx, row in peers.iterrows():
                try:
                    ticker = row["Symbol"]
                    price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
                    data.append({"Symbol": ticker, "Name": row["Name"], "Price": price})
                except:
                    continue
            if data:
                st.table(pd.DataFrame(data))
        except Exception as e:
            st.error("Unable to fetch peer data.")

# ----------- WATCHLIST TAB -----------
elif tab_idx == 3:
    st.title("Watchlist")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["AAPL", "MSFT", "NVDA"]
    action = st.radio("Action:", ["View", "Add", "Delete"], horizontal=True)
    if action == "View":
        if st.session_state.watchlist:
            for t in st.session_state.watchlist:
                try:
                    name = get_company_name(t, tickers_df)
                    price = yf.Ticker(t).history(period="1d")["Close"].iloc[-1]
                    st.write(f"**{name} ({t})**: ${price:,.2f}")
                except:
                    st.write(f"**{t}**: Data error.")
        else:
            st.info("Your watchlist is empty.")
    elif action == "Add":
        add_symbol = st.text_input("Add Ticker:")
        if st.button("Add"):
            if add_symbol and add_symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(add_symbol.upper())
                st.success(f"Added {add_symbol.upper()} to watchlist.")
    elif action == "Delete":
        del_symbol = st.selectbox("Select to Remove:", st.session_state.watchlist)
        if st.button("Delete"):
            st.session_state.watchlist.remove(del_symbol)
            st.success(f"Removed {del_symbol} from watchlist.")

# ----------- FINANCIAL HEALTH TAB -----------
elif tab_idx == 4:
    st.title("Financial Health")
    symbol = st.text_input("Enter Ticker for Financial Health:", st.session_state.selected_symbol)
    if symbol:
        try:
            name = get_company_name(symbol, tickers_df)
            st.header(f"{name} ({symbol})")
            fin, bal, cash = get_financials(symbol)
            if fin is not None and not fin.empty:
                st.write("**Income Statement**")
                st.dataframe(fin)
            if bal is not None and not bal.empty:
                st.write("**Balance Sheet**")
                st.dataframe(bal)
            if cash is not None and not cash.empty:
                st.write("**Cash Flow**")
                st.dataframe(cash)
            # Key Ratios
            try:
                info = yf.Ticker(symbol).info
                ratios = {
                    "Market Cap": info.get("marketCap", None),
                    "Trailing P/E": info.get("trailingPE", None),
                    "Forward P/E": info.get("forwardPE", None),
                    "Debt/Equity": info.get("debtToEquity", None),
                    "ROE": info.get("returnOnEquity", None)
                }
                st.write("**Key Ratios**")
                st.json(ratios)
            except:
                pass
        except Exception as e:
            st.warning("No financial data found.")

# ----------- OPTIONS ANALYTICS TAB -----------
elif tab_idx == 5:
    st.title("Options Analytics")
    symbol = st.text_input("Enter Ticker for Options Analysis:", st.session_state.selected_symbol)
    if symbol:
        calls, puts = get_options_chain(symbol)
        if calls is not None:
            st.write("**Calls**")
            st.dataframe(calls)
            st.write("**Puts**")
            st.dataframe(puts)
            st.write("**AI Trading Strategy Suggestion:**")
            df = get_data(symbol)
            df = calculate_indicators(df)
            df = generate_signals(df)
            st.info(suggest_trading_strategy(df))
        else:
            st.warning("No options data found.")

# ----------- MACRO INSIGHTS TAB -----------
elif tab_idx == 6:
    st.title("Macro Insights")
    st.write("**Major Global Indices:**")
    indices = get_macro_indices()
    st.table(pd.DataFrame(indices, columns=["Index", "Symbol", "Latest Price"]))
    st.write("**Macro News:**")
    news = get_sector_news(["macro", "economy", "interest rates", "GDP", "inflation"])
    for title, link in news:
        st.markdown(f"- [{title}]({link})")

# ----------- NEWS EXPLORER TAB -----------
elif tab_idx == 7:
    st.title("News Explorer")
    st.write("#### Browse Market-Moving News by Keyword, Ticker, or Sector")
    query = st.text_input("Enter keyword, ticker, or sector:")
    if query:
        news = get_sector_news([query])
        if not news:
            st.warning("No news found.")
        for title, link in news:
            st.markdown(f"- [{title}]({link})")
    else:
        st.info("Type a keyword, ticker, or sector above to explore the news.")

# ------ END OF FILE ------
