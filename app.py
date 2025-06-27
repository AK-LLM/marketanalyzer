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

time.sleep(0.1)
st.set_page_config(page_title="AI-Powered Market Analyzer", layout="wide")

@st.cache_data
def get_data(ticker, period='1y'):
    try:
        ticker = ticker.upper().strip()
        if '-' in ticker and not ticker.endswith('-USD'):
            ticker += '-USD'
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data.empty or 'Close' not in data.columns:
            raise ValueError("No valid data found for ticker.")
        return data
    except Exception as e:
        raise ValueError(f"Failed to fetch data for ticker: {str(e)}")

def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('longName') or info.get('shortName') or ticker
    except:
        return ticker

def calculate_indicators(df):
    if len(df) < 50:
        return df
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
    if len(df) == 0 or 'SMA_50' not in df or 'SMA_200' not in df:
        df['Signal'] = "Hold"
        df['Trend'] = 0
        df['Buy'] = False
        df['Sell'] = False
        return df
    df['Trend'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    df['Buy'] = (df['RSI'] < 30) & (df['MACD'] > df['Signal_Line']) & (df['Trend'] == 1)
    df['Sell'] = (df['RSI'] > 70) & (df['MACD'] < df['Signal_Line']) & (df['Trend'] == -1)
    df['Signal'] = np.where(df['Buy'], 'Buy', np.where(df['Sell'], 'Sell', 'Hold'))
    return df

def get_news(ticker):
    try:
        url = f"https://news.google.com/search?q={ticker}+stock"
        html = requests.get(url, timeout=6).text
        soup = BeautifulSoup(html, 'html.parser')
        return [item.text for item in soup.select('article h3 a')][:5]
    except:
        return ["No news found."]

def get_sector_news(sector_keywords):
    try:
        query = "+".join(sector_keywords) + "+sector+news"
        url = f"https://news.google.com/search?q={query}"
        html = requests.get(url, timeout=6).text
        soup = BeautifulSoup(html, 'html.parser')
        return [item.text for item in soup.select('article h3 a')][:5]
    except:
        return ["No sector news found."]

def get_fda_approvals():
    try:
        rss_url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:5]]
    except:
        return ["No FDA data available"]

def get_yahoo_rss_headlines(limit=5):
    try:
        rss_url = "https://finance.yahoo.com/rss/topstories"
        feed = feedparser.parse(rss_url)
        return [entry.title for entry in feed.entries[:limit]]
    except:
        return ["No Yahoo Finance RSS news available."]

def forecast_prices(df, ticker):
    if len(df) < 30:
        return None
    prophet_df = df[['Close']].copy().reset_index()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df['news_sentiment'] = 0

    model = Prophet(daily_seasonality=True)
    model.add_regressor('news_sentiment')
    try:
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=7)
        future['news_sentiment'] = 0
        forecast = model.predict(future)
        forecast_range = forecast.tail(7)
        x = np.arange(len(forecast_range))
        linear_coeffs = np.polyfit(x, forecast_range['yhat'], 1)
        linear_trend = np.polyval(linear_coeffs, x)
        forecast.loc[forecast.index[-7:], 'trendline'] = linear_trend
        return forecast[['ds', 'yhat', 'trendline']].tail(7)
    except:
        return None

def suggest_trading_strategy(signal):
    if signal == "Buy":
        return [
            "Market Order: Buy at current price.",
            "Limit Order: Set a buy limit slightly below current price.",
            "Protective Stop Loss: Place stop ~3-5% below entry.",
            "Options: Consider buying a CALL or selling a PUT."
        ]
    elif signal == "Sell":
        return [
            "Market Order: Sell at current price.",
            "Limit Order: Set a sell limit slightly above current price.",
            "Protective Stop Loss: Place stop ~3-5% above entry.",
            "Options: Consider buying a PUT or selling a CALL."
        ]
    else:
        return [
            "Monitor: No strong action recommended.",
            "Consider trailing stop or covered call for yield.",
            "Re-assess after next earnings or major news."
        ]

def explain_signal_vs_forecast(signal, trend, forecast):
    if forecast is None or forecast.empty:
        return ""
    forecast_trajectory = forecast['trendline'].values
    rising = forecast_trajectory[-1] > forecast_trajectory[0]
    explanation = ""
    if trend == -1 and rising:
        explanation = (
            "ℹ️ **Notice:** The long-term technical trend is still bearish, but the model predicts a possible short-term price uptick. "
            "This could be a 'bear market rally' or mean reversion. Exercise caution if trading against the broader trend."
        )
    elif trend == 1 and not rising:
        explanation = (
            "ℹ️ **Notice:** The technical trend is bullish, but the model forecasts a possible short-term pullback. "
            "Markets can have short corrections during uptrends."
        )
    elif signal == "Hold" and (rising or not rising):
        explanation = (
            "ℹ️ **Notice:** The system sees no strong buy or sell technical signal, but the forecast indicates possible movement. "
            "This may reflect recent news, volatility, or a lack of clear trend."
        )
    return explanation

# ----- New Feature Functions -----
def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        keys = ["sector", "industry", "marketCap", "trailingPE", "forwardPE", "dividendYield", "beta", "profitMargins", "pegRatio"]
        fundamentals = {k: info.get(k, 'N/A') for k in keys}
        return fundamentals
    except:
        return {}

def get_earnings_events(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        cal = ticker_obj.calendar
        splits = ticker_obj.splits
        dividends = ticker_obj.dividends
        events = {}
        if not cal.empty:
            for col in cal.columns:
                events[col] = cal[col][0]
        if not splits.empty:
            events["Last Split"] = splits.index[-1].strftime("%Y-%m-%d") + f" ({splits[-1]}:1)"
        if not dividends.empty:
            events["Last Dividend"] = dividends.index[-1].strftime("%Y-%m-%d") + f" (${dividends[-1]:.2f})"
        return events
    except:
        return {}

def get_volatility(df):
    try:
        df = df.copy()
        df['Return'] = df['Close'].pct_change()
        std_30d = df['Return'].rolling(window=30).std().iloc[-1]
        df['H-L'] = df['High'] - df['Low']
        df['ATR'] = df['H-L'].rolling(window=14).mean()
        atr_14d = df['ATR'].iloc[-1]
        return std_30d, atr_14d
    except:
        return None, None

def get_signal_history(df):
    try:
        recent = df.tail(100).copy()
        buys = recent[recent['Buy']]
        sells = recent[recent['Sell']]
        return buys, sells
    except:
        return pd.DataFrame(), pd.DataFrame()

def get_peers(ticker):
    try:
        info = yf.Ticker(ticker).info
        sector = info.get('sector')
        industry = info.get('industry')
        candidates = {
            'Tech': ['AAPL', 'MSFT', 'GOOGL'],
            'Semiconductors': ['NVDA', 'AMD', 'AVGO'],
            'Healthcare': ['JNJ', 'PFE', 'MRK'],
            'Finance': ['JPM', 'BAC', 'GS'],
            'Consumer': ['PG', 'KO', 'PEP']
        }
        if sector:
            for k in candidates:
                if k in sector:
                    return [t for t in candidates[k] if t != ticker][:2]
        return ['AAPL', 'MSFT'] if ticker != 'AAPL' else ['MSFT', 'GOOGL']
    except:
        return ['AAPL', 'MSFT']

def get_peer_comparison(ticker, peers, period='1y'):
    data = {}
    tickers = [ticker] + peers
    for t in tickers:
        try:
            d = yf.download(t, period=period, auto_adjust=True, progress=False)
            data[t] = d['Close']
        except:
            continue
    return pd.DataFrame(data)

def get_analyst_rating(ticker):
    try:
        info = yf.Ticker(ticker).info
        rec = info.get('recommendationKey', 'N/A').capitalize()
        target = info.get('targetMeanPrice', 'N/A')
        return rec, target
    except:
        return "N/A", "N/A"

def get_news_sentiment(ticker):
    try:
        from textblob import TextBlob
        headlines = get_news(ticker)
        polarity = [TextBlob(h).sentiment.polarity for h in headlines]
        if polarity:
            pos = sum(1 for p in polarity if p > 0.1)
            neg = sum(1 for p in polarity if p < -0.1)
            neu = len(polarity) - pos - neg
            return {"positive": pos, "negative": neg, "neutral": neu}
        else:
            return {"positive": 0, "negative": 0, "neutral": 0}
    except:
        return {}

# -- Formatters for market cap, P/E, and profit margin
def _format_market_cap(val):
    try:
        val = float(val)
        if val >= 1e12:
            return f"{val/1e12:.2f}T"
        elif val >= 1e9:
            return f"{val/1e9:.2f}B"
        elif val >= 1e6:
            return f"{val/1e6:.2f}M"
        elif val >= 1e3:
            return f"{val/1e3:.2f}K"
        else:
            return str(int(val))
    except:
        return "N/A"

def _format_pe(val):
    try:
        if val == 'N/A' or val is None:
            return "N/A"
        return f"{float(val):.3f}"
    except:
        return "N/A"

def _format_profit_margin(val):
    try:
        if val == 'N/A' or val is None:
            return "N/A"
        return f"{float(val)*100:.2f}%"
    except:
        return "N/A"

def get_financials_timeseries(ticker):
    try:
        fin = yf.Ticker(ticker).financials
        if fin.empty:
            return None
        fin = fin.T
        keys = {
            'Revenue': ['Total Revenue', 'Revenue'],
            'Gross Profit': ['Gross Profit'],
            'Net Income': ['Net Income', 'Net Income Applicable To Common Shares'],
            'EBITDA': ['EBITDA']
        }
        data = {}
        for k, search_keys in keys.items():
            for col in search_keys:
                if col in fin.columns:
                    data[k] = fin[col]
                    break
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except:
        return None

# --- SESSION STATE SETUP ---
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 0
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

tab_names = ["Market Information", "Suggestions"]
selected_tab = st.radio("Navigation", tab_names, index=st.session_state.selected_tab, horizontal=True, key="main_tabs")

# ========== MARKET INFORMATION TAB ==========
if selected_tab == "Market Information":
    st.header("Market Information")
    ticker = st.text_input("Enter any ticker (Stock, ETF, Crypto, FOREX, Commodity):",
                           st.session_state.selected_ticker)
    if ticker:
        try:
            df = get_data(ticker)
            df = calculate_indicators(df)
            df = generate_signals(df)
            if df.empty or len(df) < 10:
                st.error("No data available after processing for this ticker. Try another ticker or check your connection.")
            else:
                name = get_company_name(ticker)
                st.subheader(f"{name} ({ticker.upper()})")
                try:
                    st.metric("Current Price", f"${float(df['Close'].iloc[-1]):.2f}")
                    signal = str(df['Signal'].iloc[-1]) if not pd.isnull(df['Signal'].iloc[-1]) else "N/A"
                    trend_val = df['Trend'].iloc[-1] if 'Trend' in df.columns else 0
                    trend = "📈 Bullish" if trend_val == 1 else "📉 Bearish"
                    st.metric("Signal", signal)
                    st.metric("Trend", trend)
                except Exception as e:
                    st.warning("Could not display key metrics. Data may be missing.")

                # (1) Company Fundamentals
                try:
                    fundamentals = get_fundamentals(ticker)
                    if fundamentals:
                        st.subheader("Company Fundamentals")
                        st.write(f"**Sector:** {fundamentals.get('sector')}")
                        st.write(f"**Industry:** {fundamentals.get('industry')}")
                        st.write(f"**Market Cap:** {_format_market_cap(fundamentals.get('marketCap'))}")
                        st.write(f"**P/E Ratio:** {_format_pe(fundamentals.get('trailingPE'))}")
                        st.write(f"**Dividend Yield:** {fundamentals.get('dividendYield')}")
                        st.write(f"**Profit Margin:** {_format_profit_margin(fundamentals.get('profitMargins'))}")
                except:
                    pass

                # (New) Financial Performance Chart
                try:
                    st.subheader("Financial Performance (Annual)")
                    fin_df = get_financials_timeseries(ticker)
                    if fin_df is not None and not fin_df.empty:
                        col = st.selectbox("Metric", ["All"] + list(fin_df.columns), key="metric_select")
                        year_range = list(fin_df.index.year)
                        year_min, year_max = min(year_range), max(year_range)
                        year_selected = st.slider("Year Range", year_min, year_max, (year_min, year_max), key="year_slider")
                        filtered = fin_df[(fin_df.index.year >= year_selected[0]) & (fin_df.index.year <= year_selected[1])]
                        if col == "All":
                            st.bar_chart(filtered)
                        else:
                            st.bar_chart(filtered[[col]])
                    else:
                        st.info("No financial data available.")
                except Exception as e:
                    st.warning(f"Financial performance chart error: {e}")

                # (2) Earnings/Events
                try:
                    st.subheader("Key Events")
                    events = get_earnings_events(ticker)
                    if events:
                        for k, v in events.items():
                            st.write(f"{k}: {v}")
                except:
                    pass

                # (3) Volatility/ATR
                try:
                    st.subheader("Volatility")
                    std_30d, atr_14d = get_volatility(df)
                    if std_30d and atr_14d:
                        st.write(f"30d Std Dev: {std_30d:.4f}")
                        st.write(f"14d ATR: {atr_14d:.2f}")
                except:
                    pass

                # (4) Backtest Signal History on Chart
                try:
                    st.subheader("Buy/Sell Signal Chart (last 100 days)")
                    buys, sells = get_signal_history(df)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=df.index, y=df.get('SMA_50', pd.Series()), name='50 SMA', line=dict(color='orange', dash='dash')))
                    fig.add_trace(go.Scatter(x=df.index, y=df.get('SMA_200', pd.Series()), name='200 SMA', line=dict(color='green', dash='dot')))
                    fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'))
                    fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal'))
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

                # (5) Peer/Industry Comparison
                try:
                    st.subheader("Peer Comparison (1Y Total Return)")
                    peers = get_peers(ticker)
                    df_peers = get_peer_comparison(ticker, peers)
                    if not df_peers.empty:
                        returns = (df_peers / df_peers.iloc[0] - 1) * 100
                        st.line_chart(returns)
                        st.write("Peers:", ", ".join(peers))
                except:
                    pass

                # (6) Simulated "Add to Watchlist"
                try:
                    st.subheader("Watchlist Export (simulated)")
                    if st.button("Copy ticker to clipboard (manual)"):
                        st.code(ticker)
                except:
                    pass

                # (7) Analyst Rating/Target
                try:
                    st.subheader("Analyst Consensus")
                    rec, target = get_analyst_rating(ticker)
                    st.write(f"Consensus: {rec}")
                    st.write(f"Avg. Target: ${target}")
                except:
                    pass

                # (8) News Sentiment Scoring
                try:
                    st.subheader("News Sentiment Analysis")
                    from textblob import TextBlob
                    sentiment = get_news_sentiment(ticker)
                    st.write(sentiment)
                except:
                    pass

                st.subheader("AI Suggestion")
                if signal in ["Buy", "Sell"]:
                    st.success(f"**{signal}** recommendation detected.")
                else:
                    st.info("No strong Buy or Sell signal.")

                st.markdown("**Trading Strategies:**")
                for strat in suggest_trading_strategy(signal):
                    st.write("-", strat)

                st.subheader("7-Day Price Forecast")
                forecast = forecast_prices(df.copy(), ticker)
                if forecast is not None and not forecast.empty and 'trendline' in forecast:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['trendline'],
                        name='7-Day Forecast', line=dict(shape='linear', color='#FF5E5B')
                    ))
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("Forecast data unavailable.")

                trend_numeric = df['Trend'].iloc[-1] if 'Trend' in df.columns else 0
                explanation = explain_signal_vs_forecast(signal, trend_numeric, forecast)
                if explanation:
                    st.info(explanation)

                st.subheader("News Sentiment")
                news_items = get_news(ticker)
                if news_items:
                    for item in news_items:
                        st.write("-", item)
                else:
                    st.write("- No news found.")
        except Exception as e:
            st.error(f"Error: {e}")

# ========== SUGGESTIONS TAB ==========
elif selected_tab == "Suggestions":
    # ... your previously frozen suggestions tab code goes here, unchanged ...
    st.header("💡 AI Suggested Companies to Watch")

    ai_suggestions = {
        "AI": [
            "NVDA", "SMCI", "AMD", "GOOGL", "MSFT", "PLTR", "META", "TSLA", "BIDU", "ADBE", "CRM", "AI", "SNOW"
        ],
        "Semiconductors": [
            "AVGO", "QCOM", "TXN", "TSM", "ASML", "INTC", "STM", "AMAT", "MRVL", "ON", "MCHP", "NXPI", "ADI", "SWKS"
        ],
        "Tech": [
            "AAPL", "AMZN", "ORCL", "SAP", "IBM", "CSCO", "GOOG", "CRM", "ADBE", "SHOP", "UBER", "SQ", "NOW", "ZM"
        ],
        "Defense & Aerospace": [
            "LMT", "NOC", "RTX", "GD", "PLTR", "BA", "HII", "TXT", "BWXT", "TDG"
        ],
        "Nuclear & Clean Energy": [
            "BWXT", "CCJ", "SMR", "U", "CAMECO", "ICLN", "NEE", "DNN", "XOM", "SHEL"
        ],
        "Healthcare": [
            "JNJ", "PFE", "LLY", "UNH", "CVS", "MRK", "ABT", "MDT", "TMO", "DHR", "BMY", "ZBH", "SNY"
        ],
        "Healthcare Tech": [
            "ISRG", "TDOC", "MDT", "DXCM", "CNC", "HCA", "VEEV", "VRTX", "IDXX", "ALGN"
        ],
        "Pharmaceuticals & Biotech": [
            "PFE", "MRNA", "BNTX", "SNY", "AMGN", "REGN", "GILD", "VRTX", "NVO", "AZN", "BIIB", "RHHBY", "GSK"
        ],
        "Clean Energy & Renewables": [
            "ICLN", "NEE", "ENPH", "SEDG", "PLUG", "FSLR", "BE", "RUN", "TSLA", "BLDP"
        ],
        "Cloud/Data/Software": [
            "MSFT", "GOOGL", "AMZN", "ORCL", "SNOW", "DDOG", "ZS", "MDB", "NET", "PANW", "CRWD", "OKTA"
        ],
        "Fintech": [
            "V", "MA", "PYPL", "SQ", "AXP", "COIN", "SOFI", "INTU", "FIS", "FISV"
        ],
        "MegaCap & Trending": [
            "AAPL", "AMZN", "GOOG", "MSFT", "META", "TSLA", "NVDA", "BRK.B", "UNH", "V", "WMT"
        ]
    }

    sector_keywords_map = {
        "AI": ["artificial", "intelligence", "AI"],
        "Semiconductors": ["semiconductor", "chip", "microchip"],
        "Tech": ["technology", "tech"],
        "Defense & Aerospace": ["defense", "military", "aerospace"],
        "Nuclear & Clean Energy": ["nuclear", "uranium", "energy", "clean energy", "renewable"],
        "Healthcare": ["healthcare", "medical", "health"],
        "Healthcare Tech": ["healthcare", "technology", "healthtech"],
        "Pharmaceuticals & Biotech": ["pharma", "pharmaceutical", "biotech", "drug"],
        "Clean Energy & Renewables": ["clean energy", "renewable", "solar", "wind", "hydrogen"],
        "Cloud/Data/Software": ["cloud", "software", "data", "SaaS"],
        "Fintech": ["fintech", "finance", "payment", "bank", "digital"],
        "MegaCap & Trending": ["stock market", "biggest companies", "top stocks"]
    }

    for sector, tickers in ai_suggestions.items():
        st.subheader(f"🔷 {sector} Sector")
        for ticker in tickers:
            name = get_company_name(ticker)
            st.markdown(f"**{name} ({ticker})**")
            news_items = get_news(ticker)
            for n in news_items:
                st.write("-", n)
        st.markdown("*Sector-wide breaking/trend news:*")
        kw = sector_keywords_map.get(sector, [sector])
        sector_news = get_sector_news(kw)
        for headline in sector_news:
            st.write("-", headline)

    st.divider()
    st.markdown("### 📰 News")
    all_news_sources = list({ticker for sector_list in ai_suggestions.values() for ticker in sector_list})
    for ticker in all_news_sources:
        st.subheader(f"🗞️ {get_company_name(ticker)} ({ticker})")
        for n in get_news(ticker):
            st.write("-", n)
    st.subheader("🌐 Yahoo Finance Top Stories")
    for ynews in get_yahoo_rss_headlines():
        st.write("-", ynews)
    st.subheader("🧪 FDA Drug Approval News")
    for fda in get_fda_approvals():
        st.write("-", fda)
