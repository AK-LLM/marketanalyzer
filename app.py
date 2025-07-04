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

# ===== Helper Functions =====
def human_readable(val):
    try:
        val = float(val)
        if abs(val) >= 1e12:
            return f"${val/1e12:.2f}T"
        elif abs(val) >= 1e9:
            return f"${val/1e9:.2f}B"
        elif abs(val) >= 1e6:
            return f"${val/1e6:.2f}M"
        elif abs(val) >= 1e3:
            return f"${val/1e3:.2f}K"
        else:
            return f"${val:.2f}"
    except:
        return str(val)

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
    df['BB_Upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
    df['BB_Lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
    df['ATR'] = df['High'].combine(df['Low'], max) - df['Low'].combine(df['High'], min)
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

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        keys = ["sector", "industry", "marketCap", "trailingPE", "forwardPE", "dividendYield", "beta", "profitMargins", "pegRatio"]
        fundamentals = {k: info.get(k, 'N/A') for k in keys}
        return fundamentals
    except:
        return {}

# ---- Session State Setup ----
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 0
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = "AAPL"

tab_names = [
    "Market Information", "Suggestions", "Peer Comparison", "Watchlist", 
    "Financial Health", "Options Chain", "Macro Trends", "Portfolio Optimizer", "Event Timeline", "Experimental"
]
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
                st.metric("Current Price", f"${float(df['Close'].iloc[-1]):.2f}")
                signal = str(df['Signal'].iloc[-1]) if not pd.isnull(df['Signal'].iloc[-1]) else "N/A"
                trend_val = df['Trend'].iloc[-1] if 'Trend' in df.columns else 0
                trend = "📈 Bullish" if trend_val == 1 else "📉 Bearish"
                st.metric("Signal", signal)
                st.metric("Trend", trend)

                # Company Fundamentals
                fundamentals = get_fundamentals(ticker)
                if fundamentals:
                    st.subheader("Company Fundamentals")
                    st.write(f"**Sector:** {fundamentals.get('sector')}")
                    st.write(f"**Industry:** {fundamentals.get('industry')}")
                    st.write(f"**Market Cap:** {human_readable(fundamentals.get('marketCap'))}")
                    st.write(f"**P/E Ratio:** {f'{float(fundamentals.get('trailingPE')):.3f}' if fundamentals.get('trailingPE') not in ['N/A', None] else 'N/A'}")
                    st.write(f"**Profit Margin:** {f'{float(fundamentals.get('profitMargins'))*100:.2f}%' if fundamentals.get('profitMargins') not in ['N/A', None] else 'N/A'}")

                # Financial Performance (Annual)
                st.subheader("Financial Performance (Annual)")
                fin_df = get_financials_timeseries(ticker)
                if fin_df is not None and not fin_df.empty:
                    desired_cols = ["Revenue", "Gross Profit", "EBITDA", "Net Income"]
                    available_cols = [c for c in desired_cols if c in fin_df.columns]
                    fin_df = fin_df[available_cols]
                    metric = st.selectbox("Metric", ["All"] + available_cols, key="metric_select_patch")
                    year_range = fin_df.index.year.tolist()
                    min_year, max_year = min(year_range), max(year_range)
                    year_selected = st.slider("Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year), key="fin_year_slider_patch")
                    plot_df = fin_df.loc[(fin_df.index.year >= year_selected[0]) & (fin_df.index.year <= year_selected[1])]
                    fig = go.Figure()
                    metrics = available_cols if metric == "All" else [metric]
                    color_map = {
                        "Revenue": "firebrick",
                        "Gross Profit": "royalblue",
                        "EBITDA": "seagreen",
                        "Net Income": "darkorange"
                    }
                    for m in metrics:
                        fig.add_trace(go.Scatter(
                            x=plot_df.index.strftime("%Y"),
                            y=plot_df[m],
                            mode='lines+markers',
                            name=m,
                            line=dict(color=color_map.get(m, "gray"), width=3),
                            text=[human_readable(v) for v in plot_df[m]],
                            hovertemplate=f"<b>%{{x}}</b><br>{m}: %{{text}}"
                        ))
                    fig.update_layout(
                        xaxis_title="Year",
                        yaxis_title="Value (USD)",
                        legend_title="Metric",
                        hovermode="x unified",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No financial data available.")

                # 7-Day Price Forecast (Line)
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

                # Buy/Hold/Sell suggestion
                st.subheader("AI Suggestion")
                if signal in ["Buy", "Sell"]:
                    st.success(f"**{signal}** recommendation detected.")
                else:
                    st.info("No strong Buy or Sell signal.")

                st.markdown("**Trading Strategies:**")
                for strat in suggest_trading_strategy(signal):
                    st.write("-", strat)

                # Explain signal/forecast if in conflict
                trend_numeric = df['Trend'].iloc[-1] if 'Trend' in df.columns else 0
                explanation = explain_signal_vs_forecast(signal, trend_numeric, forecast)
                if explanation:
                    st.info(explanation)

                # News Sentiment
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
    st.header("💡 AI Suggested Companies to Watch")
    ai_suggestions = {
        "AI": ["NVDA", "SMCI", "AMD", "GOOGL", "MSFT", "PLTR", "META", "TSLA", "BIDU", "ADBE", "CRM", "AI", "SNOW"],
        "Semiconductors": ["AVGO", "QCOM", "TXN", "TSM", "ASML", "INTC", "STM", "AMAT", "MRVL", "ON", "MCHP", "NXPI", "ADI", "SWKS"],
        "Tech": ["AAPL", "AMZN", "ORCL", "SAP", "IBM", "CSCO", "GOOG", "CRM", "ADBE", "SHOP", "UBER", "SQ", "NOW", "ZM"],
        "Defense & Aerospace": ["LMT", "NOC", "RTX", "GD", "PLTR", "BA", "HII", "TXT", "BWXT", "TDG"],
        "Nuclear & Clean Energy": ["BWXT", "CCJ", "SMR", "U", "CAMECO", "ICLN", "NEE", "DNN", "XOM", "SHEL"],
        "Healthcare": ["JNJ", "PFE", "LLY", "UNH", "CVS", "MRK", "ABT", "MDT", "TMO", "DHR", "BMY", "ZBH", "SNY"],
        "Healthcare Tech": ["ISRG", "TDOC", "MDT", "DXCM", "CNC", "HCA", "VEEV", "VRTX", "IDXX", "ALGN"],
        "Pharmaceuticals & Biotech": ["PFE", "MRNA", "BNTX", "SNY", "AMGN", "REGN", "GILD", "VRTX", "NVO", "AZN", "BIIB", "RHHBY", "GSK"],
        "Clean Energy & Renewables": ["ICLN", "NEE", "ENPH", "SEDG", "PLUG", "FSLR", "BE", "RUN", "TSLA", "BLDP"],
        "Cloud/Data/Software": ["MSFT", "GOOGL", "AMZN", "ORCL", "SNOW", "DDOG", "ZS", "MDB", "NET", "PANW", "CRWD", "OKTA"],
        "Fintech": ["V", "MA", "PYPL", "SQ", "AXP", "COIN", "SOFI", "INTU
