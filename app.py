import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import feedparser
import plotly.graph_objs as go
import os

# ============ GLOBAL VARS ============= #
st.set_page_config(page_title="Market Terminal", layout="wide")
TICKER_FILE = "global_tickers.csv"
if os.path.exists(TICKER_FILE):
    tickers_df = pd.read_csv(TICKER_FILE)
    tickers_df = tickers_df.dropna(subset=["Symbol", "Name"])
    tickers_df["Symbol"] = tickers_df["Symbol"].astype(str)
    tickers_df["Name"] = tickers_df["Name"].astype(str)
else:
    tickers_df = pd.DataFrame(columns=["Symbol", "Name", "Exchange"])

def resolve_ticker(query):
    if not isinstance(query, str) or not query:
        return None, None
    query = query.strip().upper()
    # Try symbol first
    matches = tickers_df[tickers_df["Symbol"].str.upper() == query]
    if not matches.empty:
        return matches.iloc[0]["Symbol"], matches.iloc[0]["Name"]
    # Try name search
    name_matches = tickers_df[tickers_df["Name"].str.upper().str.contains(query)]
    if not name_matches.empty:
        # If multiple, return the first and let user know to be more specific
        return name_matches.iloc[0]["Symbol"], name_matches.iloc[0]["Name"]
    return None, None

def get_news(company_or_ticker):
    # Yahoo RSS
    yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={company_or_ticker}&region=US&lang=en-US"
    try:
        d = feedparser.parse(yahoo_url)
        news_list = []
        for entry in d.entries[:8]:
            news_list.append({"title": entry.title, "link": entry.link})
        if news_list:
            return news_list
    except:
        pass
    # Google News RSS as fallback
    google_url = f"https://news.google.com/rss/search?q={company_or_ticker}"
    try:
        d = feedparser.parse(google_url)
        news_list = []
        for entry in d.entries[:8]:
            news_list.append({"title": entry.title, "link": entry.link})
        return news_list
    except:
        pass
    return []

def get_macro_news():
    # Macro news: general economics/central banks/FX, best-effort
    macro_url = "https://news.google.com/rss/search?q=economy+OR+central+bank+OR+inflation+OR+macro"
    d = feedparser.parse(macro_url)
    news_list = []
    for entry in d.entries[:8]:
        news_list.append({"title": entry.title, "link": entry.link})
    return news_list

# ========== SIDEBAR NAVIGATION ========== #
tabs = [
    "Market Information",
    "Suggestions",
    "Peer Comparison",
    "Watchlist",
    "Financial Health",
    "Options Analytics",
    "Macro Insights",
    "News Explorer"
]
st.markdown("<h1 style='font-size:2.8rem;'>Market Terminal</h1>", unsafe_allow_html=True)
selected_tab = st.selectbox(
    "Navigation", tabs, key="nav", label_visibility="collapsed", index=0,
    options=tabs, format_func=lambda x: x
)

# ========== TABS IMPLEMENTATION ========== #

if selected_tab == "Market Information":
    st.header("Market Information")
    query = st.text_input("Type company name or ticker (Name or Ticker):", key="query")
    ticker, company_name = None, None
    options = []
    if query:
        # Get all partial matches for drop down
        options = tickers_df[
            (tickers_df["Symbol"].str.upper().str.contains(query.upper())) |
            (tickers_df["Name"].str.upper().str.contains(query.upper()))
        ].head(10)[["Symbol", "Name"]].apply(lambda row: f"{row['Name']} ({row['Symbol']})", axis=1).tolist()
    selected = st.selectbox("Select:", options, key="tickerselect") if options else None
    if selected:
        # Parse ticker from select
        selected_ticker = selected.split("(")[-1].replace(")", "")
        ticker, company_name = resolve_ticker(selected_ticker)
    elif query:
        ticker, company_name = resolve_ticker(query)
    if ticker:
        st.subheader(f"{company_name} ({ticker})")
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)
            info = yf.Ticker(ticker).info
        except Exception as e:
            df = pd.DataFrame()
            info = {}
        if df is not None and not df.empty:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
            # Mini chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'],
                mode='lines', name='Close Price'
            ))
            fig.update_layout(title="6-Month Price Trend", height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No price data available for this ticker. Please check the ticker symbol or try another.")
        # --- Signal and Trend
        signal, trend, rationale = None, None, ""
        if df is not None and not df.empty:
            # Very basic trend/signal
            if df['Close'].iloc[-1] > df['Close'].iloc[0]:
                trend = "Bullish"
            else:
                trend = "Bearish"
            pct_change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
            if pct_change > 5:
                signal = "Buy"
            elif pct_change < -5:
                signal = "Sell"
            else:
                signal = "Hold"
            rationale = (
                f"Signal: {signal} because the price has changed {pct_change:.2f}% over the last 6 months. "
                f"Trend is {trend} based on overall movement. "
                "Signals are based on price momentum, not news or fundamentals. Combine with your own research."
            )
            st.markdown(f"**Signal:** {signal} &nbsp;&nbsp; **Trend:** {trend}")
            st.info(rationale)
        # --- News Section
        st.markdown("#### Latest News")
        news = get_news(ticker or company_name)
        if news:
            for n in news:
                st.markdown(f"- [{n['title']}]({n['link']})")
        else:
            st.info("No news found for this ticker/company.")
        # --- Financials Section
        if info:
            st.markdown("#### Key Financials")
            keys = ["marketCap", "trailingPE", "grossProfits", "ebitda", "totalRevenue", "netIncomeToCommon"]
            renamed = {
                "marketCap": "Market Cap",
                "trailingPE": "P/E Ratio",
                "grossProfits": "Gross Profit",
                "ebitda": "EBITDA",
                "totalRevenue": "Total Revenue",
                "netIncomeToCommon": "Net Income"
            }
            for k in keys:
                v = info.get(k)
                if v is not None:
                    if k == "marketCap":
                        st.write(f"**{renamed[k]}:** ${v:,.0f}")
                    elif k == "trailingPE":
                        st.write(f"**{renamed[k]}:** {v:.2f}")
                    else:
                        st.write(f"**{renamed[k]}:** ${v:,.0f}")
        # --- Chart (Annual)
        try:
            fin = yf.Ticker(ticker).financials
            if not fin.empty:
                st.markdown("#### Annual Financials")
                st.dataframe(fin)
        except Exception as e:
            pass
    else:
        st.info("Type a company name and select a valid ticker from the dropdown.")

# ---- Suggestions ----
elif selected_tab == "Suggestions":
    st.header("AI Suggestions & News")
    # Demo: Show sector news for major global sectors
    sectors = {
        "AI Sector": ["NVDA", "AMD", "GOOGL", "MSFT", "TSLA", "META", "BIDU"],
        "Healthcare": ["JNJ", "PFE", "MRK", "RHHBY", "NVS"],
        "Energy": ["XOM", "CVX", "RDS.A", "BP", "TOT"],
        "Defense & Aerospace": ["LMT", "RTX", "BA", "NOC"],
        "Pharma": ["PFE", "NVS", "SNY", "AZN"],
        "Tech": ["AAPL", "MSFT", "GOOGL", "TSM", "SAMSUNG"],
        "Finance": ["JPM", "BAC", "WFC", "HSBC", "BNS"],
        "Nuclear": ["CCJ", "UEC", "DNN", "OROCF"]
    }
    for sector, tickers in sectors.items():
        st.markdown(f"### {sector}")
        for t in tickers:
            tk, name = resolve_ticker(t)
            nm = name if name else t
            st.markdown(f"**{nm} ({t})**")
            news = get_news(t)
            if news:
                for n in news:
                    st.markdown(f"- [{n['title']}]({n['link']})")
    st.markdown("---")
    st.markdown("##### Macro Trend News")
    macro_news = get_macro_news()
    for n in macro_news:
        st.markdown(f"- [{n['title']}]({n['link']})")

# ---- Peer Comparison ----
elif selected_tab == "Peer Comparison":
    st.header("Peer Comparison")
    st.info("Peer comparison for global tickers is limited by available free data. For S&P 500 and major stocks, peers will be shown if available.")
    peer_query = st.text_input("Type company name or ticker for peer comparison:")
    ticker, company_name = resolve_ticker(peer_query)
    if ticker:
        tkr = yf.Ticker(ticker)
        sector = tkr.info.get("sector")
        if sector:
            peer_df = tickers_df[tickers_df["Symbol"] != ticker]
            sector_peers = peer_df[peer_df["Name"].str.contains(sector, case=False, na=False)]
            if not sector_peers.empty:
                st.markdown(f"#### Peers in sector: {sector}")
                st.dataframe(sector_peers.head(10))
            else:
                st.info("No peers found for this ticker/sector in free dataset.")
        else:
            st.info("Sector info not available for this ticker.")
    else:
        st.info("Enter a company name or ticker above.")

# ---- Watchlist ----
elif selected_tab == "Watchlist":
    st.header("Watchlist")
    # Watchlist is a session_state list
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []
    add_query = st.text_input("Add company (name or ticker):", key="watchadd")
    add_ticker, add_name = resolve_ticker(add_query)
    if st.button("Add to Watchlist"):
        if add_ticker and add_ticker not in st.session_state["watchlist"]:
            st.session_state["watchlist"].append(add_ticker)
    if st.session_state["watchlist"]:
        remove = st.selectbox("Remove from watchlist:", st.session_state["watchlist"])
        if st.button("Remove"):
            st.session_state["watchlist"].remove(remove)
    st.markdown("#### Your Watchlist")
    for t in st.session_state["watchlist"]:
        name = resolve_ticker(t)[1]
        st.markdown(f"- **{name} ({t})**")

# ---- Financial Health ----
elif selected_tab == "Financial Health":
    st.header("Financial Health")
    fh_query = st.text_input("Select company (Name or Ticker):", key="fhquery")
    ticker, company_name = resolve_ticker(fh_query)
    if ticker:
        try:
            tkr = yf.Ticker(ticker)
            st.markdown(f"#### {company_name} ({ticker}) - Financial Ratios")
            st.json(tkr.info)
            st.markdown("#### Annual Financial Statements")
            st.dataframe(tkr.financials)
        except Exception as e:
            st.error("No financial data found for this ticker.")
    else:
        st.info("Enter a company name or ticker above.")

# ---- Options Analytics ----
elif selected_tab == "Options Analytics":
    st.header("Options Analytics")
    st.info("Options data is available for US tickers only due to free data limitations.")
    op_query = st.text_input("Type ticker (US stocks only):", key="optq")
    ticker, company_name = resolve_ticker(op_query)
    if ticker:
        try:
            tkr = yf.Ticker(ticker)
            options = tkr.options
            if options:
                st.markdown(f"#### {company_name} ({ticker}) - Options Expiry Dates")
                st.write(options)
            else:
                st.warning("No options data available for this ticker.")
        except Exception as e:
            st.error("No options data found for this ticker.")
    else:
        st.info("Enter a ticker (US stocks only).")

# ---- Macro Insights ----
elif selected_tab == "Macro Insights":
    st.header("Macro Insights")
    st.markdown("#### Macro Economic News and Indicators")
    macro_news = get_macro_news()
    for n in macro_news:
        st.markdown(f"- [{n['title']}]({n['link']})")
    st.info("For full macro indicators, see FRED, World Bank, or central bank sites.")

# ---- News Explorer ----
elif selected_tab == "News Explorer":
    st.header("News Explorer")
    news_query = st.text_input("Search news for:")
    if news_query:
        news = get_news(news_query)
        if news:
            for n in news:
                st.markdown(f"- [{n['title']}]({n['link']})")
        else:
            st.warning("No news found for your search.")

# ---- END OF APP ----
