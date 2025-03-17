import os
import openai
import yfinance as yf
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Market Symbols
SP500_SYMBOL = "^GSPC"
NASDAQ_SYMBOL = "^IXIC"
ASX200_SYMBOL = "^AXJO"
GOLD_SYMBOL = "GC=F"
MARKETS = {
    "S&P 500": SP500_SYMBOL,
    "NASDAQ": NASDAQ_SYMBOL,
    "Dow Jones": "^DJI",
    "ASX 200": ASX200_SYMBOL,
    "Gold": "GC=F",
    "US 10-Yr Bond Yield": "^TNX"
}

def calculate_macd(data):
    """Calculate MACD (Trend Momentum)."""
    short_ema = data["Close"].ewm(span=12, adjust=False).mean()
    long_ema = data["Close"].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return "🟢 MACD: Bullish" if macd.iloc[-1] > signal.iloc[-1] else "🔴 MACD: Bearish"

def calculate_rsi(data):
    """Calculate RSI (Overbought/Oversold)."""
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_value = rsi.iloc[-1]
    if rsi_value > 70:
        return f"🔴 RSI {rsi_value:.1f} (Overbought - Risk of Drop)"
    elif rsi_value < 30:
        return f"🟢 RSI {rsi_value:.1f} (Oversold - Possible Buy)"
    return f"⚪ RSI {rsi_value:.1f} (Neutral)"

def fetch_market_sentiment():
    """Fetch financial news using NewsAPI and summarize sentiment using OpenAI."""
    if not NEWS_API_KEY:
        return "⚠ No NewsAPI key provided"
    try:
        url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        news_data = response.json()
        if news_data.get("status") != "ok":
            return "⚠ Error fetching news"
        headlines = [article["title"] for article in news_data.get("articles", [])[:5]]
        if not headlines:
            return "⚠ No recent financial headlines found"
        prompt = f"Summarize these financial headlines:\n{headlines}"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception:
        return "⚠ Unable to retrieve market news"

def generate_key_takeaways(market_summary, sentiment):
    """Use LLM to generate key takeaways from market data and sentiment."""
    prompt = f"""
    Based on the market trends and sentiment analysis, generate key takeaways and investment insights. 
    Your aim is to provide recommendations on when to buy back into the market, given recent volatility.
    Do not generate anything related to sentiment on specific companies or stocks.
    Market Summary:
    {market_summary}

    Market Sentiment:
    {sentiment}

    Keep the response short and actionable.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def fetch_market_data():
    """Fetch market data with 50/100/150-day MAs, RSI, and MACD."""
    market_summary = ""
    for name, symbol in MARKETS.items():
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="200d")
        if not hist.empty:
            latest_close = hist["Close"].iloc[-1]
            ma50 = hist["Close"].rolling(window=50).mean().iloc[-1]
            ma100 = hist["Close"].rolling(window=100).mean().iloc[-1]
            ma150 = hist["Close"].rolling(window=150).mean().iloc[-1]
            ma50_signal = "🟢" if latest_close > ma50 else "🔴"
            ma100_signal = "🟢" if latest_close > ma100 else "🔴"
            ma150_signal = "🟢" if latest_close > ma150 else "🔴"
            macd_signal = calculate_macd(hist)
            rsi_signal = calculate_rsi(hist)
            trend_emoji = "🟢" if latest_close > ma50 else "🔴"
            trend_status = (
                "📈 Strong Uptrend (Above MA50, MA100, MA150)" if latest_close > ma50 > ma100 > ma150 else
                "📉 Bearish Trend (Below MA50, MA100, MA150)" if latest_close < ma50 < ma100 < ma150 else
                "⚪ Mixed Signals (Market Unclear)"
            )
            market_summary += (
                f"• *{name}:* {trend_emoji} {trend_status}\n"
                f"  - {ma50_signal} MA50 | {ma100_signal} MA100 | {ma150_signal} MA150\n"
                f"  - {rsi_signal} | {macd_signal}\n\n"
            )
        else:
            market_summary += f"• *{name}:* ⚠ No Data\n\n"
    return market_summary.strip()

def generate_market_graph(symbol, market_name):
    """Generate a graph of the last 150 days with MA overlays and a MACD subplot."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="150d")
    if hist.empty:
        print(f"⚠ No data available for {market_name}.")
        return None

    # Compute moving averages
    hist["MA50"] = hist["Close"].rolling(window=50).mean()
    hist["MA100"] = hist["Close"].rolling(window=100).mean()
    hist["MA150"] = hist["Close"].rolling(window=150).mean()

    # Calculate MACD and Signal line
    short_ema = hist["Close"].ewm(span=12, adjust=False).mean()
    long_ema = hist["Close"].ewm(span=26, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    # Create figure with two subplots: top for price & MAs, bottom for MACD
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    # Top plot: Price and MAs
    ax1.plot(hist.index, hist["Close"], label=f"{market_name} Price", color="black", linewidth=2)
    ax1.plot(hist.index, hist["MA50"], label="50-Day MA", color="blue", linestyle="dashed")
    ax1.plot(hist.index, hist["MA100"], label="100-Day MA", color="green", linestyle="dashed")
    ax1.plot(hist.index, hist["MA150"], label="150-Day MA", color="red", linestyle="dashed")
    ax1.set_title(f"{market_name} - Last 150 Days with MAs")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)

    # Bottom plot: MACD
    ax2.plot(hist.index, macd_line, label="MACD", color="magenta", linewidth=2)
    ax2.plot(hist.index, signal_line, label="Signal", color="orange", linestyle="dashed")
    ax2.set_title("MACD")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    graph_path = f"{market_name.lower().replace(' ', '_')}_chart.png"
    plt.savefig(graph_path)
    plt.close()
    return graph_path

def send_telegram_message(message):
    """Send a message via Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=data)

def send_telegram_photo(photo_path):
    """Send a photo (market graph) to Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as photo:
        data = {"chat_id": TELEGRAM_CHAT_ID}
        requests.post(url, files={"photo": photo}, data=data)

if __name__ == "__main__":
    # Fetch market data and sentiment
    market_data = fetch_market_data()
    sentiment = fetch_market_sentiment()
    key_takeaways = generate_key_takeaways(market_data, sentiment)

    # Send market overview and key takeaways as separate messages
    send_telegram_message(f"📊 *Market Overview (50, 100, 150-Day MA)*\n\n{market_data}")
    send_telegram_message(f"💡 *Key Takeaways:*\n{key_takeaways}")
    send_telegram_message(f"{sentiment}")
    # Generate and send graphs for S&P 500, NASDAQ, and ASX 200
    for symbol, name in [(SP500_SYMBOL, "S&P 500"), (NASDAQ_SYMBOL, "NASDAQ"), (ASX200_SYMBOL, "ASX 200"), (GOLD_SYMBOL, "Gold")]:
        graph_path = generate_market_graph(symbol, name)
        if graph_path:
            send_telegram_photo(graph_path)
