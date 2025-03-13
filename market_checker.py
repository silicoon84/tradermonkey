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
MARKETS = {
    "S&P 500": SP500_SYMBOL,
    "NASDAQ": NASDAQ_SYMBOL,
    "Dow Jones": "^DJI",
    "ASX 200": "^AXJO",
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

        prompt = f"Summarize these financial headlines in simple terms:\n{headlines}"

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
    Based on the market trends and sentiment analysis, generate key takeaways and investment insights. Do not comment on any specific stock.

    Market Summary:
    {market_summary}

    Market Sentiment:
    {sentiment}

    Keep the response short and actionable.
    """

    response = client.chat.completions.create(
        model="gpt-4",
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

def send_telegram_message(message):
    """Send a message via Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=data)

if __name__ == "__main__":
    market_data = fetch_market_data()
    sentiment = fetch_market_sentiment()  # Used for key takeaways but NOT displayed
    key_takeaways = generate_key_takeaways(market_data, sentiment)

    send_telegram_message(f"📊 *Market Overview (50, 100, 150-Day MA)*\n\n{market_data}")
    send_telegram_message(f"💡 *Key Takeaways:*\n{key_takeaways}")
