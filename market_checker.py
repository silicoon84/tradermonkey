import os
import openai
import yfinance as yf
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import json

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
    """Fetch financial news from multiple sources and summarize market-moving news using OpenAI."""
    if not NEWS_API_KEY:
        return "⚠ No NewsAPI key provided"
    
    try:
        # Fetch from multiple sources
        sources = [
            "reuters",
            "bloomberg",
            "financial-times",
            "the-wall-street-journal",
            "cnbc"
        ]
        
        all_headlines = []
        for source in sources:
            url = f"https://newsapi.org/v2/top-headlines?sources={source}&language=en&apiKey={NEWS_API_KEY}"
            response = requests.get(url)
            news_data = response.json()
            
            if news_data.get("status") == "ok":
                articles = news_data.get("articles", [])
                for article in articles:
                    # Only include articles with market-related keywords
                    title = article["title"].lower()
                    keywords = ["market", "stocks", "economy", "fed", "inflation", "gdp", "trade", "dow", "s&p", "nasdaq"]
                    if any(keyword in title for keyword in keywords):
                        all_headlines.append({
                            "title": article["title"],
                            "source": source,
                            "url": article["url"]
                        })
        
        if not all_headlines:
            return "⚠ No relevant market news found"
        
        # Format headlines for analysis
        headlines_text = "\n".join([f"{h['title']} (Source: {h['source']})" for h in all_headlines])
        
        # Use GPT-4 to analyze and filter market-moving news
        prompt = f"""
        Analyze these financial headlines and identify only the market-moving news that could significantly impact markets.
        Focus on major economic events, policy changes, and significant market movements.
        Ignore minor market fluctuations and company-specific news unless they have broad market implications.
        
        Headlines:
        {headlines_text}
        
        Provide a concise summary of only the most impactful market-moving news, organized by category:
        1. Major Economic Events
        2. Policy Changes
        3. Significant Market Movements
        4. Key Market Indicators
        
        Keep the summary focused and actionable, highlighting only news that could influence market direction.
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        
        # Add source links to the summary
        summary = response.choices[0].message.content
        summary += "\n\nSources:\n"
        for headline in all_headlines:
            summary += f"- {headline['title']}: {headline['url']}\n"
        
        return summary
        
    except Exception as e:
        print(f"Error fetching market sentiment: {str(e)}")
        return "⚠ Unable to retrieve market news"

def load_market_memory():
    """Load previous market data from memory file."""
    try:
        if os.path.exists('market_memory.json'):
            with open('market_memory.json', 'r') as f:
                return json.load(f)
        return {'history': []}
    except Exception as e:
        print(f"Error loading market memory: {str(e)}")
        return {'history': []}

def save_market_memory(market_data, sentiment, key_takeaways):
    """Save current market data to memory file."""
    try:
        memory = load_market_memory()
        
        # Generate market history data
        market_history = {}
        for name, symbol in MARKETS.items():
            history = fetch_market_history(symbol)
            if history:
                market_history[name] = history
        
        current_data = {
            'timestamp': datetime.now().isoformat(),
            'market_data': market_data,
            'sentiment': sentiment,
            'key_takeaways': key_takeaways,
            'market_history': market_history
        }
        memory['history'].append(current_data)
        # Keep only last 7 days of history
        memory['history'] = memory['history'][-7:]
        with open('market_memory.json', 'w') as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        print(f"Error saving market memory: {str(e)}")

def fetch_market_history(symbol, days=50):
    """Fetch historical price data for a market."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{days}d")
        if not hist.empty:
            # Calculate daily returns
            hist['Daily_Return'] = hist['Close'].pct_change() * 100
            # Calculate key metrics
            total_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100).round(2)
            avg_daily_return = hist['Daily_Return'].mean().round(2)
            volatility = hist['Daily_Return'].std().round(2)
            positive_days = (hist['Daily_Return'] > 0).sum()
            negative_days = (hist['Daily_Return'] < 0).sum()
            
            return {
                'total_return': total_return,
                'avg_daily_return': avg_daily_return,
                'volatility': volatility,
                'positive_days': int(positive_days),
                'negative_days': int(negative_days),
                'current_price': hist['Close'].iloc[-1].round(2),
                'start_price': hist['Close'].iloc[0].round(2)
            }
    except Exception as e:
        print(f"Error fetching history for {symbol}: {str(e)}")
    return None

def generate_market_history_summary():
    """Generate a summary of historical market data."""
    # Load historical data from memory
    memory = load_market_memory()
    history_summary = "\nMarket History (Last 50 Days):\n"
    
    if memory['history']:
        # Use the most recent market history data
        latest_data = memory['history'][-1]
        market_history = latest_data.get('market_history', {})
        
        for name, history in market_history.items():
            history_summary += f"\n{name}:\n"
            history_summary += f"- Price Movement: {history['start_price']} → {history['current_price']} ({history['total_return']}%)\n"
            history_summary += f"- Average Daily Return: {history['avg_daily_return']}%\n"
            history_summary += f"- Volatility: {history['volatility']}%\n"
            history_summary += f"- Up/Down Days: {history['positive_days']}/{history['negative_days']}\n"
    else:
        # If no memory data, fetch fresh data
        for name, symbol in MARKETS.items():
            history = fetch_market_history(symbol)
            if history:
                history_summary += f"\n{name}:\n"
                history_summary += f"- Price Movement: {history['start_price']} → {history['current_price']} ({history['total_return']}%)\n"
                history_summary += f"- Average Daily Return: {history['avg_daily_return']}%\n"
                history_summary += f"- Volatility: {history['volatility']}%\n"
                history_summary += f"- Up/Down Days: {history['positive_days']}/{history['negative_days']}\n"
    
    return history_summary

def generate_key_takeaways(market_summary, sentiment):
    """Use LLM to generate key takeaways from market data and sentiment."""
    # Load historical context
    memory = load_market_memory()
    historical_context = ""
    
    if memory['history']:
        historical_context = "\nAnalysis History (Last 7 Days):\n"
        for day in memory['history'][-3:]:  # Use last 3 days for context
            historical_context += f"\nDate: {day['timestamp']}\n"
            historical_context += f"Previous Takeaways: {day['key_takeaways']}\n"
    
    # Add market history data
    market_history = generate_market_history_summary()
    
    prompt = f"""
    Based on the market trends, sentiment analysis, historical context, and detailed market data, generate key takeaways and investment insights. 
    Your aim is to provide recommendations on when to buy back into the market, given recent volatility.
    Do not generate anything related to sentiment on specific companies or stocks.
    
    Current Market Summary:
    {market_summary}

    Current Market Sentiment:
    {sentiment}
    
    {market_history}
    
    {historical_context}

    Please analyze:
    1. Overall market direction based on price movements and volatility
    2. Market sentiment shifts compared to previous days
    3. Risk levels based on volatility metrics
    4. Potential entry points considering both technical and sentiment data
    
    Keep the response actionable and data-driven, considering both historical price action and sentiment trends.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def fetch_fear_and_greed_index():
    """Fetch the CNN Fear & Greed Index for stocks."""
    try:
        # Using CNN's Fear & Greed Index endpoint
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data and 'fear_and_greed' in data:
                score = round(float(data['fear_and_greed']['score']), 2)
                rating = data['fear_and_greed']['rating']
                
                emoji = {
                    'Extreme Fear': '😨',
                    'Fear': '😰',
                    'Neutral': '😐',
                    'Greed': '😊',
                    'Extreme Greed': '😄'
                }.get(rating, '❓')
                
                return f"{emoji} Fear & Greed Index: {score} ({rating})"
        return "⚠ Unable to fetch Fear & Greed Index"
    except Exception as e:
        print(f"Error fetching Fear & Greed Index: {str(e)}")
        return "⚠ Error fetching Fear & Greed Index"

def fetch_market_data():
    """Fetch market data with 50/100/125-day MAs, RSI, and MACD."""
    market_summary = ""
    
    # Add Fear & Greed Index at the top
    fear_greed = fetch_fear_and_greed_index()
    market_summary += f"*Market Sentiment:*\n{fear_greed}\n\n"
    
    for name, symbol in MARKETS.items():
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="200d")
        if not hist.empty:
            latest_close = hist["Close"].iloc[-1]
            ma50 = hist["Close"].rolling(window=50).mean().iloc[-1]
            ma100 = hist["Close"].rolling(window=100).mean().iloc[-1]
            ma125 = hist["Close"].rolling(window=125).mean().iloc[-1]
            ma50_signal = "🟢" if latest_close > ma50 else "🔴"
            ma100_signal = "🟢" if latest_close > ma100 else "🔴"
            ma125_signal = "🟢" if latest_close > ma125 else "🔴"
            macd_signal = calculate_macd(hist)
            rsi_signal = calculate_rsi(hist)
            trend_emoji = "🟢" if latest_close > ma50 else "🔴"
            trend_status = (
                "📈 Strong Uptrend (Above MA50, MA100, MA125)" if latest_close > ma50 > ma100 > ma125 else
                "📉 Bearish Trend (Below MA50, MA100, MA125)" if latest_close < ma50 < ma100 < ma125 else
                "⚪ Mixed Signals (Market Unclear)"
            )
            market_summary += (
                f"• *{name}:* {trend_emoji} {trend_status}\n"
                f"  - {ma50_signal} MA50 | {ma100_signal} MA100 | {ma125_signal} MA125\n"
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
    hist["MA125"] = hist["Close"].rolling(window=125).mean()

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
    ax1.plot(hist.index, hist["MA125"], label="125-Day MA", color="red", linestyle="dashed")
    ax1.set_title(f"{market_name} - Last 125 Days with MAs")
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

    # Save current data to memory
    save_market_memory(market_data, sentiment, key_takeaways)

    # Send market overview and key takeaways as separate messages
    send_telegram_message(f"📊 *Market Overview (50, 100, 125-Day MA)*\n\n{market_data}")
    send_telegram_message(f"💡 *Key Takeaways:*\n{key_takeaways}")
    send_telegram_message(f" *News Overview*\n{sentiment}")
    # Generate and send graphs for S&P 500, NASDAQ, and ASX 200
    for symbol, name in [(SP500_SYMBOL, "S&P 500"), (NASDAQ_SYMBOL, "NASDAQ"), (ASX200_SYMBOL, "ASX 200"), (GOLD_SYMBOL, "Gold")]:
        graph_path = generate_market_graph(symbol, name)
        if graph_path:
            send_telegram_photo(graph_path)
