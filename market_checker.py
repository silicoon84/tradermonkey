import os
import openai
import yfinance as yf
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import logging
import pandas as pd
from fredapi import Fred
import wbgapi as wb
import time
from functools import wraps
from yfinance.exceptions import YFRateLimitError
import pickle
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_checker.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Fetch credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # Add this to your .env file

# OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize FRED API client
fred = Fred(api_key=FRED_API_KEY)

# Market Symbols with Alpha Vantage symbols
MARKETS = {
    "S&P 500": {"yahoo": "^GSPC", "alpha_vantage": "SPY"},
    "NASDAQ": {"yahoo": "^IXIC", "alpha_vantage": "QQQ"},
    "Dow Jones": {"yahoo": "^DJI", "alpha_vantage": "DIA"},
    "ASX 200": {"yahoo": "^AXJO", "alpha_vantage": "EWA"},  # Using iShares MSCI Australia ETF as proxy
    "Gold": {"yahoo": "GC=F", "alpha_vantage": "GLD"},  # Using GLD ETF as proxy
    "US 10-Yr Bond Yield": {"yahoo": "^TNX", "alpha_vantage": "IEF"}  # Using IEF ETF as proxy
}

# Cache configuration
CACHE_DIR = Path('cache')
CACHE_EXPIRY = timedelta(minutes=15)  # Cache data for 15 minutes

def ensure_cache_dir():
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(exist_ok=True)

def get_cache_path(symbol: str, data_type: str) -> Path:
    """Get the cache file path for a symbol and data type."""
    return CACHE_DIR / f"{symbol}_{data_type}.pkl"

def save_to_cache(data, symbol: str, data_type: str):
    """Save data to cache."""
    ensure_cache_dir()
    cache_path = get_cache_path(symbol, data_type)
    cache_data = {
        'timestamp': datetime.now(),
        'data': data
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

def load_from_cache(symbol: str, data_type: str):
    """Load data from cache if it exists and is not expired."""
    cache_path = get_cache_path(symbol, data_type)
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check if cache is expired
        if datetime.now() - cache_data['timestamp'] > CACHE_EXPIRY:
            logger.info(f"Cache expired for {symbol} {data_type}")
            return None
        
        logger.info(f"Using cached data for {symbol} {data_type}")
        return cache_data['data']
    except Exception as e:
        logger.error(f"Error loading cache for {symbol} {data_type}: {str(e)}")
        return None

def retry_on_rate_limit(max_retries=3, delay=5):
    """Decorator to retry functions on rate limit errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except YFRateLimitError:
                    retries += 1
                    if retries == max_retries:
                        logger.error(f"Max retries ({max_retries}) reached for {func.__name__}")
                        raise
                    logger.warning(f"Rate limit hit, retrying in {delay} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

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
            model="gpt-4o-mini",
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
                data = json.load(f)
                logger.info("Successfully loaded market memory")
                logger.info(f"Number of historical entries: {len(data.get('history', []))}")
                return data
        logger.info("No market memory file found, creating new one")
        return {'history': []}
    except Exception as e:
        logger.error(f"Error loading market memory: {str(e)}")
        return {'history': []}

def save_market_memory(market_data, sentiment, key_takeaways):
    """Save current market data to memory file."""
    try:
        memory = load_market_memory()
        
        # Generate market history data
        market_history = {}
        for name, symbol_info in MARKETS.items():
            history = fetch_market_history(symbol_info['yahoo'])
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
        logger.info("Successfully saved market memory")
        logger.info(f"Current number of historical entries: {len(memory['history'])}")
    except Exception as e:
        logger.error(f"Error saving market memory: {str(e)}")

def fetch_alpha_vantage_data(symbol, function="TIME_SERIES_DAILY"):
    """Fetch market data from Alpha Vantage."""
    try:
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "outputsize": "full"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Error Message" in data:
            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            return None
            
        if "Note" in data and "API call frequency" in data["Note"]:
            logger.warning("Alpha Vantage API rate limit reached")
            return None
            
        # Convert to pandas DataFrame
        if function == "TIME_SERIES_DAILY":
            time_series = data.get("Time Series (Daily)")
            if not time_series:
                return None
                
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            return df
            
    except Exception as e:
        logger.error(f"Error fetching Alpha Vantage data: {str(e)}")
        return None

def fetch_market_data_with_fallback(symbol_info):
    """Fetch market data with fallback from Alpha Vantage to Yahoo Finance."""
    # Try Alpha Vantage first
    if ALPHA_VANTAGE_API_KEY:
        logger.info(f"Attempting to fetch data from Alpha Vantage for {symbol_info['alpha_vantage']}")
        data = fetch_alpha_vantage_data(symbol_info['alpha_vantage'])
        if data is not None:
            logger.info(f"Successfully fetched data from Alpha Vantage for {symbol_info['alpha_vantage']}")
            return data
    
    # Fallback to Yahoo Finance
    logger.info(f"Falling back to Yahoo Finance for {symbol_info['yahoo']}")
    try:
        ticker = yf.Ticker(symbol_info['yahoo'])
        data = ticker.history(period="200d")
        if not data.empty:
            logger.info(f"Successfully fetched data from Yahoo Finance for {symbol_info['yahoo']}")
            return data
    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance data: {str(e)}")
    
    return None

@retry_on_rate_limit(max_retries=3, delay=5)
def fetch_market_data():
    """Fetch market data with fallback options."""
    market_summary = ""
    
    # Add Fear & Greed Index at the top
    fear_greed = fetch_fear_and_greed_index()
    market_summary += f"*Market Sentiment:*\n{fear_greed}\n\n"
    
    for name, symbol_info in MARKETS.items():
        try:
            data = fetch_market_data_with_fallback(symbol_info)
            if data is not None and not data.empty:
                latest_close = data["Close"].iloc[-1]
                ma50 = data["Close"].rolling(window=50).mean().iloc[-1]
                ma100 = data["Close"].rolling(window=100).mean().iloc[-1]
                ma125 = data["Close"].rolling(window=125).mean().iloc[-1]
                
                # Calculate 100-day high/low percentages
                hundred_day_high = data["High"].max()
                hundred_day_low = data["Low"].min()
                down_from_high = ((hundred_day_high - latest_close) / hundred_day_high) * 100
                up_from_low = ((latest_close - hundred_day_low) / hundred_day_low) * 100
                
                ma50_signal = "🟢" if latest_close > ma50 else "🔴"
                ma100_signal = "🟢" if latest_close > ma100 else "🔴"
                ma125_signal = "🟢" if latest_close > ma125 else "🔴"
                macd_signal = calculate_macd(data)
                rsi_signal = calculate_rsi(data)
                trend_emoji = "🟢" if latest_close > ma50 else "🔴"
                trend_status = (
                    "📈 Strong Uptrend (Above MA50, MA100, MA125)" if latest_close > ma50 > ma100 > ma125 else
                    "📉 Bearish Trend (Below MA50, MA100, MA125)" if latest_close < ma50 < ma100 < ma125 else
                    "⚪ Mixed Signals (Market Unclear)"
                )
                
                market_summary += (
                    f"• *{name}:* {trend_emoji} {trend_status}\n"
                    f"  - {ma50_signal} MA50 | {ma100_signal} MA100 | {ma125_signal} MA125\n"
                    f"  - {rsi_signal} | {macd_signal}\n"
                    f"  - 📉 Down {down_from_high:.1f}% from 100d high | 📈 Up {up_from_low:.1f}% from 100d low\n\n"
                )
            else:
                market_summary += f"• *{name}:* ⚠ No Data\n\n"
            
            # Add a small delay between requests
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error processing {name}: {str(e)}")
            market_summary += f"• *{name}:* ⚠ Error: {str(e)}\n\n"
            continue
            
    return market_summary.strip()

@retry_on_rate_limit(max_retries=3, delay=5)
def fetch_market_history(symbol, days=50):
    """Fetch historical price data with fallback options."""
    symbol_info = next((info for info in MARKETS.values() if info['yahoo'] == symbol), None)
    if not symbol_info:
        return None
        
    data = fetch_market_data_with_fallback(symbol_info)
    if data is not None and not data.empty:
        # Calculate daily returns
        data['Daily_Return'] = data['Close'].pct_change() * 100
        # Calculate key metrics
        total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100).round(2)
        avg_daily_return = data['Daily_Return'].mean().round(2)
        volatility = data['Daily_Return'].std().round(2)
        positive_days = (data['Daily_Return'] > 0).sum()
        negative_days = (data['Daily_Return'] < 0).sum()
        
        return {
            'total_return': total_return,
            'avg_daily_return': avg_daily_return,
            'volatility': volatility,
            'positive_days': int(positive_days),
            'negative_days': int(negative_days),
            'current_price': data['Close'].iloc[-1].round(2),
            'start_price': data['Close'].iloc[0].round(2)
        }
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
        for name, symbol_info in MARKETS.items():
            history = fetch_market_history(symbol_info['yahoo'])
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
        logger.info("\n=== Historical Context ===")
        for day in memory['history'][-3:]:  # Use last 3 days for context
            historical_context += f"\nDate: {day['timestamp']}\n"
            historical_context += f"Previous Takeaways: {day['key_takeaways']}\n"
            logger.info(f"\nDate: {day['timestamp']}")
            logger.info(f"Previous Takeaways: {day['key_takeaways']}")
        logger.info("=== End Historical Context ===\n")
    
    # Add market history data
    market_history = generate_market_history_summary()
    
    # Fetch leading indicators
    leading_indicators = fetch_leading_indicators()
    indicators_summary = ""
    if leading_indicators:
        indicators_summary = "\nLeading Economic Indicators:\n"
        for indicator, data in leading_indicators.items():
            change_emoji = "📈" if data['change'] > 0 else "📉"
            indicators_summary += f"{change_emoji} {indicator}: {data['value']:.2f} ({data['change']:+.2f}%)\n"
    
    prompt = f"""
    Based on the market trends, sentiment analysis, historical context, and detailed market data, generate key takeaways and investment insights. 
    Your aim is to provide recommendations on when to buy back into the market, given recent volatility.
    Do not generate anything related to sentiment on specific companies or stocks.
    The historical context you've been provided is what you've previously generated. If its the same as the previous day or days, just provide a very concise summary of what you've previously said.
    
    Current Market Summary:
    {market_summary}

    Current Market Sentiment:
    {sentiment}
    
    {market_history}
    
    {indicators_summary}
    
    {historical_context}

    Please provide your analysis in two parts:

    1. EXECUTIVE SUMMARY (1-2 sentences):
    Start with a concise summary of your main recommendation and key market direction. Be brief but specific.

    2. DETAILED ANALYSIS:
    Then provide a concise analysis covering:
    - Market Direction & Volatility (1-2 lines)
    - Sentiment & Risk (1-2 lines)
    - Entry Points (1-2 lines)
    - Economic Indicators Impact (1-2 lines)
    
    Keep everything concise and actionable. Focus on the most important points only.
    """
    
    logger.info("Generating key takeaways using GPT-4o-mini...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    logger.info("Successfully generated key takeaways")
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

@retry_on_rate_limit(max_retries=3, delay=5)
def generate_market_graph(symbol, market_name):
    """Generate a graph with fallback options."""
    symbol_info = next((info for info in MARKETS.values() if info['yahoo'] == symbol), None)
    if not symbol_info:
        return None
        
    data = fetch_market_data_with_fallback(symbol_info)
    if data is None or data.empty:
        logger.error(f"⚠ No data available for {market_name}.")
        return None

    # Compute moving averages
    data["MA50"] = data["Close"].rolling(window=50).mean()
    data["MA100"] = data["Close"].rolling(window=100).mean()
    data["MA125"] = data["Close"].rolling(window=125).mean()

    # Calculate MACD and Signal line
    short_ema = data["Close"].ewm(span=12, adjust=False).mean()
    long_ema = data["Close"].ewm(span=26, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    # Calculate RSI
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Create figure with three subplots: price & MAs, MACD, and RSI
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Top plot: Price and MAs
    ax1.plot(data.index, data["Close"], label=f"{market_name} Price", color="black", linewidth=2)
    ax1.plot(data.index, data["MA50"], label="50-Day MA", color="blue", linestyle="dashed")
    ax1.plot(data.index, data["MA100"], label="100-Day MA", color="green", linestyle="dashed")
    ax1.plot(data.index, data["MA125"], label="125-Day MA", color="red", linestyle="dashed")
    ax1.set_title(f"{market_name} - Last 125 Days with MAs")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)

    # Middle plot: MACD
    ax2.plot(data.index, macd_line, label="MACD", color="magenta", linewidth=2)
    ax2.plot(data.index, signal_line, label="Signal", color="orange", linestyle="dashed")
    ax2.set_title("MACD")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True)

    # Bottom plot: RSI
    ax3.plot(data.index, rsi, label="RSI", color="purple", linewidth=2)
    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax3.set_title("RSI")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Value")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    graph_path = f"{market_name.lower().replace(' ', '_')}_chart.png"
    plt.savefig(graph_path)
    plt.close()
    
    # Save graph path to cache
    save_to_cache(graph_path, symbol, 'graph')
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

def fetch_abs_data(series_id, start_date):
    """Fetch data from ABS API."""
    try:
        # ABS API endpoint
        url = f"https://api.data.abs.gov.au/data/{series_id}"
        params = {
            'startPeriod': start_date,
            'format': 'json'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract values and dates
        values = []
        dates = []
        for observation in data['data']['dataSets'][0]['series']['0']['observations']['0']:
            values.append(float(observation))
        
        for observation in data['data']['dataSets'][0]['series']['0']['observations']['1']:
            dates.append(observation)
        
        # Create pandas Series
        series = pd.Series(values, index=pd.to_datetime(dates))
        return series
        
    except Exception as e:
        logger.error(f"Error fetching ABS data: {str(e)}")
        return None

def fetch_inflation_data():
    """Fetch US inflation data using FRED API."""
    try:
        # US CPI (Consumer Price Index) - Monthly data
        # CPIAUCSL - Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
        us_cpi = fred.get_series('CPIAUCSL')
        
        # Calculate US inflation rate (year-over-year change)
        us_inflation = ((us_cpi - us_cpi.shift(12)) / us_cpi.shift(12)) * 100
        
        # Get last 24 months of data
        us_inflation = us_inflation.tail(24)
        
        # Add some logging to debug the data
        logger.info(f"US Inflation data points: {len(us_inflation)}")
        logger.info(f"US Inflation values: {us_inflation.values}")
        
        return us_inflation
        
    except Exception as e:
        logger.error(f"Error fetching inflation data: {str(e)}")
        # Try alternative series if primary fails
        try:
            logger.info("Attempting to fetch alternative inflation series...")
            # Alternative US CPI series
            us_cpi = fred.get_series('CPILFESL')  # Core CPI (excluding food and energy)
            us_inflation = ((us_cpi - us_cpi.shift(12)) / us_cpi.shift(12)) * 100
            us_inflation = us_inflation.tail(24)
            
            logger.info("Successfully fetched alternative inflation series")
            return us_inflation
            
        except Exception as e2:
            logger.error(f"Error fetching alternative inflation data: {str(e2)}")
            return None

def generate_inflation_graph():
    """Generate a graph showing US inflation rate."""
    us_inflation = fetch_inflation_data()
    
    if us_inflation is None:
        logger.error("Failed to generate inflation graph due to missing data")
        return None
    
    plt.figure(figsize=(12, 6))
    
    # Plot US inflation
    plt.plot(us_inflation.index, us_inflation.values, label='US Inflation', color='blue', linewidth=2, marker='o')
    
    # Add target inflation lines
    plt.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='2% Target')
    plt.axhline(y=3.0, color='orange', linestyle='--', alpha=0.5, label='3% Target')
    
    plt.title('US Inflation Rate (24 Months)')
    plt.xlabel('Date')
    plt.ylabel('Inflation Rate (%)')
    plt.legend()
    plt.grid(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    graph_path = 'inflation_comparison.png'
    plt.savefig(graph_path)
    plt.close()
    
    return graph_path

def fetch_leading_indicators():
    """Fetch leading economic indicators from FRED."""
    try:
        # Dictionary of important economic indicators and their descriptions
        indicators = {
            'VIXCLS': 'CBOE Volatility Index (VIX)',
            'UMCSENT': 'Consumer Sentiment Index',
            'M2SL': 'M2 Money Stock',
            'INDPRO': 'Industrial Production Index',
            'UNRATE': 'Unemployment Rate',
            'FEDFUNDS': 'Federal Funds Rate',
            'GS10': '10-Year Treasury Rate',
            'BAA10Y': 'Moody\'s Seasoned Baa Corporate Bond Yield',
            'HOUST': 'Housing Starts',
            'RSXFS': 'Retail Sales',
            'PCE': 'Personal Consumption Expenditures',
            'GDP': 'Gross Domestic Product',
            'CPIAUCSL': 'Consumer Price Index',
            'PPIACO': 'Producer Price Index',
            'MNFCTRIRSA': 'Manufacturing Production Index'
        }
        
        indicator_data = {}
        for series_id, description in indicators.items():
            try:
                data = fred.get_series(series_id)
                # Get the latest value and calculate month-over-month change
                latest_value = data.iloc[-1]
                prev_value = data.iloc[-2]
                change_pct = ((latest_value - prev_value) / prev_value) * 100
                
                indicator_data[description] = {
                    'value': latest_value,
                    'change': change_pct,
                    'series_id': series_id
                }
            except Exception as e:
                logger.error(f"Error fetching {series_id}: {str(e)}")
                continue
        
        return indicator_data
        
    except Exception as e:
        logger.error(f"Error fetching leading indicators: {str(e)}")
        return None

def clear_cache():
    """Clear all cached data."""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob('*.pkl'):
            try:
                cache_file.unlink()
                logger.info(f"Cleared cache file: {cache_file}")
            except Exception as e:
                logger.error(f"Error clearing cache file {cache_file}: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting market analysis...")
    
    # Clear cache at the start of each run
    clear_cache()
    
    # Fetch market data and sentiment
    logger.info("Fetching market data...")
    market_data = fetch_market_data()
    logger.info("Fetching market sentiment...")
    sentiment = fetch_market_sentiment()
    
    logger.info("Generating key takeaways...")
    key_takeaways = generate_key_takeaways(market_data, sentiment)
    
    # Output key takeaways to console and log
    print("\n=== Market Analysis ===")
    print(key_takeaways)
    print("=====================\n")
    logger.info("\n=== Market Analysis ===")
    logger.info(key_takeaways)
    logger.info("=====================\n")
    
    logger.info("Saving market memory...")
    save_market_memory(market_data, sentiment, key_takeaways)
    
    logger.info("Sending Telegram messages...")
    # Split market data into smaller chunks if needed
    market_data_chunks = [market_data[i:i+4000] for i in range(0, len(market_data), 4000)]
    for chunk in market_data_chunks:
        send_telegram_message(f"📊 *Market Overview*\n\n{chunk}")
    
    # Send key takeaways
    send_telegram_message(f"💡 *Key Takeaways:*\n{key_takeaways}")
    
    # Send sentiment in chunks if needed
    sentiment_chunks = [sentiment[i:i+4000] for i in range(0, len(sentiment), 4000)]
    for chunk in sentiment_chunks:
        send_telegram_message(f"📰 *News Overview*\n{chunk}")
    
    # Generate and send graphs for S&P 500, NASDAQ, and ASX 200
    logger.info("Generating and sending market graphs...")
    for symbol, name in [(info['yahoo'], name) for name, info in MARKETS.items() if 'yahoo' in info]:
        graph_path = generate_market_graph(symbol, name)
        if graph_path:
            send_telegram_photo(graph_path)
            logger.info(f"Sent graph for {name}")
    
    # Generate and send inflation graph
    logger.info("Generating and sending inflation graph...")
    graph_path = generate_inflation_graph()
    if graph_path:
        send_telegram_photo(graph_path)
        logger.info("Sent inflation graph")
    
    logger.info("Market analysis completed successfully")
