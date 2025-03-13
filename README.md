# 📊 TradeMonkey - Market Trend Analyzer & Telegram Bot

**TradeMonkey** is an automated market trend analyzer that tracks key stock indices, calculates moving averages, RSI, MACD, and provides insights via Telegram. It also generates charts for major indices and sends them to a Telegram group.

## 🚀 Features

- Fetches key stock market data (S&P 500, NASDAQ, ASX 200, etc.)
- Calculates technical indicators:
  - 50, 100, and 150-day Moving Averages
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
- Generates and sends trading signals
- Fetches market sentiment from news sources (via NewsAPI)
- Uses GPT-4 to generate key takeaways based on market trends
- Sends automated Telegram updates with market data & graphs
- Generates charts with price trends & MACD analysis

## 🛠 Installation & Setup

### 1️⃣ Clone the Repository  
Run the following command to clone the repository and enter the project folder:  

```
git clone https://github.com/silicoon84/tradermonkey.git
cd tradermonkey
```

### 2️⃣ Create a Virtual Environment (Recommended)  
Set up a Python virtual environment and activate it:  

```
python3 -m venv market_env
source market_env/bin/activate
```

### 3️⃣ Install Dependencies  
Install the required dependencies using pip:  

```
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables  
Create a `.env` file in the project directory with the following content:  

```
OPENAI_API_KEY=your_openai_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
NEWS_API_KEY=your_newsapi_key
```

Save the file.

## 📌 Usage

### Run the script manually  
Execute the script to fetch market data and send updates to Telegram:  

```
python market_checker.py
```

### Automate the script using Cron (Linux)  
To schedule the script to run every morning at 7 AM, open the crontab editor:  

```
crontab -e
```

Add the following line at the bottom:  

```
0 7 * * * /path/to/market_env/bin/python3 /path/to/tradermonkey/market_checker.py >> /path/to/tradermonkey/market_checker.log 2>&1
```

Save and exit.

## 📊 Example Output in Telegram  

**Market Overview (50, 100, 150-Day MA)**  

```
• S&P 500: 🔴 Mixed Signals (Market Unclear)
  - 🟢 MA50 | 🟢 MA100 | 🔴 MA150
  - ⚪ RSI 48.3 (Neutral) | 🔴 MACD: Bearish
```

**Key Takeaways**  

```
- S&P 500 shows mixed signals but remains under long-term resistance.
- Nasdaq and ASX 200 are below key moving averages, suggesting caution.
- RSI indicates no strong buy/sell signals, and MACD remains bearish.
```

**Generated Graphs:**  
- Price + Moving Averages  
- MACD Analysis  

## 🔧 Configuration & Customization  

### Modify Markets to Track  
To add or remove markets, edit the `MARKETS` dictionary inside `market_checker.py`.  

### Change the Telegram Output Format  
Modify the `send_telegram_message()` function in `market_checker.py` to adjust the message format.

## 📜 License  
This project is licensed under the **MIT License**.

