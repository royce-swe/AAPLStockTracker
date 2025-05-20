from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from sentiment import analyze_sentiment
from newsapi import NewsApiClient
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)  # This will allow all origins; for production, you might want to limit the allowed origins

# Initialize API keys
newsapi = NewsApiClient(api_key='b3e6bd1cba854d27bbeab04402585b58')
ALPHA_VANTAGE_API_KEY = 'K2NH2WG16X1H4L5P'

# Route for getting stock data
@app.route('/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    r = requests.get(url)
    print(f"Alpha Vantage status: {r.status_code}")
    print(f"Alpha Vantage response: {r.text}")
    
    try:
        data = r.json()
        time_series = data['Time Series (Daily)']
        return jsonify(time_series)
    except Exception as e:
        print(f"Error parsing Alpha Vantage JSON: {e}")
        return jsonify({'error': 'Failed to fetch stock data'}), 500


# Route for sentiment analysis
@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    text = request.json.get('text', '')
    score = analyze_sentiment(text)
    return jsonify({'sentiment_score': score})

# Route for news headlines
@app.route('/news/<symbol>', methods=['GET'])
def get_news(symbol):
    company = yf.Ticker(symbol).info.get("longName", symbol)
    print(f"Searching news for: {company}")
    
    try:
        articles = newsapi.get_everything(q=company, language='en', sort_by='publishedAt', page_size=5)
        print(f"NewsAPI response: {articles}")
        headlines = [article['title'] for article in articles['articles']]
        return jsonify({"headlines": headlines})
    except Exception as e:
        print(f"Error fetching or parsing NewsAPI data: {e}")
        return jsonify({"headlines": [], "error": str(e)}), 500



# Route for fetching stock price from Yahoo Finance
@app.route('/price/<symbol>', methods=['GET'])
def get_stock_price(symbol):
    try:
        print(f"Fetching data for: {symbol}")  # Debug statement
        stock = yf.Ticker(symbol)
        info = stock.info
        price = info.get('regularMarketPrice', None)
        name = info.get('shortName', symbol)

        if price is None:
            print(f"Price data not available for {symbol}")  # Debug statement
            return jsonify({'error': 'Price data not available'}), 404

        print(f"Price for {symbol}: {price}")  # Debug statement
        return jsonify({
            'symbol': symbol.upper(),
            'name': name,
            'price': price
        })
    except Exception as e:
        print(f"Error fetching price for {symbol}: {str(e)}")  # Debug statement
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<symbol>', methods=['GET'])
def predict_price(symbol):
    try:
        # Step 1: Get recent news headlines
        company = yf.Ticker(symbol).info.get("longName", symbol)
        articles = newsapi.get_everything(q=company, language='en', sort_by='publishedAt', page_size=5)

        headlines = [article['title'] for article in articles['articles']]
        sentiment_scores = [analyze_sentiment(h) for h in headlines]

        if not sentiment_scores:
            return jsonify({'error': 'No sentiment data available'}), 500

        avg_sentiment = np.mean(sentiment_scores)

        # Step 2: Get recent stock prices
        stock = yf.Ticker(symbol)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=10)

        hist = stock.history(period='10d')
        if len(hist) < 2:
            return jsonify({'error': 'Not enough stock data'}), 500

        closes = hist['Close'].values[-5:]  # last 5 closing prices
        returns = np.diff(closes) / closes[:-1]  # daily returns

        # Step 3: Create model
        # Align sentiment scores and returns
        min_len = min(len(sentiment_scores), len(returns))
        X = np.array(sentiment_scores[-min_len:]).reshape(-1, 1)
        y = returns[-min_len:]

        model = LinearRegression().fit(X, y)
        predicted_return = model.predict([[avg_sentiment]])[0]

        print("Dates from Yahoo Finance history:")
        print(hist.index)

        return jsonify({
            'symbol': symbol.upper(),
            'avg_sentiment': avg_sentiment,
            'predicted_daily_return': predicted_return,
            'latest_price': closes[-1]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Starting the Flask app
if __name__ == '__main__':
    app.run(debug=True)
