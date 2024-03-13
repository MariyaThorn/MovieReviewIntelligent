from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

app = Flask(__name__)

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)['compound']
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        df = pd.read_csv(file)
        df_subset = df.head(500)
        df_subset['Sentiment'] = df_subset['text'].apply(analyze_sentiment)
        sentiment_counts = df_subset['Sentiment'].value_counts()
        sentiment_results = {
            'positive': sentiment_counts.get('Positive', 0),
            'negative': sentiment_counts.get('Negative', 0),
            'neutral': sentiment_counts.get('Neutral', 0)
        }
        return render_template('analysis.html', sentiment_results=sentiment_results)
    else:
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
