

from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import re
#regex patterns
SYMBOLS = re.compile(r"[.;:!\'?,\"()\[\]]")
TAGS = re.compile(r"(<br\s*/><br\s*/>)|(\-)|(\/)")

POSITIVE = "Positive"
NEGATIVE = "Negative"
NEUTRAL = "Neutral"

# Load CSV file
def load_data(file):
    df = pd.read_csv(file)
    return df['Review']

# Preprocess reviews: lowercase, remove symbols and tags
def preprocess_reviews(reviews):
    reviews = reviews.dropna()
    cleaned = []
    for line in reviews:
        line = str(line).lower()
        line = SYMBOLS.sub("", line)
        line = TAGS.sub(" ", line)
        cleaned.append(line)
    return cleaned


def get_polarity_score(sentence):
    return TextBlob(sentence).sentiment.polarity



def get_sentiment(reviews):
    sentiments = []
    for review in reviews:
        polarity = get_polarity_score(review)
        if polarity >= 0.05:
            sentiments.append(POSITIVE)
        elif polarity <= -0.05:
            sentiments.append(NEGATIVE)
        else:
            sentiments.append(NEUTRAL)
    return sentiments


def write_sentiment(sentiments, csv_file):
    df = pd.read_csv(csv_file)
    df['Sentiment'] = sentiments
    df.to_excel("output.xlsx", index=False, engine='openpyxl')


def count(sentiments):
    return [
        sentiments.count(POSITIVE),
        sentiments.count(NEGATIVE),
        sentiments.count(NEUTRAL)
    ]


def plot_pie(sentiments):
    sentiment_counts = count(sentiments)
    df = pd.DataFrame({'Sentiment': sentiment_counts},
                      index=[POSITIVE, NEGATIVE, NEUTRAL])
    df.plot.pie(y='Sentiment', figsize=(6, 6), autopct='%1.1f%%', startangle=90)
    plt.title("Sentiment Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("statistics.png")
    plt.show()

# Main
def main():

    df = pd.read_csv('hogwarts_legacy_reviews.csv')
    df = df.dropna(subset=['Review'])
    processed_reviews = preprocess_reviews(df['Review'])
    sentiments = get_sentiment(processed_reviews)
    df['Sentiment'] = sentiments

    # Save to CSV
    df.to_csv("output.csv", index=False)
    plot_pie(sentiments)

if __name__ == '__main__':
    main()
