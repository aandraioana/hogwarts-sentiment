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

from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
#stopwords are like the, is, and
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
def extract_keywords_from_negative(reviews, sentiments):
    negative_reviews = [review for review, sentiment in zip(reviews, sentiments) if sentiment == NEGATIVE]
    all_words = []

    for review in negative_reviews:
        words = review.split()
        filtered = [word for word in words if word not in STOPWORDS and len(word) > 2]
        all_words.extend(filtered)

    # Count most common words
    word_freq = Counter(all_words)
    return word_freq
def plot_keywords(word_freq, top_n=20):
    most_common = word_freq.most_common(top_n)
    words = [item[0] for item in most_common]
    counts = [item[1] for item in most_common]

    plt.figure(figsize=(10, 6))
    plt.barh(words[::-1], counts[::-1], color='red')
    plt.title("Top Keywords in Negative Reviews")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig("negative_keywords.png")
    plt.show()

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
from sklearn.feature_extraction.text import CountVectorizer

def get_ngrams(reviews, n=2, top_n=20):
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    X = vectorizer.fit_transform(reviews)
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]
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
    word_freq = extract_keywords_from_negative(processed_reviews, sentiments)
    plot_keywords(word_freq)
    negative_reviews = [review for review, sentiment in zip(processed_reviews, sentiments) if sentiment == NEGATIVE]

    # Get top bigrams (n=2) from negative reviews
    top_bigrams = get_ngrams(negative_reviews, n=3, top_n =20 )

    # Print or log the top bigrams
    print("Top Bigrams in Negative Reviews 3 words:")
    for phrase, freq in top_bigrams:
        print(f"{phrase}: {freq}")
    top_bigrams = get_ngrams(negative_reviews, n=2, top_n=15)

    # Print or log the top bigrams
    print("Top Bigrams in Negative Reviews 2 words:")
    for phrase, freq in top_bigrams:
        print(f"{phrase}: {freq}")
if __name__ == '__main__':
    main()