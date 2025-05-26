from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# regex patterns
SYMBOLS = re.compile(r"[.;:!\'?,\"()\[\]]")
TAGS = re.compile(r"(<br\s*/><br\s*/>)|(\-)|(\/)")

POSITIVE = "Positive"
NEGATIVE = "Negative"
STOPWORDS = set(stopwords.words('english'))

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
        if polarity <= -0.05:  # Consider reviews with polarity <= -0.05 as negative
            sentiments.append(NEGATIVE)
        else:
            sentiments.append(POSITIVE)
    return sentiments

def extract_keywords_from_negative(reviews, sentiments):
    negative_reviews = [review for review, sentiment in zip(reviews, sentiments) if sentiment == NEGATIVE]
    all_words = []
    for review in negative_reviews:
        words = review.split()
        filtered = [word for word in words if word not in STOPWORDS and len(word) > 2]
        all_words.extend(filtered)
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
    plt.ylabel("Keywords")
    plt.tight_layout()
    plt.savefig("negative_keywords.png")
    plt.show()

def get_ngrams(reviews, n=2, top_n=20):
    # Filter out stopwords from the n-grams
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    X = vectorizer.fit_transform(reviews)
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]

def main():
    df = pd.read_csv('hogwarts_legacy_reviews.csv')
    df = df.dropna(subset=['Review'])
    processed_reviews = preprocess_reviews(df['Review'])
    sentiments = get_sentiment(processed_reviews)
    df['Sentiment'] = sentiments

    # Filter for only negative reviews
    negative_reviews_df = df[df['Sentiment'] == NEGATIVE]
    processed_negative_reviews = [review for review, sentiment in zip(processed_reviews, sentiments) if sentiment == NEGATIVE]

    # --- Top Keywords in Negative Reviews ---
    print("--- Top Keywords in Negative Reviews ---")
    word_freq_negative = extract_keywords_from_negative(processed_negative_reviews, sentiments)
    plot_keywords(word_freq_negative)

    # --- Top 2-Word N-grams in Negative Reviews ---
    print("\n--- Top 2-Word N-grams in Negative Reviews ---")
    top_bigrams_negative = get_ngrams(processed_negative_reviews, n=2, top_n=15)
    for phrase, freq in top_bigrams_negative:
        print(f"{phrase}: {freq}")

    # --- Top 3-Word N-grams in Negative Reviews ---
    print("\n--- Top 3-Word N-grams in Negative Reviews ---")
    top_trigrams_negative = get_ngrams(processed_negative_reviews, n=3, top_n=15)
    for phrase, freq in top_trigrams_negative:
        print(f"{phrase}: {freq}")

    # --- Thematic Analysis (Appearance Counts in Negative Reviews) ---
    print("\n--- Thematic Analysis in Negative Reviews ---")
    themes = {
        'music': ['sound', 'music', 'audio', 'instrument', 'soundtrack', 'voice acting', 'song', 'effect', 'atmosphere',
                  'orchestra', 'harry potter theme', 'magical melodies', 'spell sounds', 'ambient sounds'],
        'story': ['story', 'plot', 'narrative', 'character', 'mission', 'quest', 'writing', 'dialogue', 'relationships',
                  'family', 'gods', 'hogwarts', 'wizarding world', 'ancient magic', 'mystery', 'secrets', 'professors',
                  'students', 'villains', 'companions', 'house common room', 'choices', 'consequences'],
        'game play': ['gameplay', 'rogue-like', 'mechanics', 'controls', 'action', 'fight', 'attack', 'battle',
                      'weapon', 'moves', 'power', 'combat', 'upgrade', 'spells', 'dueling', 'exploration', 'flying',
                      'broom', 'potions', 'herbology', 'beasts', 'gear', 'talents', 'challenges', 'puzzles',
                      'side quests', 'collectibles', 'stealth', 'rpg elements', 'open world', 'crafting', 'merchants'],
        'visuals': ['visuals', 'graphics', 'art', 'images', 'color', 'artwork', 'animation', '2d', '3d', 'lighting',
                    'hogwarts castle', 'forbidden forest', 'hogsmeade', 'character models', 'spell effects',
                    'environmental details', 'fidelity', 'design', 'architecture', 'magical effects']
    }

    theme_appearance_counts = {theme: 0 for theme in themes}
    for review in processed_negative_reviews:
        for theme, words in themes.items():
            # Check if any word from the theme list is present in the review
            if any(word in review for word in words):
                theme_appearance_counts[theme] += 1

    # Plotting theme appearance counts for negative reviews
    theme_names = list(theme_appearance_counts.keys())
    counts = list(theme_appearance_counts.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=theme_names, y=counts, palette='viridis')
    plt.title('Theme Appearances in Negative Reviews')
    plt.xlabel('Themes')
    plt.ylabel('Number of Appearances')
    plt.tight_layout()
    plt.savefig("negative_theme_appearances.png")
    plt.show()
    plt.figure(figsize=(10, 6))
    polarity_scores = [get_polarity_score(review) for review in processed_reviews]
    sns.histplot(polarity_scores, bins=30, kde=True, color='purple')
    plt.title('Distribution of Sentiment Polarity Scores')
    plt.xlabel('Polarity Score (-1.0 to 1.0)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("polarity_distribution.png")
    plt.show()
    # --- Sentiment Distribution by Playtime Bin ---
    print("\n--- Sentiment Distribution by Playtime Bin ---")
    # Drop rows where 'Playtime' might still be NaN after initial dropna, or if 0 playtime
    df_filtered_playtime = df[df['Playtime'] > 0].copy()

    # Create 3 playtime bins (quantiles)
    df_filtered_playtime['Playtime_Bin'] = pd.qcut(
        df_filtered_playtime['Playtime'],
        q=3,
        labels=['Short Playtime', 'Medium Playtime', 'Long Playtime'],
        duplicates='drop'  # Handle cases with duplicate bin edges if data is sparse
    )

    # Group by Playtime_Bin and Sentiment, then count
    sentiment_by_playtime = df_filtered_playtime.groupby(['Playtime_Bin', 'Sentiment']).size().unstack(fill_value=0)

    # Plotting
    sentiment_by_playtime.plot(kind='bar', stacked=False, figsize=(12, 7),
                               color={'Positive': 'green', 'Negative': 'red'})
    plt.title('Positive vs. Negative Reviews by Playtime Category')
    plt.xlabel('Playtime Category')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig("sentiment_by_playtime.png")
    plt.show()

if __name__ == '__main__':
    main()