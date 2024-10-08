import pandas as pd
from textblob import TextBlob

data = {
    "comments": [
        "I love reading manga! It's so entertaining.",
        "Manhwa has the best art style, I prefer it over manga.",
        "I think both are great but manga is a bit better.",
        "Manga is just boring compared to manhwa.",
        "Manhwa is amazing! Can't get enough of it.",
        "I dislike the pacing of manga, it's too slow.",
        "Both manga and manhwa have their strengths.",
        "I find manhwa to be more visually appealing.",
        "Manga characters are more relatable to me.",
        "I don't enjoy reading manga at all.",
        "I love how detailed the artwork is in manhwa compared to manga!",
        "Manga has a unique storytelling style that just hits differently.",
        "Honestly, I find manhwa more appealing because of the color and style.",
        "Both have their merits, but I prefer manga for its narrative depth.",
        "Manhwa feels more modern and relatable, especially the romance stories.",
        "I don’t get why people argue about this; both are amazing in their own ways!",
        "Manga tends to have better character development, in my opinion.",
        "I enjoy the pacing of manhwa; it feels more relaxed and enjoyable.",
        "Manga's black-and-white art gives it a classic feel that I love.",
        "Some manhwa series are too long-winded; they drag out the plot unnecessarily.",
        "I appreciate the variety in themes and genres in both formats.",
        "Manga's history and tradition really resonate with me as a reader.",
        "I think manhwa does a better job of engaging younger audiences.",
        "Manga can sometimes feel rushed, especially in adaptations.",
        "The humor in manhwa is often more relatable and light-hearted.",
        "I prefer the reading experience of manga, flipping through the pages.",
        "Both have their unique charms; I enjoy switching between them.",
        "The emotional depth in some manga series just blows me away.",
        "I feel manhwa focuses more on art aesthetics than story depth.",
        "I like the cultural influences that come through in both manga and manhwa."
    ]
}

df = pd.DataFrame(data)

def get_sentiment(comment):
    analysis = TextBlob(comment)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['comments'].apply(get_sentiment)

sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100

print("Sentiment Analysis Results:")
print(df[['comments', 'sentiment']])
print("\nSummary of Results:")
print(f"Positive comments: {sentiment_counts.get('Positive', 0):.2f}%")
print(f"Negative comments: {sentiment_counts.get('Negative', 0):.2f}%")
print(f"Neutral comments: {sentiment_counts.get('Neutral', 0):.2f}%")
