import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

# Load dataset
data = pd.read_excel("dataset sentimen/brics_processing.xlsx")

# Sidebar untuk memilih jenis visualisasi
st.sidebar.header('Visualisasi Sentimen BRICS')
option = st.sidebar.radio('Pilih Visualisasi:', ['Barchart Sentimen', 'Wordcloud Sentimen', 'Trigrams Sentimen', 'Top Words Sentimen'])

# Warna untuk sentimen
colors = {
    'Positif': 'green',
    'Negatif': 'red',
    'Netral': 'blue'
}

# Visualisasi Barchart Sentimen
if option == 'Barchart Sentimen':
    jumlah_sentimen = data['sentimen'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = sns.barplot(
        x=jumlah_sentimen.index,
        y=jumlah_sentimen.values,
        palette=[colors[label] for label in jumlah_sentimen.index],
        ax=ax
    )
    for bar, count in zip(bars.patches, jumlah_sentimen.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f'{count}',
            ha='center', va='center', color='white', fontweight='bold'
        )
    ax.set_title("Sentimen BRICS")
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

# Visualisasi Wordcloud Sentimen
elif option == 'Wordcloud Sentimen':
    sentimen_option = st.sidebar.selectbox('Pilih Sentimen:', ['Positif', 'Negatif', 'Netral'])
    text_column = 'stemming_text' if sentimen_option != 'Positif' else 'stopwords_text'

    sentimen_text = ' '.join(data[data['sentimen'] == sentimen_option][text_column])
    colormap = 'Greens' if sentimen_option == 'Positif' else 'Reds' if sentimen_option == 'Negatif' else 'Blues'

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(sentimen_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"Word Cloud Sentimen {sentimen_option} BRICS")
    st.pyplot(fig)

# Visualisasi Trigrams Sentimen
elif option == 'Trigrams Sentimen':
    sentimen_option = st.sidebar.selectbox('Pilih Sentimen:', ['Positif', 'Negatif', 'Netral'])

    def get_top_trigrams(data, sentimen, n=5):
        sentimen_data = data[data['sentimen'] == sentimen]
        vectorizer = CountVectorizer(ngram_range=(3, 3))
        trigrams_matrix = vectorizer.fit_transform(sentimen_data['stemming_text'])
        trigrams = vectorizer.get_feature_names_out()
        freqs = trigrams_matrix.sum(axis=0).A1
        trigram_freq = dict(zip(trigrams, freqs))
        return Counter(trigram_freq).most_common(n)

    top_trigrams = get_top_trigrams(data, sentimen_option)
    trigrams, frequencies = zip(*top_trigrams)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(frequencies), y=list(trigrams), color=colors[sentimen_option], ax=ax)
    ax.set_title(f"Top Trigrams Sentimen {sentimen_option}")
    ax.set_xlabel("Frekuensi")
    ax.set_ylabel("Trigrams")
    st.pyplot(fig)

# Visualisasi Top Words Sentimen
elif option == 'Top Words Sentimen':
    sentimen_option = st.sidebar.selectbox('Pilih Sentimen:', ['Positif', 'Negatif', 'Netral'])

    def get_word_frequency(data, sentimen, top_n=10):
        sentimen_data = data[data['sentimen'] == sentimen]
        vectorizer = CountVectorizer()
        word_matrix = vectorizer.fit_transform(sentimen_data['stemming_text'])
        words = vectorizer.get_feature_names_out()
        freqs = word_matrix.sum(axis=0).A1
        words_freq = dict(zip(words, freqs))
        return sorted(words_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]

    top_words = get_word_frequency(data, sentimen_option)
    words, frequencies = zip(*top_words)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=list(frequencies), y=list(words), color=colors[sentimen_option], ax=ax)
    ax.set_title(f"Top Words Sentimen {sentimen_option}")
    ax.set_xlabel("Frekuensi")
    ax.set_ylabel("Kata")
    st.pyplot(fig)