# %%
import pandas as pd
files = pd.read_csv("syndrome_aerotoxique - Sheet1.csv", header=1)
files.shape
# %%
files
# %%
# number of articles per month
import plotly.express as px 

fig = px.histogram(files, x = "date")
fig.update_layout(
    title = "Number of articles"
)
fig.show()
# %%

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import re
from langdetect import detect

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Get both French and English stopwords
french_stop = set(stopwords.words('french'))
english_stop = set(stopwords.words('english'))
combined_stopwords = french_stop.union(english_stop)

def detect_language(text):
    try:
        return detect(str(text))
    except:
        return 'en'  # default to English if detection fails

def preprocess_text(text):
   text = str(text)
   # Keep accented chars and apostrophes within words
   text = re.sub(r"([^a-zA-ZÀ-ÿ'\s])|(' )|( ')", ' ', text)
   text = text.lower()
   tokens = word_tokenize(text)
   tokens = [token for token in tokens if token not in combined_stopwords]
   return ' '.join(tokens)

def print_top_words_per_topic(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append({f"Topic {topic_idx + 1}": top_words})
    return topics

# Add language detection
files['language'] = files['content'].apply(detect_language)

# Preprocess the content
files['processed_content'] = files['content'].apply(preprocess_text)
# %% 
# Create TF-IDF representation with adjusted parameters for French
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),  # Include bigrams
    strip_accents=None,  # Keep accents
)
tfidf_matrix = vectorizer.fit_transform(files['processed_content'])
words_frequent = vectorizer.get_feature_names_out()
# %%
words_frequent.__len__()
wordsdf = pd.DataFrame(tfidf_matrix.toarray(), columns=[words_frequent])
wordsdf
# %%
wordsdf_freq = wordsdf.sum(axis = 0 )
wordsdf_freq.sort_values()
# %% 
px.bar(wordsdf_freq.renamesort_values().to_frame().reset_index(), x =  )
# Apply LDA
# %% 
n_topics = 20
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=20,
    learning_method='batch'
)
lda_output = lda.fit_transform(tfidf_matrix)

# %% 
# Get feature names and topics
feature_names = vectorizer.get_feature_names_out()
topics = print_top_words_per_topic(lda, feature_names)

# Add dominant topic to dataframe
dominant_topics = np.argmax(lda_output, axis=1) + 1
files['dominant_topic'] = dominant_topics

# Create summary DataFrame with language distribution
topic_summary = pd.DataFrame({
    'Topic': range(1, n_topics + 1),
    'Total_Articles': [sum(dominant_topics == i) for i in range(1, n_topics + 1)],
    'French_Articles': [sum((dominant_topics == i) & (files['language'] == 'fr')) for i in range(1, n_topics + 1)],
    'English_Articles': [sum((dominant_topics == i) & (files['language'] == 'en')) for i in range(1, n_topics + 1)]
})

# Print results
print("\nTop words in each topic:")
for topic in topics:
    print(topic)

print("\nTopic and language distribution:")
print(topic_summary)

# Show sample articles by topic and language
for topic_num in range(1, n_topics + 1):
    print(f"\nSample titles from Topic {topic_num}:")
    for lang in ['fr', 'en']:
        mask = (files['dominant_topic'] == topic_num) & (files['language'] == lang)
        sample_titles = files[mask]['title'].head(2)
        print(f"\n{lang.upper()} titles:")
        if not sample_titles.empty:
            print(sample_titles.to_string())
# %%


files