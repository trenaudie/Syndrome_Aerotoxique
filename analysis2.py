# %%
# Standard library imports
import re
from collections import Counter
from datetime import datetime

# Third-party imports: Core data science
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# NLP and text processing
import nltk
import spacy
from gensim import corpora, models
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# sklearn components
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK downloads
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

french_stop = set(stopwords.words('french'))
english_stop = set(stopwords.words('english'))
combined_stopwords = french_stop.union(english_stop)


def preprocess_text(text):
   text = str(text)
   # Keep accented chars and apostrophes within words
   text = re.sub(r"([^a-zA-ZÀ-ÿ'\s])|(' )|( ')", ' ', text)
   text = text.lower()
   tokens = word_tokenize(text)
   print(tokens)
   tokens = [token for token in tokens if token not in combined_stopwords]
   return ' '.join(tokens)

def detect_language(text):
    try:
        return detect(str(text))
    except:
        return 'en'  # default to English if detection fails

class PressCorpusAnalyzer:
    def __init__(self, df, content_column='processed_content'):
        self.df = df
        self.content_column = content_column
        self.nlp = spacy.load('en_core_web_sm')
        
    def basic_stats(self):
        """Calculate basic text statistics"""
        self.df['doc_length'] = self.df[self.content_column].str.len()
        self.df['word_count'] = self.df[self.content_column].str.split().str.len()
        self.df['avg_word_length'] = self.df[self.content_column].apply(
            lambda x: np.mean([len(w) for w in x.split()])
        )
        
        return {
            'total_documents': len(self.df),
            'avg_document_length': self.df['doc_length'].mean(),
            'avg_word_count': self.df['word_count'].mean(),
            'avg_word_length': self.df['avg_word_length'].mean()
        }
    
    def extract_key_phrases(self, n_terms=20):
        """Extract key phrases using TF-IDF"""
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        tfidf_matrix = tfidf.fit_transform(self.df[self.content_column])
        feature_names = tfidf.get_feature_names_out()
        
        # Get top terms for entire corpus
        mean_tfidf = tfidf_matrix.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[-n_terms:][::-1]
        
        return [(feature_names[i], mean_tfidf[i]) for i in top_indices]
    
    def topic_modeling(self, n_topics=5):
        """Perform topic modeling using NMF"""
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        
        tfidf_matrix = tfidf.fit_transform(self.df[self.content_column])
        feature_names = tfidf.get_feature_names_out()
        
        nmf = NMF(n_components=n_topics, random_state=42)
        topic_matrix = nmf.fit_transform(tfidf_matrix)
        
        topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_terms = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            topics.append({
                'topic_id': topic_idx,
                'terms': top_terms,
                'prevalence': np.mean(topic_matrix[:, topic_idx])
            })
        
        return topics
    
    def named_entity_analysis(self):
        """Extract and analyze named entities"""
        entities = []
        for text in self.df[self.content_column]:
            doc = self.nlp(text)
            entities.extend([(ent.text, ent.label_) for ent in doc.ents])
        
        entity_counts = Counter(entities)
        
        # Organize entities by type
        entity_by_type = {}
        for (text, label), count in entity_counts.items():
            if label not in entity_by_type:
                entity_by_type[label] = []
            entity_by_type[label].append((text, count))
        
        # Sort each type by frequency
        for label in entity_by_type:
            entity_by_type[label] = sorted(
                entity_by_type[label],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        
        return entity_by_type
    
    def sentiment_analysis(self):
        """Perform sentiment analysis"""
        sentiments = []
        for text in self.df[self.content_column]:
            blob = TextBlob(text)
            sentiments.append({
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            })
        
        sentiment_df = pd.DataFrame(sentiments)
        
        return {
            'avg_polarity': sentiment_df['polarity'].mean(),
            'avg_subjectivity': sentiment_df['subjectivity'].mean(),
            'sentiment_distribution': {
                'positive': len(sentiment_df[sentiment_df['polarity'] > 0]),
                'neutral': len(sentiment_df[sentiment_df['polarity'] == 0]),
                'negative': len(sentiment_df[sentiment_df['polarity'] < 0])
            }
        }
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'basic_stats': self.basic_stats(),
            'key_phrases': self.extract_key_phrases(),
            'topics': self.topic_modeling(),
            'named_entities': self.named_entity_analysis(),
            'sentiment': self.sentiment_analysis()
        }
        return report

# Example usage:
# analyzer = PressCorpusAnalyzer(files)
# report = analyzer.generate_report()
# %% 
# %%

files = pd.read_csv("syndrome_aerotoxique - Sheet1.csv", header=1)
files['language'] = files['content'].apply(detect_language)

# Preprocess the content
files['processed_content'] = files['content'].apply(preprocess_text)
# %%
# Initialize and run analysis
analyzer = PressCorpusAnalyzer(files)
report = analyzer.generate_report()

# Access specific analyses
topics = analyzer.topic_modeling(n_topics=7)
entities = analyzer.named_entity_analysis()
# %%





def sentiment_distrib_graph(sentiment_distrib:dict):
    """
    keys : positive, neutral, negative
    values : number of articles 
    """
    # Create translation mapping
    sentiment_translation = {
        'positive': 'positif',
        'neutral': 'neutre',
        'negative': 'négatif'
    }

    # Convert dictionary to DataFrame and translate sentiments
    sentiment_df = pd.DataFrame({
        'Sentiment': sentiment_distrib.keys(),
        'Count': sentiment_distrib.values()
    })
    sentiment_df['Sentiment_FR'] = sentiment_df['Sentiment'].map(sentiment_translation)

    # Create the bar plot with French labels
    fig = px.bar(
        sentiment_df,
        x='Sentiment_FR',
        y='Count',
        color='Sentiment_FR',
        color_discrete_map={
            'positif': '#2ecc71',   # Green
            'neutre': '#f1c40f',    # Yellow
            'négatif': '#e74c3c'    # Red
        },
        title="Prise de position des articles 'syndrome aérotoxique'"
    )

    # Improve layout with French labels
    fig.update_layout(
        showlegend=False,
        xaxis_title='Sentiment',
        yaxis_title="Nombre d'articles",
        template='plotly_white'
    )

    fig.show()

sentiment_distrib = report["sentiment"]["sentiment_distribution"]
# sentiment_distrib_graph(sentiment_distrib)
# %%

topics