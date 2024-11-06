# Press Analysis Pipeline

This project implements a comprehensive text analysis pipeline for analyzing press articles, with support for both French and English content. It includes sentiment analysis, topic modeling, named entity recognition, and text embeddings generation.

## Setup

### Required Dependencies
```bash
pip install numpy pandas matplotlib plotly seaborn nltk spacy gensim langdetect textblob scikit-learn sentence-transformers
```

### NLTK Downloads
The following NLTK resources are required:
```python
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### Spacy Model
```bash
python -m spacy download en_core_web_sm
```

## Core Components

### 1. Text Preprocessing
- Handles both French and English text
- Removes special characters while preserving accented characters and meaningful apostrophes
- Language detection
- Stopword removal (French and English)

### 2. PressCorpusAnalyzer Class
The main analysis class provides several key functionalities:
- Basic text statistics (document length, word count, etc.)
- Key phrase extraction using TF-IDF
- Topic modeling using Non-negative Matrix Factorization (NMF)
- Named Entity Recognition using spaCy
- Sentiment Analysis using TextBlob

### 3. Text Embeddings
Uses the bilingual sentence transformer model for generating text embeddings:
- Model: 'Lajavaness/bilingual-embedding-large'
- Supports both French and English content
- Generates dense vector representations for each article

## Usage

### Basic Analysis Pipeline
```python
# Load and preprocess data
files = pd.read_csv("syndrome_aerotoxique - Sheet1.csv", header=1)
files['language'] = files['content'].apply(detect_language)
files['processed_content'] = files['content'].apply(preprocess_text)

# Initialize analyzer and run analysis
analyzer = PressCorpusAnalyzer(files)
report = analyzer.generate_report()
```

### Generate Embeddings
```python
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('Lajavaness/bilingual-embedding-large', trust_remote_code=True)

# Generate embeddings
sentences = files['processed_content'].values
embeddings = model.encode(sentences)

# Save embeddings to DataFrame
files['embeddings'] = embeddings.tolist()
files.to_parquet('/kaggle/working/files_with_embeddings.parquet')
```

### Visualization
The project includes a custom visualization function for sentiment distribution:
```python
sentiment_distrib = report["sentiment"]["sentiment_distribution"]
sentiment_distrib_graph(sentiment_distrib)
```

## Output Files
- `files_with_embeddings.parquet`: Contains processed articles with their corresponding embeddings
- Generated visualizations for sentiment analysis and topic modeling

## Data Structure
The analysis expects input data with the following columns:
- `content`: Raw text content of articles
- `processed_content`: Preprocessed text (automatically generated)
- `language`: Detected language (automatically generated)
- `embeddings`: Vector embeddings (added during embedding generation)

## Technical Notes
- The embedding model generates 1024-dimensional vectors
- Topic modeling is configured for 5-7 topics by default
- Sentiment analysis provides scores for both polarity and subjectivity
