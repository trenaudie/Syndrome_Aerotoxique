"""
In this analysis, perform an embedding clustering over the corpus"""

# %%%

# data
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
# %%

