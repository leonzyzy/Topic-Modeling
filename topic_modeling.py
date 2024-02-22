import pandas as pd
import re
import string
import nltk
import hdbscan
from umap import UMAP
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Stopword removal, converting uppercase into lower case, and lemmatization
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# import data
df = pd.read_csv('Reviews.csv', encoding = "ISO-8859-1")

# load model
sbert = SentenceTransformer('all-MiniLM-L6-v2')

# sample some rows
text = df['Text'][:100]
embedding = sbert.encode(text, convert_to_numpy=True, show_progress_bar=True)

# perform umap
map = UMAP(n_components=8, init='random', random_state=528)
embedding_proj = map.fit_transform(embedding)

# use dbscan
cluster = hdbscan.HDBSCAN(min_cluster_size=5).fit(embedding_proj)
cluster_labels = cluster.fit_predict(embedding_proj)

text_group = pd.DataFrame(text)
text_group['label'] = cluster_labels

# for each label perform tfidf
for i in range(0, len(text_group['label'].unique())):
    label = text_group[text_group['label'] == i]
    label_text = label['Text']
    
    # Preprocess text data
    preprocessed_texts = []
    for t in label_text:
        t = re.sub(r'[^\w\s]','',t)
        tokens = word_tokenize(t)
        clean_tokens = [token.lower() for token in tokens if token not in string.punctuation]
        filtered_tokens = [token for token in clean_tokens if token not in stop_words]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        preprocessed_texts.append(' '.join(lemmatized_tokens))

    
    vectorizer = TfidfVectorizer() 
    vectors = vectorizer.fit_transform(preprocessed_texts)
    tf_idf = pd.DataFrame(vectors.todense()) 
    tf_idf.columns = vectorizer.get_feature_names_out()
    tfidf_matrix = tf_idf.T
    tfidf_matrix.columns = ['response'+ str(i) for i in range(1, len(tfidf_matrix.columns)+1)]
    tfidf_matrix['count'] = tfidf_matrix.sum(axis=1)
    tfidf_matrix = tfidf_matrix.sort_values(by ='count', ascending=False)[:10] 
    print(tfidf_matrix['count'].head(10))

