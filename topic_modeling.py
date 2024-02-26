import pandas as pd
import numpy as np
import re
import string
import nltk
import hdbscan
import umap.umap_ as umap
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

# Stopword removal, converting uppercase into lower case, and lemmatization
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# define model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# import data
df = pd.read_csv('tripadvisor_hotel_reviews.csv')

# random sample of 1000 reviews
df = np.array(df.sample(50)['Review']).tolist()

# define a function to optimize the number of clusters
def optimal_k(inputs, max_k):
    # define the range of clusters
    intra_cluster_distances = []
    inter_cluster_distances = []
    optimal_dist = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(inputs)
        
        # Inertia is the sum of squared distances of samples to their closest cluster center
        intra_cluster_distances.append(kmeans.inertia_)
        
        # Calculate inter-cluster distance using silhouette score
        silhouette_avg = silhouette_score(inputs, kmeans.labels_)
        inter_cluster_distances.append(silhouette_avg)
        
        # we want to max inter_cluster_distances and min intra_cluster_distances
        # equals to max inter_cluster_distances - min intra_cluster_distances
        # we want to find the point where the difference is maximized
        optimal_dist.append(inter_cluster_distances[-1] - intra_cluster_distances[-1])
    
    # use kneedle algorithm to find the optimal number of clusters
    kn = KneeLocator(range(2, max_k + 1), 
                     optimal_dist, 
                     curve='concave', 
                     direction='increasing')

    return kn.knee

# define a function to assign labels to clusters
def assign_labels(x, model, max_k):
    # embedding
    embeddings = model.encode(x, convert_to_numpy=True, show_progress_bar=True)
    
    # perform UMAP
    map = umap.UMAP(n_components=5, n_neighbors=12, random_state=528)
    embeddings_proj = map.fit_transform(embeddings)
    
    # find the optimal number of clusters
    k = optimal_k(embeddings_proj, max_k)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings_proj)
    y = kmeans.fit_predict(embeddings_proj)
    text_group = pd.DataFrame({'text': x, 'label': y})
    
    return text_group

text_group = assign_labels(df, model, 10)
    
    
# find topic for each cluster using tfidf
# for each label perform tfidf
for i in range(0, len(text_group['label'].unique())):
    label = text_group[text_group['label'] == i]
    label_text = label['text']
    
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
    print(tfidf_matrix['count'].head(5))
    
