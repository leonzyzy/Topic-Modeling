import pandas as pd
import numpy as np
import re
import string
import nltk
import hdbscan
import seaborn as sns
import umap.umap_ as umap
import matplotlib.pyplot as plt
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
df = pd.read_csv('df_file.csv')

# random sample of 1000 reviews
df = np.array(df['Text']).tolist()

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
    kn = KneeLocator(range(2, 10 + 1), 
                     optimal_dist, 
                     curve='concave', 
                     direction='increasing')

    return kn.knee

# define a function to assign labels to clusters
def assign_labels(x, model, max_k):
    # embedding
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(x, convert_to_numpy=True, show_progress_bar=True)
    
    # perform UMAP
    map = umap.UMAP(n_components=8, n_neighbors=12, random_state=42)
    embeddings_proj = map.fit_transform(embeddings)
    
    # find the optimal number of clusters
    k = optimal_k(embeddings_proj, max_k)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings_proj)
    y = kmeans.fit_predict(embeddings_proj)
    text_group = pd.DataFrame({'text': x, 'label': y})
    
    return text_group

text_group = assign_labels(df, model, 10)


# Function to find duplicate words between two dictionaries
def find_duplicates(dict1, dict2):
    # Extracting the keys (words) from the dictionaries
    words_dict1 = set(dict1.keys())
    words_dict2 = set(dict2.keys())
    
    # Finding the common words (intersection)
    common_words = words_dict1.intersection(words_dict2)
    
    return list(common_words)


# create a dic to count frequency of words in each cluster
words_dic = []

for i in range(0, len(text_group['label'].unique())):
    label = text_group[text_group['label'] == i]
    label_text = label['text']
    topic = {}
    
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
    tfidf_matrix = tfidf_matrix.sort_values(by ='count', ascending=False)[:20] 

    words_dic.append(tfidf_matrix['count'].to_dict())


duplicate_words = []
for i in range(len(words_dic)-1):
    for j in range(i+1, len(words_dic)):
        common_words = find_duplicates(words_dic[i], words_dic[j])
        duplicate_words.append(common_words)

# Removing duplicates from the list
duplicate_words = list(set([item for sublist in duplicate_words for item in sublist]))

# remove keys if they are in duplicate_words
for i in range(0, len(words_dic)):
    for word in duplicate_words:
        if word in words_dic[i]:
            del words_dic[i][word]    

    # only keep top 10 keys
    words_dic[i] = dict(sorted(words_dic[i].items(), key=lambda item: item[1], reverse=True)[:10])


# plot a bar chart for each topic within word_dic, plot it horizontally
for i in range(0, len(words_dic)):
    plt.barh(list(words_dic[i].keys()), list(words_dic[i].values()))
    plt.title('Topic ' + str(i))
    plt.show()
