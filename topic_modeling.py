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
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns


# define a function to optimize the number of clusters
def optimal_k(inputs, max_k, s):
    # define the range of clusters
    intra_cluster_distances = []
    inter_cluster_distances = []
    optimal_dist = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=s)
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
def assign_labels(x, model, n, max_k, s):
    # embedding
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(x, convert_to_numpy=True, show_progress_bar=True)
    
    # perform UMAP
    map = umap.UMAP(n_components=n, n_neighbors=12, random_state=s)
    embeddings_proj = map.fit_transform(embeddings)
    
    # find the optimal number of clusters
    k = optimal_k(embeddings_proj, max_k, s)
    kmeans = KMeans(n_clusters=k, random_state=s).fit(embeddings_proj)
    y = kmeans.fit_predict(embeddings_proj)
    text_group = pd.DataFrame({'text': x, 'label': y})
    
    return text_group

# Function to find duplicate words between two dictionaries
def find_duplicates(dict1, dict2):
    # Extracting the keys (words) from the dictionaries
    words_dict1 = set(dict1.keys())
    words_dict2 = set(dict2.keys())
    
    # Finding the common words (intersection)
    common_words = words_dict1.intersection(words_dict2)
    
    return list(common_words)

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

# Preprocess text data
preprocessed_texts = []
for t in df:
    t = re.sub(r'[^\w\s]','',t)
    tokens = word_tokenize(t)
    clean_tokens = [token.lower() for token in tokens if token not in string.punctuation]
    filtered_tokens = [token for token in clean_tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    preprocessed_texts.append(' '.join(lemmatized_tokens))
    
text_group = assign_labels(preprocessed_texts, model, 32, 10, 42)


# replace text column with preprocessed_texts
text_group['text'] = preprocessed_texts

# create a dic to count frequency of words in each cluster
words_dic = []

for i in range(0, len(text_group['label'].unique())):
    label = text_group[text_group['label'] == i]
    text = label['text']
    topic = {}
    vectorizer = TfidfVectorizer() 
    vectors = vectorizer.fit_transform(text)
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


# use word2vec 
data = []
# iterate through each sentence in the file
for i in preprocessed_texts:
    temp = []
 
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
 
    data.append(temp)

word2vec = Word2Vec(data)
word2vec.wv.similarity('budget', 'mobile')
 
 
topic1 = list(words_dic[0].keys())
topic2 = list(words_dic[1].keys())
topic3 = list(words_dic[2].keys())
topic4 = list(words_dic[3].keys())

# define a function to find the most similar words
def word_importance(x, word_encoder):
    similarity_matrix = pd.DataFrame()
    for i in x:
        similarity = []
        for j in x:
            similarity.append(word_encoder.wv.similarity(i, j))
        similarity_matrix[i] = similarity
    return similarity_matrix.mean(axis=0).sort_values(ascending=False)[:7]

topic1_words_important = word_importance(topic1, word2vec).to_dict()
topic2_words_important = word_importance(topic2, word2vec).to_dict()
topic3_words_important = word_importance(topic3, word2vec).to_dict()
topic4_words_important = word_importance(topic4, word2vec).to_dict()

# plot each 
plt.figure(figsize=(10, 5))
sns.barplot(x=list(topic1_words_important.values()), y=list(topic1_words_important.keys()), palette='viridis')
plt.title('Top 7 words for topic 1')
plt.show()
