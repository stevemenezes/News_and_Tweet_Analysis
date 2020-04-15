import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from gensim import corpora
import pickle
import gensim

#read tweets extracted locally
with open('C:\\Users\\steve\\tweets.txt',encoding='utf8') as f:
    data=f.read()
#splitting data    
split_data=data.split('\n\n')
words = set(nltk.corpus.words.words())

tweets=[]

#tweet cleansing
for s in split_data:
    s=s.strip()
    s=s.split(' ')
    for idx,element in enumerate(s):
        #remove text after retweets since those are not needed
        if element=='RT':
            break
    s=' '.join(s[:idx])
    #remove urls from text
    s=re.sub(r'http\S+','', s)
    s=re.sub('[^A-Za-z0-9]+', ' ', s)
    filtered_words=s.split(' ')
    final=[]
    final=[fw.lower() for fw in filtered_words if not fw.isdigit()]
    tweets.append(final)

final_tweets=[]
#remove stopwords from tweets
for tweet in tweets:
    t1=[]
    t1 = [word for word in tweet if word not in stopwords.words('english')]
    t2=[]
    t2= [word for word in t1 if len(word)>1]
    final_tweets.append(' '.join(t2))
    
#performing TFIDF for each tweet data    
vect = TfidfVectorizer()
result = vect.fit_transform(final_tweets)
feature_names = vect.get_feature_names()
tweet_index = [i for i in range(0,len(final_tweets))]
df = pd.DataFrame(result.todense(), index=tweet_index, columns=feature_names)
data=cosine_similarity(df)
df=pd.DataFrame(data=data[0:,0:],index=[i for i in range(data.shape[0])],columns=[''+str(i) for i in range(data.shape[1])])
#perform PCA to reduce dimensionality
pca = PCA(n_components=2).fit(data)
data2D = pca.transform(data)
#predined number of clusters were 4
km = KMeans(n_clusters=4)
#kmeans clustering done
km.fit(data2D)
plt.scatter(data2D[:, 0], data2D[:, 1],c=km.labels_)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

#tokenize the text
token=[]
for text in final_tweets:
    words=text.split(' ')
    words=list(set(words))
    arr=[]
    for word in words:
        #adding those words which are features
        if word in feature_names:
            arr.append(word)
    token.append(arr)     

#making a corpus of the tweets
dictionary = corpora.Dictionary(token) 
corpus = [dictionary.doc2bow(text) for text in token]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

#performing LDA
NUM_TOPICS = 4
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
#find 8 relevant words in each category to define topic name
topics = ldamodel.print_topics(num_words=8)
for topic in topics:
    print(topic)
    