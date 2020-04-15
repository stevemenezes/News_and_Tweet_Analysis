import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import re
from gensim import corpora
import pickle
import gensim

#reading news excel file from local machine
news=pd.read_excel('C:\\Users\\steve\\News.xlsx')
#reading only the article body
main_article=news['Body'].values.tolist()

#filter out empty articles
final_articles=[]
for m in main_article:
    try:
        import math
        boolean=math.isnan(float(main_article[25]))
    except:
        boolean=False
    if boolean==False:
        final_articles.append(m)
        
#removing stopwords        
final_stopwords_list = list(fr_stop) + list(en_stop)
vect = TfidfVectorizer(stop_words=final_stopwords_list)
tfidf_matrix = vect.fit_transform(dataframe['Body'].values.astype('U'))

#TFIDF vectors form each document
df = pd.DataFrame(tfidf_matrix.toarray(), columns = vect.get_feature_names())
cols=df.columns
arr=df.columns
#removing non-english words from dataframe
filtered_words = [word for word in arr if word not in stopwords.words('english')]
final_fitler=[]
final_fitler=[word for word in filtered_words if word not in stopwords.words('french')]
ultra_final=[word for word in final_fitler if not word.isdigit()]
new_frame = df.loc[:, ultra_final]
data=cosine_similarity(new_frame)
df=pd.DataFrame(data=data[0:,0:],index=[i for i in range(data.shape[0])],columns=[''+str(i) for i in range(data.shape[1])])
#perform PCA
pca = PCA(n_components=10).fit(data)
data2D = pca.transform(data)
#kmeans clustering on n=4 clusters
km = KMeans(n_clusters=4)
km.fit(data2D)
plt.scatter(data2D[:, 0], data2D[:, 1],c=km.labels_)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

news_data=news['Body']
subset_news=[]
#remove empty data
for i in range(3525):
    try:
        import math
        val= math.isnan(float(news_data[i]))
    except:
        val= False
    if val==False:
        subset_news.append(news_data[i])

#process articles
words = set(nltk.corpus.words.words())
filtered_data=[]
for s in subset_news:
    filtered_words=s.split(' ')
    final=[]
    final=[fw.lower() for fw in filtered_words if not fw.isdigit()]
    filtered_data.append(final) 
    
#tokenize and remove stopwords    
token=[]
for text in filtered_data:
    words=list(set(text))
    arr1=[]
    for word in words:
        if word in cols and word not in stopwords.words('english'):
            arr1.append(word)
    token.append(arr1)      

#corpus
dictionary = corpora.Dictionary(token) 
corpus = [dictionary.doc2bow(text) for text in token]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

#lda and topic generation on news articles
NUM_TOPICS = 4
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=10)
ldamodel.save('model4.gensim')

topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)