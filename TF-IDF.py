# -*- coding: utf-8 -*-
# https://blog.csdn.net/mbx8x9u/article/details/78851815
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["I come to China to travel", "This is a car polupar in China", "I love tea and Apple ", "The work is to write some papers in science"] 

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpus)
print(tfidf.shape)

words = vectorizer.get_feature_names()
vectorizer.get_params()
vectorizer.get_stop_words()

for i in range(len(corpus)):
    print('----Document %d----' % (i))
    for j in range(len(words)):
        if tfidf[i,j] > 1e-5:
              print(words[j], tfidf[i,j])
