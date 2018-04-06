import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# reading the dataset
df=pd.read_csv("train.csv")

# cleaning the tweets
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
corpusw=[]
for i in range(0,31962):
    new_tweets=re.sub('[^a-zA-z]',' ',df.tweet[i])
    new_tweets=new_tweets.lower()
    new_tweets=new_tweets.split()
    ps=PorterStemmer()
    new_tweets=[ps.stem(word) for word in new_tweets if word not in set(stopwords.words('english'))]
    new_tweets=' '.join(new_tweets)
    corpusw.append(new_tweets)


cv=CountVectorizer(max_features=3000)
X=cv.fit_transform(corpusw).toarray()
y=df.label.values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(----X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import classification_report 
print (classification_report(y_test,y_pred))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Fitting Naive Bayes to the Test test  set
"""from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)


df2=pd.read_csv("test.csv")
corpusw1=[]
for i in range(0,17197):
    new_tweets1=re.sub('[^a-zA-z]',' ',df2.tweet[i])
    new_tweets1=new_tweets.lower()
    new_tweets1=new_tweets.split()
    ps=PorterStemmer()
    new_tweets1=[ps.stem(word) for word in new_tweets if word not in set(stopwords.words('english'))]
    new_tweets1=' '.join(new_tweets)
    corpusw1.append(new_tweets)

cv=CountVectorizer(max_features=3000)
X_test=cv.fit_transform(corpusw1).toarray()
"""









