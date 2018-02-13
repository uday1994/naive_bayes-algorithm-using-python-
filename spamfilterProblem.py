# -*- coding: utf-8 -*-
"""
Created on Fri Jan 05 16:59:58 2018

@author:
"""
# Conditional Probability Examples

'''
Purchasing mac book when you already purchased the iPhone.
Having a refreshing drink when you are in the movie theater.
Buying peanuts when you brought a chilled soft drink.
'''



'''
1. Preparing the text data.
2. Creating word dictionary.
3. Feature extraction process
4. Training the classifier

'''

# 1. Preparing the text data.
'''
train_mail folder:
training set =  702 --> equally divided between spam 
and not spam
test_mail folder:
test set = 260 --> equally divided between 
spam and not spam
'''


'''
--> In any text mining problem, text cleaning is the first step 
where we remove those words from the document 
    which may not contribute to the information we want to extract. 
    
--> Emails may contain a lot of undesirable characters like 
punctuation marks, stop words, digits,
    etc which may not be helpful in detecting the spam email.
    
--> The emails in Ling-spam corpus have been already preprocessed
 in the following ways:
    
    a) Removal of stop words – Stop words like “and”, “the”, “of”, etc are very common in all English sentences
       and are not very meaningful in deciding spam or legitimate status, 
       so these words have been removed from the emails.
    b) Lemmatization – It is the process of grouping together the different inflected forms of a word so 
       they can be analysed as a single item. For example, “include”, “includes,” and “included” would 
       all be represented as “include”. The context of the sentence is also preserved in lemmatization
       as opposed to stemming (another buzz word in text mining which does not consider meaning of the sentence).
    
--> Several ways of doing it. Let s look at dictionary way 

'''

# 2. Creating word dictionary.

'''
Subject: posting

hi , ' m work phonetics project modern irish ' m hard source . anyone recommend book article english ? ' , 
specifically interest palatal ( slender ) consonant , work helpful too . 
thank ! laurel sutton ( sutton @ garnet . berkeley . edu

'''

import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    print type(dictionary)
    
    list_to_remove = dictionary.keys()
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary

def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix

# Create a dictionary of words with its frequency

train_dir = 'C:\Users\User\Desktop\data science class\case_study\bayesian\ling-spam\train-mails'
dictionary = make_Dictionary(train_dir)
print dictionary

print len(dictionary)
# Here We have chosen 3000 most frequently used words in the dictionary.

# -----------------------------------------------------------------------------------------------------------------

# 3. Feature extraction process.

'''
#Once the dictionary is ready, we can extract word count vector 
#(our feature here) of 3000 dimensions
#for each email of training set.

#Each word count vector contains the frequency 
of 3000 words in the training file. 
'''
#Prepare feature vectors per training mail and its labels

# Prepare feature vectors per training mail and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)

# 4. Training the classifiers.

model1 =  MultinomialNB()
model2 = GaussianNB()

model1.fit(train_matrix,train_labels)
model2.fit(train_matrix,train_labels)

# Test the unseen mails for Spam

test_dir = 'D:\\project\\ds_practice\\ds_practice\\case_study\\bayesian\\ling-spam\\test-mails'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1


result1 = model1.predict(test_matrix)
print (result1)
result2 = model2.predict(test_matrix)
print(result2)

print (confusion_matrix(test_labels,result1))
print (confusion_matrix(test_labels,result2))










