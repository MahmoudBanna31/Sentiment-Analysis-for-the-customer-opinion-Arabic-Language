import numpy as np
import pandas as pd
from pandas import DataFrame 
import itertools
import csv
import re 
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer 
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report 
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier


#Read DataSete

files=('RES.csv')
reviews=pd.read_csv(files)


reviews
def normalizeArabic(text):
    text = re.sub(r"[إأٱآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    noise = re.compile("""  ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)
    return text
    
print(normalizeArabic("علي على"))

def stopWordRmove(text):
  #  ar_stop_list = open ("ar_stop_word_list.txt", "r")
   # stop_words = ar_stop_list.read().split('\n')
    needed_words = []
    words = word_tokenize(text)
    for w in words:
        if w not in (stopWords):
             needed_words.append(w)    
    filtered_sentence = " ".join(needed_words)
    return filtered_sentence
    
    
print(stopWordRmove(" قال الرجل ان هذا ما لن"))


import nltk
nltk.download('popular')

def stimming(text):
    st = ISRIStemmer()
    stemmed_words = []
    words = word_tokenize(text)
    for w in words:
        stemmed_words.append(st.stem(w))
    stemmed_sentence = " ".join(stemmed_words)
    return stemmed_sentence
    
print (stimming(" بسم الله الرحمن الرحيم "))



def prepareDataSets(reviews):
    sentence = []
    global text 
    for index, r in reviews.iterrows():
        text = stopWordRmove(r['text'])
        text = normalizeArabic(r['text'])
        text = stemming(r['text'])
        if r['polarity'] == -1:
            sentence.append([text, 'neg'])
        else:
            sentence.append([text, 'pos'])
                               
    df_sentences = DataFrame(sentence, columns=['text', 'polarity'])
    return df_sentences 
 
 
 prepareDataSets(reviews)
 
 
 
from sklearn.externals import joblib
def featureExtraction(data):
    vectorizer = TfidfVectorizer(min_df=10, max_df=.75, ngram_range=(1,3))
    tfidf_data = vectorizer.fit_transform(data)
    # save your model in disk
    joblib.dump(vectorizer, 'tfidf.pkl') 

    # load your model
    tfidf = joblib.load('tfidf.pkl') 
    return tfidf_data


#method Classification

def learning(Knn, X , Y ):
        X_train, X_test, Y_train, Y_test = \
        train_test_split(X,Y, test_size=0.9, random_state=43)
        knn = Knn()
        knn.fit(X_train, Y_train)
        
        scores = cross_val_score(knn, X_test, Y_test, cv=10, scoring='accuracy') 
        predict = cross_val_predict(knn, X_test, Y_test, cv=10)
        print(scores)
        print ("Accuracy of %s: %0.2f (+/- %0.2f) " % (knn, scores.mean(), scores.std() *2))
        print (classification_report(Y_test, predict))
        
        
        

def main(clf):
    preprocessed_reviews = prepareDataSets(reviews)
    data, target = preprocessed_reviews['text'], preprocessed_reviews['polarity']
    tfidf_data = featureExtraction(data)
    learning(clf, tfidf_data, target)
    
 
 
 main(LinearSVC)
 
 
 #Apply in different types Classification
 
 clfs = [ MultinomialNB, BernoulliNB, LogisticRegression, SGDClassifier, SVC, LinearSVC , KNeighborsClassifier]
 
 
 for clfs in clfs:
    main(clfs)
