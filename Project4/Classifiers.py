
# coding: utf-8

# In[225]:

# using nltk
import sys
import argparse
import pandas as pd
import nltk
import glob
import numpy as np
from nltk.corpus import stopwords

from sklearn import naive_bayes

from sklearn.feature_extraction.text import TfidfVectorizer

# in charge of spliting dataset into train and test
from sklearn.model_selection import train_test_split

from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

#importing dataset which divides into data and target sets
from sklearn.datasets import load_files  

#regex for preprocessing
import re

#lemmatization
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize


#Stemming
from nltk.stem.lancaster import LancasterStemmer


#ploting graph
# import seaborn as sns

#serialization
# A common pattern in Python 2.x is to have one version of a module implemented 
# in pure Python, with an optional accelerated version implemented as a C extension; 
#for example, pickle
import pickle


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
typeClassifier = sys.argv[1]
version = sys.argv[2]

# In[280]:

dataReviews = pd.read_csv('allreviews.txt', 
                          delimiter="\t", 
                          names=['dataSentences', 
                                 'dataLabels'])

review_x, review_y =dataReviews['dataSentences'], dataReviews['dataLabels']

# for gt in review_x:
#     print (gt)

#shows a representation of data sentences and corresponding labels
dataReviews.head()

# sns.countplot(x='Sentiment',y='count', data=dataReviews)

# print(type(review_y))


# In[329]:

#preprocessing
def preprocessing(reviews):
   
    preprocessedDoc = []
    lancaster_stemmer = LancasterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()

    for sentence in range(0, len(reviews)):  
    #     print(review_x[sentence])
        # Remove all the special characters
        a = re.sub(r'\W', ' ', str(reviews[sentence]))

        # remove all single characters
        a = re.sub(r'\s+[a-zA-Z]\s+', ' ', a)

        # Remove single characters from the start
        a = re.sub(r'\^[a-zA-Z]\s+', ' ', a) 

        # Substituting multiple spaces with single space
        a = re.sub(r'\s+', ' ', a, flags=re.I)
        
        # Converting to lowercase
#         a = a.str.lower()
                     
        preprocessedDoc.append(a)
        
    preprocessedDoc = [[wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(s)]
              for s in preprocessedDoc]

    return preprocessedDoc


# In[326]:

def prepre(review):
    lancaster_stemmer = LancasterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    
    a = re.sub(r'\W', ' ', str(review))

    # remove all single characters
    a = re.sub(r'\s+[a-zA-Z]\s+', ' ', a)

    # Remove single characters from the start
    a = re.sub(r'\^[a-zA-Z]\s+', ' ', a) 

    # Substituting multiple spaces with single space
    a = re.sub(r'\s+', ' ', a, flags=re.I)
    
    r = [wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(review)]
    
    return r

# prepre("review_x was really bad and boring")


# In[328]:

#Normalised : split dataset into training data and testing data

# # Converting preprocessedDoc type:list to numpy array
# numpy_preprocessedDoc = np.asarray(preprocessing(review_x))

# Converting preprocessedDoc type:list to pandas.core.series.Series
pandas_preprocessedDoc =  pd.Series(preprocessing(review_x)).astype(str).str.zfill(11)

# print(type(pandas_preprocessedDoc))

x_NormTrain, x_NormTest, y_NormTrain, y_NormTest = train_test_split(
                                                    pandas_preprocessedDoc, 
                                                     review_y, 
                                                    test_size=0.05,
                                                    train_size = 0.95,
                                                    random_state = 42) 

#UnNormalised : split dataset into training data and testing data
x_UnNormTrain, x_UnNormTest, y_UnNormTrain, y_UnNormTest = train_test_split(
                                                    review_x, 
                                                     review_y, 
                                                    test_size=0.05,
                                                    train_size = 0.95,
                                                    random_state = 42)      
# print (X_UnNormTrain)
# print(a.reshape(a.shape[1:]))


# In[294]:

# In this funtion, we train for logistic regression classifier
#We use the bag of words approach for convertting the text to numbers.
#To resolve the issue of the bag of words approach not taking into account 
#that the word might also be having a high frequency of occurrence in other 
#documents as well.

def logisticRegTrainNorm():
    
    #Normalisation of text 
    stopSet = set(stopwords.words('english'))
    
    #Term-frequency times document frequency Vectorizer was used because 
    #it does not only provide the number of occurrences but also tells
    #about the importance of the word.
    tfid_vectorizer_n_lr = TfidfVectorizer(
                                      use_idf=True, # IDF is "t" when use_idf is given
                                      smooth_idf = True, #adds "1" to the numerator and denominator
#                                       lowercase=True, 
                                      strip_accents=ascii, 
                                      stop_words=stopSet
                                      )
    
    tfid_vectorizer_n_lr.fit(x_NormTrain)
    norm_x_lg = tfid_vectorizer_n_lr.transform(x_NormTrain).toarray()  
    print(norm_x_lg.shape)
    print(y_NormTrain.shape)



    # Training the normalized Logistic Regression Classifier
    logistic_norm_classifier = LogisticRegression().fit(norm_x_lg,y_NormTrain)
    
#     testing
    x_test_features_lr_n = tfid_vectorizer_n_lr.transform(x_NormTest).toarray()
    x_predict_lr_n = logistic_norm_classifier.predict(x_test_features_lr_n)
    
    print("--------LOGISTIC REGRESSION NORMALIZED----------")
    
    print("ACCURACY SCORE: ", accuracy_score(y_NormTest, x_predict_lr_n)*100) 
    
    print()

    print("CONTINGENCY TABLE : \n" ,classification_report(y_NormTest, x_predict_lr_n))
        
    return logistic_norm_classifier, tfid_vectorizer_n_lr 

# logistic_norm_classifier, tfid_vectorizer_n_lr = logisticRegTrainNorm()


# In[289]:

def logisticRegTrainUnNorm():
    
    #Term-frequency times document frequency Vectorizer was used because 
    #it does not only provide the number of occurrences but also tells
    #about the importance of the word.
    
    #to get unnormalized version of logistic regression set parameters to
    #False and None
    tfid_vectorizer_un_lr = TfidfVectorizer(
                                      use_idf=True,
                                      smooth_idf = True
                                      )

    unNorm_x_lg = tfid_vectorizer_un_lr.fit_transform(x_UnNormTrain).toarray()    

    # Training the normalized Logistic Regression Classifierr
    logistic_un_classifier = LogisticRegression().fit(unNorm_x_lg, y_UnNormTrain)
    
    
    # testing
    x_test_features_lr_un = tfid_vectorizer_un_lr.transform(x_UnNormTest)
    x_predict_lr_un = logistic_un_classifier.predict(x_test_features_lr_un)
    
    print("--------LOGISTIC REGRESSION UNNORMALIZED----------")

    print("ACCURACY SCORE: ", accuracy_score(y_UnNormTest, x_predict_lr_un)*100) 

    print()
    
    print("CONTINGENCY TABLE: \n" ,classification_report(y_UnNormTest, x_predict_lr_un))
    
    return logistic_un_classifier, tfid_vectorizer_un_lr 

# logistic_un_classifier, tfid_vectorizer_un_lr = logisticRegTrainUnNorm()


# In[290]:

def trainUnNormNB():
    # Using TFIDF, short for term frequency–inverse document frequency
    # This transforms text to feature vectors

    #to get unnormalized version of logistic regression set parameters to
    #False and None
    tfid_vectorizer_un_nb = TfidfVectorizer(
                                        use_idf=True, 
                                        lowercase=False, 
                                        strip_accents=None, 
                                        stop_words= None
                                       )

    unNorm_x_nb = tfid_vectorizer_un_nb.fit_transform(x_UnNormTrain)

    # Training the unnormalized Naive Bayes Classifier
    unNorm_nbClassifier = naive_bayes.MultinomialNB().fit(unNorm_x_nb, y_UnNormTrain)
    
       # testing
    x_test_features_nb_un = tfid_vectorizer_un_nb.transform(x_UnNormTest)
    x_predict_nb_un = unNorm_nbClassifier.predict(x_test_features_nb_un)
    
    print("--------NAIVE BAYES UNNORMALIZED CLASSIFIER----------")

    print("ACCURACY SCORE: ", accuracy_score(y_UnNormTest, x_predict_nb_un)*100)
    
    print()
    
    print("CONTINGENCY TABLE: \n" ,classification_report(y_UnNormTest, x_predict_nb_un))
    
    return unNorm_nbClassifier, tfid_vectorizer_un_nb

# unNorm_nbClassifier,tfid_vectorizer_un_nb = trainUnNormNB()


# In[293]:

def trainNormNB():
    
    #Normalisation of text 
    stopSet = set(stopwords.words('english'))
    
    #Term-frequency times document frequency Vectorizer was used because 
    #it does not only provide the number of occurrences but also tells
    #about the importance of the word.
    tfid_vectorizer_n_nb = TfidfVectorizer(
                                      use_idf=True, # IDF is "t" when use_idf is given
                                      smooth_idf = True, #adds "1" to the numerator and denominator
                                      lowercase=True, 
                                      strip_accents='ascii', 
                                      stop_words=stopSet
                                      )

    norm_x_nb = tfid_vectorizer_n_nb.fit_transform(x_NormTrain).toarray()    


    # Training the normalized Logistic Regression Classifier
    norm_nbClassifier = naive_bayes.MultinomialNB().fit(norm_x_nb, y_NormTrain)
    
    # testing classifier
    x_test_features_nb_n = tfid_vectorizer_n_nb.transform(x_NormTest)
    x_predict_nb_n = norm_nbClassifier.predict(x_test_features_nb_n)
    
    print("--------NAIVE BAYES NORMALIZED CLASSIFIER----------")

    print("ACCURACY SCORE: ", accuracy_score(y_NormTest, x_predict_nb_n)*100) 
    
    print()
    
    print("CONTINGENCY TABLE: \n" ,classification_report(y_NormTest, x_predict_nb_n))
    
    return norm_nbClassifier, tfid_vectorizer_n_nb   

# norm_nbClassifier, tfid_vectorizer_n_nb = trainNormNB()


# In[331]:

#Serialization of both classifier and vectorizer is done in here.
#In text classification, it is not enough to just store the classfier
#therefore the vectorizer is also needed for future usage

# # --------LOGISTIC REGRESSION SERIALIZATION----------

# with open('logistic_norm_classifier_vectorizer.sav', 'wb') as pickle_lg_n:  
#     pickle.dump((logistic_norm_classifier,tfid_vectorizer_n_lr),pickle_lg_n)
    
# with open('logistic_unNorm_classifier_vectorizer.sav', 'wb') as pickle_lg_un:  
#     pickle.dump((logistic_un_classifier,tfid_vectorizer_un_lr),pickle_lg_un)
    
# # -----------NAIVE BAYES SERIALIZATION----------
# with open('nb_unNorm_classifier_vectorizer.sav', 'wb') as pickle_nb_un:  
#     pickle.dump((unNorm_nbClassifier,tfid_vectorizer_un_nb),pickle_nb_un)
    
# with open('nb_norm_classifier_vectorizer.sav', 'wb') as pickle_nb_n:  
#     pickle.dump((norm_nbClassifier,tfid_vectorizer_n_nb),pickle_nb_n)  
    
  

# with open('logistic_norm_classifier_vectorizer.sav', 'rb') as load_pickle_lg_n:
#       logistic_norm_classifier,tfid_vectorizer_n_lr = pickle.load(load_pickle_lg_n)
    
#     with open('logistic_unNorm_classifier_vectorizer.sav', 'rb') as load_pickle_lg_un:
#       logistic_un_classifier,tfid_vectorizer_un_lr = pickle.load(load_pickle_lg_un)
    
#     with open('nb_unNorm_classifier_vectorizer.sav', 'rb') as load_pickle_nb_un:
#       unNorm_nbClassifier,tfid_vectorizer_un_nb = pickle.load(load_pickle_nb_un)
    
#     with open('nb_norm_classifier_vectorizer.sav', 'rb') as load_pickle_nb_un:
#       norm_nbClassifier,tfid_vectorizer_un_nb = pickle.load(load_pickle_nb_un)
   
    


# In[330]:

def testingUnNormLog(testdoc, version, typeClassifier):
    predict= []

    with open (testdoc, "r") as openedTestdoc:
        if typeClassifier == "lr":
            if version == "un":
                logistic_un_classifier, tfid_vectorizer_un_lr = logisticRegTrainUnNorm()
                for sentence in openedTestdoc:
                    predict.append(sentence.strip('\r\n'))
                x_test_features_lr_un = tfid_vectorizer_un_lr.transform(predict)
                x_predict_lr_un = logistic_un_classifier.predict(x_test_features_lr_un)
                
#               write results to document
                with open("result-lr-un.txt", 'w', newline = '') as r1:
                    for prediction_lr_un in x_predict_lr_un:
                        r1.write(str(prediction_lr_un) + '\n')             
            
            elif version == "n":
                logistic_norm_classifier, tfid_vectorizer_n_lr = logisticRegTrainNorm ()
                for sentence in openedTestdoc:  
                    preText= prepre(sentence.lower())
                x_test_features_lr_n = tfid_vectorizer_n_lr.transform(preText)
                x_predict_lr_n = logistic_norm_classifier.predict(x_test_features_lr_n)
                
                with open("result-lr-n.txt", 'w', newline = '') as r2:
                    for prediction_lr_n in x_predict_lr_n:
                        r2.write(str(prediction_lr_n) + '\n')
                
            else:
                print('Invalid version specified')
                
        if typeClassifier == "nb":
            if version == "un":
                unNorm_nbClassifier, tfid_vectorizer_un_nb = trainUnNormNB ()
                for sentence in openedTestdoc:
                    predict.append(sentence.strip('\r\n'))
                x_test_features_nb_un = tfid_vectorizer_un_nb.transform(predict)
                x_predict_nb_un = unNorm_nbClassifier.predict(x_test_features_nb_un)
                
                with open("result-nb-un.txt", 'w', newline = '') as r3:
                    for prediction_nb_un in x_predict_nb_un:
                        r3.write(str(prediction_nb_un) + '\n')
            
            elif version == "n":
                norm_nbClassifier, tfid_vectorizer_n_nb = trainNormNB()
                for sentence in openedTestdoc:
                    pretext= prepre(sentence.lower())
                x_test_features_nb_n = tfid_vectorizer_n_nb.transform(pretext)
                x_predict_nb_n = norm_nbClassifier.predict(x_test_features_nb_n)

                with open("result-nb-n.txt", 'w', newline = '') as r4:
                    for prediction_nb_n in x_predict_nb_n:
                        r4.write(str(prediction_nb_n) + '\n')

            else:
                print('Invalid version specified')

testingUnNormLog("test_sentences.txt", version, typeClassifier)
    


# In[189]:




# In[200]:

# if __name__ == '__main__':
    
#      # accept command-line arguments
#     parser =argparse.ArgumentParser(description="Sentiment analysis for reviews using both a Logistic Regression model and a Multinomial Naive Bayes Classification model.")
#     parser.add_argument("classifier-type", help="specifies the type of classifier to use. 'nb'=naiveBayes, and 'lr'=logisticRegression")
#     parser.add_argument("version", help="specifies which version of classifier to use. 'u'=un-normalized or 'n'=normalized")
#     parser.add_argument("testfile", help="accepts the text file to peform sentiment analysis on.")

    
#     args = vars(parser.parse_args())
     
#     # extract arguments passed from command-line
#     classifier_type = args.get('classifier-type', None)
#     version = args.get('version', None)
#     testfile = args.get('testfile', None)
    
#     testingUnNormLog(testfile,version, classifier_type )


# In[ ]:



