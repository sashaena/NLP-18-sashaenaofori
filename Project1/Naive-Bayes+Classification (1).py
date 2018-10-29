
# coding: utf-8

# In[1]:

import re
import math
import random


# In[2]:

def processFiles(fname):
    
#     creating dictionary to story various classes
    dict_corpus = {0:[],1:[]}
    
#     put various files in a list to continously read files
    for i in fname:
        with open (i, "r") as openedFile:
            for line in openedFile:
                lineSplit = line.strip('\n').split('\t')
                
#                 print(lineSplit)
                if len(lineSplit) > 1 and int(lineSplit[1]) == 0:
                    lineSplitFormat = ''.join(lineSplit[0]).lower()
                    lineSplitFormat = re.sub(r'[,.:!+&$?;""()''/|]', '', lineSplitFormat)
                    dict_corpus[0].append(lineSplitFormat.split())
                else:
                    lineSplitFormat = ''.join(lineSplit[0]).lower()
                    lineSplitFormat = re.sub(r'[,.:!+&$?;""()''/|]', '', lineSplitFormat)
                    dict_corpus[1].append(lineSplitFormat.split())

#     print(dict_corpus[1])
    return dict_corpus

# globally accessed
name= ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]
dict_corpus=processFiles(name)    


# In[3]:

# This function calculates the log_prior of the two classes positive(1) and negative(0)

def calculate_logprior(dict_corpus):
    positive_class = len(dict_corpus[1])
    negative_class = len(dict_corpus[0])
    num_documents = positive_class + negative_class
    
    log_prior = {0:math.log(negative_class/num_documents), 1:math.log(positive_class/num_documents)}
    
    print (positive_class, negative_class, num_documents)
    print(log_prior)
    return positive_class,negative_class,log_prior
    

positive_class,negative_class,log_prior=calculate_logprior(dict_corpus)


# In[4]:

def calculate_loglikelihood(dict_corpus):
    
# creating a dictionary to store the number of occurences her word in each class
    wordCountPositive= {}
    wordCountNegative= {}
    denominator= {}
    vocab = []
    
# counting the word occurrences in the negative review dictionary
    for review in dict_corpus[0]:
        for word in review:
            wordCountNegative[word] = wordCountNegative.get(word, 0) + 1
                
# counting the word occurrences in the positive review dictionary
# for each review in my positive dictionary
    for review in dict_corpus[1]:
        for word in review:
            wordCountPositive[word]= wordCountPositive.get(word, 0) +1
                
#     print(wordCountPositive)
    print(len(wordCountNegative.keys()))

#     print(wordCountNegative)

# the vocab is all the individual words in the dict corpus
# returns a distinct/unique words because we wrapped in the collection "set"
    vocab = set(list(wordCountPositive.keys())+ list(wordCountNegative.keys()))
    print(len(vocab))
  
    countPos = 0
    countNeg = 0
    for word in vocab:
        countPos+=wordCountPositive.get(word, 0) + 1
    denominator[1] = countPos
    
    for word in vocab:
        countNeg+=wordCountNegative.get(word, 0) + 1
    denominator[0] = countNeg
    
#     print(denominator)
    
    return wordCountPositive, wordCountNegative, denominator, vocab

wordCountPositive, wordCountNegative, denominator, vocab = calculate_loglikelihood(dict_corpus)


# In[5]:

# This function predicts the class of a sentence

def predictsentence(test_sentence):
    sum= {0: 0 , 1:0 }
    
    for word in test_sentence.split():
        loglikehood_positive = math.log((wordCountPositive.get(word, 0)+1)/denominator[1])
        loglikehood_negative = math.log((wordCountNegative.get(word, 0)+1)/denominator[0])
        sum[1]+=loglikehood_positive
        sum[0]+= loglikehood_negative
        
# added the value of the log prior to the log likelihood    
    sum[0] = sum[0] + log_prior[0]
    sum[1] = sum[1] + log_prior[1]
    
#     print(log_prior)

# Determining the class of the sentence
    if sum[1] > sum[0]:
        return 1
    else:
        return 0
            
predictsentence("bad")


# In[6]:

# This function predicts the class of a document.
# It utilises the function predictSentence above to predict individual sentences in a text file
def predictDocKnownLabels(testdoc, results):
    computed = []
    knownLabel = []
    
    with open (testdoc, "r") as openedTestdoc,open (results, "w", newline = "") as openedresultdoc:
            for line in openedTestdoc:
                lineSplitFormat = ''.join(line).lower()
                lineSplitFormat = re.sub(r'[,.:!+<>&$?;""()''/|]', '', lineSplitFormat)
                
#               this splits by tab and strips the newline character and return a list of reviews and their labels
                x= lineSplitFormat.strip('\n').split('\t')
#                 print(x)
                
#               append the label of the various reviews as an integer and append to my knownLabel list
                knownLabel.append(int(x[1]))
    
#               call the function predictSentence and pass in only the reviews
                label = predictsentence(x[0])
            
#               append the predicted  labels to the list computed
                computed.append(label)
    
#               write to the results file the predicted labels
                openedresultdoc.write(str(label) + "\n")
            
#             print(knownLabel)
#             print(computed)
            return computed, knownLabel
                
knownLabel,computed  = predictDocKnownLabels("yelp_labelled.txt", "results.txt")   


# In[7]:

# This function calculates for the accuracy of the predictions
# This builds up on the function predictUnknown

def accuracy(knownLabel, computed):
    correct = 0
    for i in range(len(knownLabel)):
        if knownLabel[i] == computed[i]:
            correct+=1
            
#print statement 
    accuracy = round((correct/ len(knownLabel)) *100, 2) 
#     print("Accuracy:" , accuracy )
            
    return accuracy
                
accuracy(knownLabel, computed)


# In[8]:

# This function predicts the class of a document with unknown labels.
# It utilises the function predictSentence above to predict individual sentences in a text file
def predictDocUnknownLabels(testdoc, results):
    
    with open (testdoc, "r") as openedTestdoc,open (results, "w", newline = "") as openedresultdoc:
            for line in openedTestdoc:
                lineSplitFormat = ''.join(line).lower()
#                 print(lineSplitFormat)
                lineSplitFormat = re.sub(r'[,.:!+<>&$?;""()''/|]', '', lineSplitFormat)
                
#               call the function predictSentence and pass in only the reviews
                label = predictsentence(lineSplitFormat)
#                 print(label)
    
#               write to the results file the predicted labels
                openedresultdoc.write(str(label) + "\n")

# To test this Naive Bayes classifier, relapce testdoc.txt with intended test file               
predictDocUnknownLabels("testdoc.txt", "results.txt")   


# In[ ]:




# In[ ]:



